"""In-process driver for the conformal-prediction grid.

Replaces the bash runner: no sed, no subprocess. For each cell it loads the
model's config, overrides the window/protocol fields in memory, applies the
hardware profile, and calls Experiment(args).run(). Resume state is shared with
the shell runners byte-for-byte via build_run_dir + the sl_test marker.
"""
import gc
import importlib
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from rich.console import Console

from forecasting import hardware
from forecasting.grids import GRIDS, SEED
from forecasting.run_path import build_run_dir, PER_CHANNEL_SETUP

# Last conformal metric written to log.json; its presence means train + test +
# conformal recalibration all finished for a seed.
MARKER = "sl_test/interval_width_mean"

# SEED is defined in forecasting/grids.py (beside the seed selection --seeds
# uses) and re-exported here, where perchannel.py and the tests already read it.

# EMForecaster's patch pipeline needs ~29.5 GiB of activations for a 512 batch at
# seq_len 336 -- more than the shared 31.35 GiB card -- and its emf2023 cells (29
# channels, 55k train rows) OOM whenever a co-tenant process holds a few GiB.
# Those cells train on smaller micro-batches with gradient accumulation, keeping
# the effective batch (and the onecycle schedule) identical to every other cell.
OOM_RETRIES = 2


def _memory_profile(model_id, dataset, seq_len, cluster_k=None):
    """(batch_size, accum_steps) override for known memory-heavy cells, or None."""
    if cluster_k is not None and model_id == "EMForecaster":
        # ClusteredHead's per-cluster Linear(size*F, size*pred_len) scales with
        # cluster-size^2, so at small k the head alone can dwarf the rest of the
        # model -- but it shrinks back out of the way as k grows, and only the
        # small-k end needs the 8-way micro-step split (measured peak on a
        # 27-channel L336_O96 cell: 15.6 GiB at k=2 vs 7.9 GiB at k=9). The
        # effective batch is 512 either way: exp.py scales the loss by
        # 1/accum_steps and steps the scheduler once per optimizer step, so
        # batch x accum is an equivalence class and this is purely a speed knob,
        # not a change to the optimization trajectory.
        #
        # This only helps activation memory -- if a cell still OOMs after
        # retries, the head's weights/optimizer state (not activations) are
        # likely the cause, and the fix is excluding that cell from the sweep
        # (see forecasting/grids.py's GRIDS["cluster"] k_values note), not a
        # smaller batch size.
        return (64, 8) if cluster_k <= 3 else (128, 4)
    # LSTM is deliberately absent from the clause above: its clustered head peaks
    # at 0.4 GiB even at batch 64 / k=9, so clamping it only bought micro-steps
    # (24.1 vs 12.5 ms per 512 examples at batch 64 vs the config's 512).
    if model_id == "EMForecaster" and (
        seq_len >= 336 or dataset == "EMF_2023_university_5_with_PTV"
    ):
        return 128, 4
    return None

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Grid name -> the package owning its configs and Experiment class. Two grids do
# not match their own name: univariate_series still runs the original
# forecasting.univariate package (it was renamed when "univariate" became the
# per-band grid), and the per-band grid reuses forecasting.multivariate, since it
# is the same models and job configs run one band at a time.
SETUP_PACKAGE = {
    "multivariate": "multivariate",
    "univariate": "multivariate",
    "univariate_series": "univariate",
    "cluster": "multivariate",
}

# Grid name -> the <setup> folder its runs land in under logs/. The names differ
# from the grid keys: univariate_series is the single-series package, which
# writes logs/univariate/, while the per-band "univariate" grid writes
# logs/univariate (per channel)/ (that entry is only for completeness -- the
# per-band grid runs through forecasting/perchannel.py, not run_grid).
LOG_SETUP = {
    "multivariate": "multivariate",
    "univariate": PER_CHANNEL_SETUP,
    "univariate_series": "univariate",
    "cluster": "cluster",
}


def _package(setup):
    return SETUP_PACKAGE.get(setup, setup)


def setup_modules(setup):
    pkg = f"forecasting.{_package(setup)}"
    load_config = importlib.import_module(f"{pkg}.config.config").load_config
    Experiment = importlib.import_module(f"{pkg}.exp.exp").Experiment
    return load_config, Experiment


def args_path(setup, job):
    return os.path.join(_REPO_ROOT, "forecasting", _package(setup), "jobs", "exp",
                        *job.split("/"), "args.yaml")


def cell_done(setup, model_id, dataset, seq_len, pred_len, seeds=(SEED,),
              logs_dir="logs", tag=""):
    run_dir = build_run_dir(LOG_SETUP.get(setup, setup), model_id, dataset,
                            seq_len, pred_len, conf_on=True, logs_dir=logs_dir,
                            tag=tag)
    for seed in seeds:
        p = os.path.join(run_dir, str(seed), "log.json")
        try:
            with open(p) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        if MARKER not in data:
            return False
    return True


def _cleanup():
    # In-process hygiene the subprocess model got for free: close figures, drop
    # references, and bound the Inductor cache when compile is on.
    plt.close("all")
    gc.collect()
    try:
        torch._dynamo.reset()
    except Exception:
        pass


def run_grid(setup, dry_run=False, logs_dir="logs", windows=None, seeds=None,
             k_values=None):
    """Run every cell of the grid for one setup. Returns the list of failed runs.

    logs_dir only affects the resume check; Experiment.run() always writes under
    the default logs/ tree, so pass a custom logs_dir for dry-run/testing only.
    windows overrides the grid's window list (already validated by the caller);
    None runs all of them. seeds is the list of seeds each cell is trained on
    (also caller-validated); None means just SEED, the pre---seeds behaviour.
    k_values overrides a cluster grid's cluster counts (also caller-validated);
    None means the grid's own k_values, i.e. the pre---k behaviour.
    """
    console = Console()
    grid_def = GRIDS[setup]
    windows = grid_def["windows"] if windows is None else windows
    seeds = [SEED] if seeds is None else list(seeds)
    if grid_def.get("per_channel"):
        # A per-band cell is many runs, not one, and needs the process pool in
        # forecasting/perchannel.py. Fail loudly instead of silently training the
        # joint model the caller did not ask for.
        raise ValueError(
            f"'{setup}' is a per-channel grid; run it via "
            f"forecasting.perchannel.run_per_channel_grid "
            f"(python main.py --grid --setup {setup})."
        )
    load_config, Experiment = setup_modules(setup)
    passed, skipped, failed = [], [], []
    grid_k = grid_def.get("k_values")
    k_values = grid_k if k_values is None else list(k_values)
    # --k may ask for a k outside the declared sweep on purpose (k=1 and
    # k=num_channels are excluded from GRIDS["cluster"] as one-off sanity checks).
    # Say so, so an off-grid cell is never mistaken for part of the sweep when
    # the results are read back later.
    off_grid = [k for k in (k_values or []) if k not in (grid_k or [])]
    if off_grid:
        console.log(f"[NOTE ] k={', '.join(str(k) for k in off_grid)} outside the "
                    f"{setup} grid's k_values {grid_k}; running as a one-off")
    if k_values:
        cells = [(m, d, w, k)
                 for m in grid_def["models"]
                 for d in grid_def["datasets"]
                 for w in windows
                 for k in k_values]
    else:
        cells = [(m, d, w, None)
                 for m in grid_def["models"]
                 for d in grid_def["datasets"]
                 for w in windows]
    # Seed-major, not cell-major: an interrupted multi-seed sweep then leaves a
    # complete grid at one seed rather than the first few cells with error bars
    # and the rest missing entirely. Resume is per (cell, seed) either way, so
    # restarting picks up exactly where it stopped.
    runs = [(cell, seed) for seed in seeds for cell in cells]
    total = len(runs)

    for n, (((label, model_id, job), dataset, (seq_len, pred_len), k), seed) in enumerate(runs, 1):
        k_suffix = f" k={k}" if k is not None else ""
        # Label the seed only when there is more than one, so an ordinary sweep's
        # log lines do not all carry the same constant.
        seed_suffix = f" seed={seed}" if len(seeds) > 1 else ""
        cell_tag = (f"[{n:2d}/{total}] {label:<12} "
                    f"{dataset}/L{seq_len}/O{pred_len}{k_suffix}{seed_suffix}")
        run_tag = f"k{k}" if k is not None else ""

        if cell_done(setup, model_id, dataset, seq_len, pred_len, seeds=(seed,),
                     logs_dir=logs_dir, tag=run_tag):
            console.log(f"[SKIP] {cell_tag}")
            skipped.append(cell_tag)
            continue

        if dry_run:
            console.log(f"[DRY ] {cell_tag} -> {job} dataset={dataset} L={seq_len} O={pred_len}")
            passed.append(cell_tag)
            continue

        try:
            args = load_config(args_path(setup, job))
            hardware.normalize_config(args)
            args.data.dataset = dataset
            args.data.seq_len = seq_len
            args.data.pred_len = pred_len
            args.conf.conf = True
            args.conf.alpha = grid_def["alpha"]
            args.exp.neptune = False
            args.exp.id = ""
            args.exp.ablation_id = 1
            args.exp.seed_list = [seed]
            args.exp.seed = seed
            args.exp.time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if k is not None:
                # Must come after the args.exp.id = "" reset above, or different
                # k's would collide in the same log directory.
                args.cluster.enabled = True
                args.cluster.k_clusters = k
                args.exp.id = run_tag
            profile = _memory_profile(model_id, dataset, seq_len, cluster_k=k)
            if profile is not None:
                args.sl.batch_size, args.sl.accum_steps = profile
                console.log(
                    f"[MEM ] {cell_tag} batch_size={args.sl.batch_size} "
                    f"x accum {args.sl.accum_steps}"
                )
            console.log(f"[RUN ] {cell_tag}")
            for attempt in range(OOM_RETRIES + 1):
                try:
                    Experiment(args).run()
                    break
                except torch.cuda.OutOfMemoryError:
                    # A cell must never silently vanish on OOM: free the cache and
                    # retry on smaller micro-batches with the same effective batch.
                    if attempt == OOM_RETRIES:
                        raise
                    args.sl.batch_size = max(1, args.sl.batch_size // 2)
                    args.sl.accum_steps *= 2
                    _cleanup()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    console.log(
                        f"[OOM ] {cell_tag} retrying with batch_size="
                        f"{args.sl.batch_size} x accum {args.sl.accum_steps}"
                    )
            console.log(f"[ OK ] {cell_tag}")
            passed.append(cell_tag)
        except Exception as exc:  # keep the grid alive on a bad cell
            console.log(f"[FAIL] {cell_tag} ({type(exc).__name__}: {exc})")
            console.print_exception()
            failed.append(cell_tag)
        finally:
            _cleanup()

    console.log("")
    console.log(f"===== {setup} CP grid summary =====")
    console.log(f"  passed:  {len(passed)}")
    console.log(f"  skipped: {len(skipped)} (already complete)")
    console.log(f"  failed:  {len(failed)}")
    for c in failed:
        console.log(f"    - {c}")

    # Persist the outcome next to the logs: a crashed cell used to leave nothing
    # but its args.yaml behind, so a whole model could silently drop out of the
    # results table. The report survives the console scrollback.
    report = {
        "setup": setup,
        "finished": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "seeds": seeds,
        # None on every non-cluster grid; recorded so a --k run's report says which
        # cluster counts it actually covered rather than implying the full sweep.
        "k_values": k_values,
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
    }
    if not dry_run:
        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, f"grid_report_{setup}.json"), "w") as f:
            json.dump(report, f, indent=2)
    return failed
