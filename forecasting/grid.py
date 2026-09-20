"""Driver for the conformal-prediction grid.

Replaces the bash runner: no sed. For each cell it loads the model's config,
overrides the window/protocol fields in memory, applies the hardware profile, and
calls Experiment(args).run(). Resume state is shared with the shell runners
byte-for-byte via build_run_dir + the sl_test marker.

Two execution modes, same cells and same outputs either way:

- jobs=1 (default) runs every cell in this process, one after another. Lowest
  overhead and the easiest to read a traceback out of, so it stays the default.
- jobs>1 runs each cell in its own `main.py <job> --ablation ...` subprocess,
  `jobs` at a time, mirroring forecasting/perchannel.py. Read run_grid's `jobs`
  note before reaching for it: the measured payoff on this hardware is much
  smaller than the free VRAM suggests.
"""
import gc
import importlib
import json
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
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


def _memory_profile(model_id, dataset, seq_len):
    """(batch_size, accum_steps) override for known memory-heavy cells, or None.

    No cluster-grid entry, deliberately. CCM adds k parallel per-cluster layers,
    so cost does grow with k, but from a small base: measured backward peak at 27
    channels / L336_O96 / batch 512, k=2 -> k=9, was DLinear 1.05 -> 1.26 GiB,
    PatchTST 3.15 -> 3.19, TSMixer 1.54 -> 1.82 against a 31.3 GiB card
    (measured on the retired upstream-network implementation; the native one has
    the same layer structure, so the shape of the growth carries over). Every
    cluster cell therefore runs at its job config's own batch size, matching the
    multivariate cell it is compared against, and OOM_RETRIES still covers a
    co-tenant squeezing the card.
    """
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


def cell_ablation(model_id, dataset, seq_len, pred_len, alpha, seed, k=None,
                  run_tag="", workers=None, concurrency=1):
    """The --ablation payload that makes `main.py <job>` run exactly one cell.

    Mirrors the in-process overrides in run_grid field for field, so a jobs>1 run
    and a jobs=1 run produce byte-identical run directories. main() applies the
    ablation after hardware.normalize_config, so anything set here wins over the
    per-machine profile -- which is what lets exp.compile be forced off below.
    """
    ablation = {
        "global.data.dataset": dataset,
        "global.data.seq_len": seq_len,
        "global.data.pred_len": pred_len,
        "global.conf.conf": True,
        "global.conf.alpha": alpha,
        "global.exp.neptune": False,
        # Without a distinct tag every k would overwrite the same run directory.
        "global.exp.id": run_tag,
        "global.exp.ablation_id": 1,
        "global.exp.seed_list": [seed],
        "global.exp.seed": seed,
        # Every benchmark cell runs eager. hardware.normalize_config turns
        # torch.compile on for PatchTST and the Transformer on CUDA, and a
        # compiled run is not run-to-run deterministic (~0.5% drift measured on
        # the Transformer), so the compiled multivariate PatchTST cell could
        # never be matched by its eager cluster twin (0.5439 vs 0.5447 on
        # ED2022 L336/O96). CCM additionally cannot be traced at all: it
        # rebinds module attributes inside forward. Contract C5 in
        # .personal/PLAN.md.
        "global.exp.compile": False,
        # How many trainings share the GPU, stamped into the run so a timing
        # is only compared under the same contention (contract C7).
        "global.exp.concurrency": int(concurrency),
    }
    if k is not None:
        ablation["global.cluster.enabled"] = True
        ablation["global.cluster.k_clusters"] = k
    if workers is not None:
        ablation["global.data.num_workers"] = workers
    profile = _memory_profile(model_id, dataset, seq_len)
    if profile is not None:
        ablation["global.sl.batch_size"], ablation["global.sl.accum_steps"] = profile
    return ablation


def run_cell_subprocess(job, ablation, threads=None):
    """Train one cell in its own process. Returns (ok, error_tail)."""
    cmd = [sys.executable, os.path.join(_REPO_ROOT, "main.py"), job,
           "--ablation", json.dumps(ablation)]
    env = dict(os.environ)
    if threads is not None:
        # Each worker would otherwise size its BLAS/OMP pools for the whole
        # machine, so `jobs` processes fight over the same cores.
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
    # Captured, not streamed: `jobs` processes writing to one terminal interleave
    # into unreadable noise. Only the tail of a failure is surfaced.
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True,
                          env=env)
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout or "").strip()[-1200:]


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
             k_values=None, models=None, datasets=None, jobs=1):
    """Run every cell of the grid for one setup. Returns the list of failed runs.

    logs_dir only affects the resume check; Experiment.run() always writes under
    the default logs/ tree, so pass a custom logs_dir for dry-run/testing only.
    windows overrides the grid's window list (already validated by the caller);
    None runs all of them. seeds is the list of seeds each cell is trained on
    (also caller-validated); None means just SEED, the pre---seeds behaviour.
    k_values overrides a cluster grid's cluster counts (also caller-validated);
    None means the grid's own k_values, i.e. the pre---k behaviour.
    models and datasets restrict the sweep to a subset of the grid's own (also
    caller-validated); None means all of them. Each is a plain filter -- the
    cells it keeps are byte-for-byte the cells the full sweep would have run, so
    a trial run's results drop straight into the full grid's tables.

    jobs>1 trains that many cells concurrently, each in its own subprocess (see
    the module docstring). Expect much less than a linear speedup, and on a
    GPU-heavy sweep expect almost none.

    Measured on the RTX 5090, 15 cluster cells (DLinear/PatchTST/TSMixer over the
    three datasets at L336_O96) at jobs=4 against the same cells run
    sequentially: every cell slowed by a mean 3.15x per epoch, so four workers
    netted 1.2x -- 29m23s against a ~35 min sequential projection. Per-model
    slowdown at jobs=4 was DLinear 1.0-2.7x, TSMixer 3.0-3.5x, PatchTST 2.9-3.7x.

    Free VRAM is not the signal to size this by, and neither is
    `nvidia-smi utilization.gpu`: that reports the fraction of *time* a kernel was
    resident, not occupancy, so the 52%/5-of-32-GiB reading during a sequential
    sweep did not mean half the card was spare. These models issue many small
    kernels, so concurrent processes mostly make CUDA time-slice between contexts.
    Contention tracks how GPU-bound the model is -- a DLinear-only sweep still
    parallelizes usefully, a PatchTST one does not.

    Where jobs>1 does pay: cheap models, and sweeps whose wall clock is dominated
    by the per-cell fixed cost (data loading, conformal recalibration, plotting),
    which is ~28% of a DLinear cell (38.5s training inside a 53s cell).

    Memory is a hard ceiling on top of that, sized by the *peak* of the heaviest
    cell, not the average: EMForecaster and the Transformer are one to two orders
    heavier than these (EMForecaster already needs the batch-128 profile above to
    fit one cell), so jobs>2 on a grid containing them will OOM. The default of 1
    keeps every existing invocation on the in-process path unchanged.
    """
    console = Console()
    grid_def = GRIDS[setup]
    windows = grid_def["windows"] if windows is None else windows
    seeds = [SEED] if seeds is None else list(seeds)
    models = grid_def["models"] if models is None else list(models)
    if datasets is None:
        datasets = grid_def.get("default_datasets", grid_def["datasets"])
    else:
        datasets = list(datasets)
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
    # --window is equally allowed to ask for a custom lookback/horizon outside
    # the declared sweep (select_windows passes them through on purpose).
    off_grid_w = [w for w in windows if w not in grid_def["windows"]]
    if off_grid_w:
        pretty = ", ".join(f"L{s}_O{p}" for s, p in off_grid_w)
        have = ", ".join(f"L{s}_O{p}" for s, p in grid_def["windows"])
        console.log(f"[NOTE ] window(s) {pretty} outside the {setup} grid's "
                    f"windows ({have}); running as a one-off")
    if k_values:
        cells = [(m, d, w, k)
                 for m in models
                 for d in datasets
                 for w in windows
                 for k in k_values]
    else:
        cells = [(m, d, w, None)
                 for m in models
                 for d in datasets
                 for w in windows]
    # Seed-major, not cell-major: an interrupted multi-seed sweep then leaves a
    # complete grid at one seed rather than the first few cells with error bars
    # and the rest missing entirely. Resume is per (cell, seed) either way, so
    # restarting picks up exactly where it stopped.
    runs = [(cell, seed) for seed in seeds for cell in cells]
    total = len(runs)

    if jobs > 1 and not dry_run:
        return _run_grid_parallel(setup, grid_def, runs, jobs, logs_dir, seeds,
                                  k_values, models, datasets, windows, console)

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
            # Eager for every cell, not only cluster ones -- see cell_ablation
            # for why (determinism; CCM is untraceable besides).
            args.exp.compile = False
            # This path trains one cell at a time.
            args.exp.concurrency = 1
            profile = _memory_profile(model_id, dataset, seq_len)
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

    _finish(setup, console, passed, skipped, failed, seeds, k_values, models,
            datasets, windows, logs_dir, write=not dry_run)
    return failed


def _finish(setup, console, passed, skipped, failed, seeds, k_values, models,
            datasets, windows, logs_dir, write=True):
    """Print the summary and persist the run report. Shared by both modes."""
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
        # Same reason: a --model/--dataset/--window trial run writes a report at
        # the same path as a full sweep, so the report has to say what it covered.
        "models": [model_id for _, model_id, _ in models],
        "datasets": list(datasets),
        "windows": [f"L{s}_O{p}" for s, p in windows],
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
    }
    if write:
        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, f"grid_report_{setup}.json"), "w") as f:
            json.dump(report, f, indent=2)


def _run_grid_parallel(setup, grid_def, runs, jobs, logs_dir, seeds, k_values,
                       models, datasets, windows, console):
    """run_grid's jobs>1 path: one subprocess per cell, `jobs` at a time."""
    alpha = grid_def["alpha"]
    total = len(runs)
    passed, skipped, failed = [], [], []

    # Same split perchannel.py uses: divide the machine's cores between the
    # workers so `jobs` processes do not each size their BLAS pool for all of it.
    threads = max(1, (os.cpu_count() or 8) // jobs)

    work = []
    for n, (((label, model_id, job), dataset, (seq_len, pred_len), k), seed) in enumerate(runs, 1):
        k_suffix = f" k={k}" if k is not None else ""
        seed_suffix = f" seed={seed}" if len(seeds) > 1 else ""
        cell_tag = (f"[{n:2d}/{total}] {label:<12} "
                    f"{dataset}/L{seq_len}/O{pred_len}{k_suffix}{seed_suffix}")
        run_tag = f"k{k}" if k is not None else ""
        # Resume is checked up front, single-threaded, so a completed cell never
        # occupies a worker slot and the [SKIP] lines stay in grid order.
        if cell_done(setup, model_id, dataset, seq_len, pred_len, seeds=(seed,),
                     logs_dir=logs_dir, tag=run_tag):
            console.log(f"[SKIP] {cell_tag}")
            skipped.append(cell_tag)
            continue
        work.append((cell_tag, model_id, job, dataset, seq_len, pred_len, k,
                     run_tag, seed))

    console.log(f"Running {len(work)} cells on {jobs} workers "
                f"({threads} BLAS threads each).")

    def _one(item):
        cell_tag, model_id, job, dataset, seq_len, pred_len, k, run_tag, seed = item
        ablation = cell_ablation(model_id, dataset, seq_len, pred_len, alpha,
                                 seed, k=k, run_tag=run_tag, concurrency=jobs)
        ok, err = run_cell_subprocess(job, ablation, threads=threads)
        return cell_tag, ok, err

    # Threads, not processes: each one only blocks on subprocess.run, so the GIL
    # is irrelevant and max_workers is exactly the training concurrency.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for cell_tag, ok, err in pool.map(_one, work):
            if ok:
                console.log(f"[ OK ] {cell_tag}")
                passed.append(cell_tag)
            else:
                console.log(f"[FAIL] {cell_tag}")
                console.log(err)
                failed.append(cell_tag)

    _finish(setup, console, passed, skipped, failed, seeds, k_values, models,
            datasets, windows, logs_dir)
    return failed
