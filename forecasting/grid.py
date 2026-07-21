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
from forecasting.grids import GRIDS
from forecasting.run_path import build_run_dir, PER_CHANNEL_SETUP

# Last conformal metric written to log.json; its presence means train + test +
# conformal recalibration all finished for a seed.
MARKER = "sl_test/interval_width_mean"
SEED = 69421

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
              logs_dir="logs"):
    run_dir = build_run_dir(LOG_SETUP.get(setup, setup), model_id, dataset,
                            seq_len, pred_len, conf_on=True, logs_dir=logs_dir)
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


def run_grid(setup, dry_run=False, logs_dir="logs", windows=None):
    """Run every cell of the grid for one setup. Returns the list of failed cells.

    logs_dir only affects the resume check; Experiment.run() always writes under
    the default logs/ tree, so pass a custom logs_dir for dry-run/testing only.
    windows overrides the grid's window list (already validated by the caller);
    None runs all of them.
    """
    console = Console()
    grid_def = GRIDS[setup]
    windows = grid_def["windows"] if windows is None else windows
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
    cells = [(m, d, w)
             for m in grid_def["models"]
             for d in grid_def["datasets"]
             for w in windows]
    total = len(cells)

    for n, ((label, model_id, job), dataset, (seq_len, pred_len)) in enumerate(cells, 1):
        tag = f"[{n:2d}/{total}] {label:<12} {dataset}/L{seq_len}/O{pred_len}"

        if cell_done(setup, model_id, dataset, seq_len, pred_len, logs_dir=logs_dir):
            console.log(f"[SKIP] {tag}")
            skipped.append(tag)
            continue

        if dry_run:
            console.log(f"[DRY ] {tag} -> {job} dataset={dataset} L={seq_len} O={pred_len}")
            passed.append(tag)
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
            args.exp.seed_list = [SEED]
            args.exp.seed = SEED
            args.exp.time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            console.log(f"[RUN ] {tag}")
            Experiment(args).run()
            console.log(f"[ OK ] {tag}")
            passed.append(tag)
        except Exception as exc:  # keep the grid alive on a bad cell
            console.log(f"[FAIL] {tag} ({type(exc).__name__}: {exc})")
            console.print_exception()
            failed.append(tag)
        finally:
            _cleanup()

    console.log("")
    console.log(f"===== {setup} CP grid summary =====")
    console.log(f"  passed:  {len(passed)}")
    console.log(f"  skipped: {len(skipped)} (already complete)")
    console.log(f"  failed:  {len(failed)}")
    for c in failed:
        console.log(f"    - {c}")
    return failed
