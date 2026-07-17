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
from forecasting.run_path import build_run_dir

# Last conformal metric written to log.json; its presence means train + test +
# conformal recalibration all finished for a seed.
MARKER = "sl_test/interval_width_mean"
SEED = 69421

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_modules(setup):
    pkg = f"forecasting.{setup}"
    load_config = importlib.import_module(f"{pkg}.config.config").load_config
    Experiment = importlib.import_module(f"{pkg}.exp.exp").Experiment
    return load_config, Experiment


def args_path(setup, job):
    return os.path.join(_REPO_ROOT, "forecasting", setup, "jobs", "exp",
                        *job.split("/"), "args.yaml")


def cell_done(setup, model_id, dataset, seq_len, pred_len, seeds=(SEED,),
              logs_dir="logs"):
    run_dir = build_run_dir(setup, model_id, dataset, seq_len, pred_len,
                            conf_on=True, logs_dir=logs_dir)
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


def run_grid(setup, dry_run=False, logs_dir="logs"):
    """Run every cell of the grid for one setup. Returns the list of failed cells."""
    console = Console()
    grid_def = GRIDS[setup]
    load_config, Experiment = setup_modules(setup)
    passed, skipped, failed = [], [], []
    cells = [(m, d, w)
             for m in grid_def["models"]
             for d in grid_def["datasets"]
             for w in grid_def["windows"]]
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
