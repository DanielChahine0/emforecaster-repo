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
