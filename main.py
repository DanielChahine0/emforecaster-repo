import os
import json
import socket
import argparse
import sys
import importlib
from datetime import datetime
import torch
import torch.multiprocessing as mp
from dotenv import load_dotenv
from rich.console import Console
from rich.pretty import pprint
import warnings
from pydantic import BaseModel
from typing import Dict, Any
# Set the environment variable for OpenMP
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

# Filter out the specific OpenMP warning
warnings.filterwarnings("ignore", message="OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.")
warnings.filterwarnings("ignore", message="h5py not installed, hdf5 features will not be supported.")

def update_global_config(ablation_config: Dict[str, Any], global_config: BaseModel, ablation_id: int) -> BaseModel:
    """
    Updates the global config for ablation studies and hyperparameter tuning. For example,
    if the ablation_config is {'global_config.sl.lr': 0.01}, then the global_config.sl.lr will be
    updated to 0.01.
    """

    for key, value in ablation_config.items():
        parts = key.split('.')
        if len(parts) == 2:
            sub_model, param = parts
            if hasattr(global_config, sub_model):
                sub_config = getattr(global_config, sub_model)
                if hasattr(sub_config, param):
                    setattr(sub_config, param, value)
                else:
                    print(f"Warning: {sub_model} does not have attribute {param}")
            else:
                print(f"Warning: global_config does not have attribute {sub_model}")
        elif len(parts) == 3:
            model, sub_model, param = parts
            if model == "global" and hasattr(global_config, sub_model):
                sub_config = getattr(global_config, sub_model)
                if hasattr(sub_config, param):
                    setattr(sub_config, param, value)
                else:
                    print(f"Warning: {sub_model} does not have attribute {param}")
            else:
                print(f"Warning: Invalid key format or 'global' not specified: {key}")
        else:
            print(f"Warning: Invalid key format: {key}")

    global_config.exp.ablation_id = ablation_id

    return global_config

def main(job_name="test", ablation=None, ablation_id=1, quick=False):

    # Rich console
    load_dotenv()
    console = Console()

    # Select the setup from the job name. Multivariate jobs (rf_emf_mv/*) live in
    # the forecasting.multivariate package; everything else in forecasting.univariate.
    # Both setups coexist under forecasting/ and are chosen per run.
    package = (
        "forecasting.multivariate"
        if job_name.startswith("rf_emf_mv")
        else "forecasting.univariate"
    )
    load_config = importlib.import_module(f"{package}.config.config").load_config
    Experiment = importlib.import_module(f"{package}.exp.exp").Experiment

    # Load experimental configuration. Split the dotted package into path parts so
    # the config dir resolves to forecasting/<setup>/jobs/exp/... on disk.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    args_path = os.path.join(
        base_dir, *package.split("."), "jobs", "exp", job_name, "args.yaml"
    )
    args = load_config(args_path)
    # Auto-adapt device / dataloader / compile / batch_size to this machine so
    # the same YAML runs on CUDA, MPS, or CPU without per-machine edits.
    from forecasting import hardware
    hardware.normalize_config(args)
    if ablation is not None:
        args = update_global_config(ablation, args, ablation_id)
    console.print("Pydantic Configuration Loaded Successfully:", style="bold green")

    # Run experiment on multiple seeds (optional)
    seed_list = args.exp.seed_list

    # Quick mode: run a single seed for fast iteration
    if quick:
        seed_list = seed_list[:1]
        console.print(
            f"Quick mode: running 1 seed ({seed_list[0]}) instead of {len(args.exp.seed_list)}.",
            style="bold yellow",
        )

    for i in range(len(seed_list)):
        args.exp.seed = seed_list[i]
        args.exp.time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Print args
        pprint(args, expand_all=True)

        # Initialize experiment
        exp = Experiment(args)


        console.log("Using single device")
        exp.run()

def prompt_setup():
    """Ask the user which grid to run. Returns a key of forecasting.grids.GRIDS."""
    while True:
        ans = input(
            "Run [u]nivariate (one model per band), [m]ultivariate (one model per "
            "cell), [s]eries (DET/PTV single-series), or [h]ybrid (k-cluster "
            "channel grouping) grid? "
        ).strip().lower()
        if ans in ("u", "uni", "univariate"):
            return "univariate"
        if ans in ("m", "mv", "multi", "multivariate"):
            return "multivariate"
        if ans in ("s", "series", "univariate_series"):
            return "univariate_series"
        if ans in ("h", "hybrid"):
            return "hybrid"
        print("Please enter 'u', 'm', 's', or 'h'.")

if __name__ == "__main__":
    # Non-hyperparameter tuning
    warnings.filterwarnings("ignore", message="h5py not installed, hdf5 features will not be supported.")
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("job_name", type=str, nargs="?", default=None,
                        help="Name of a single job. Omit to run a full grid.")
    parser.add_argument("--grid", action="store_true",
                        help="Run a full model x dataset x window grid instead of one job.")
    parser.add_argument("--setup",
                        choices=["univariate", "multivariate", "univariate_series", "hybrid"],
                        default=None,
                        help="Grid setup. 'univariate' trains one model per frequency "
                             "band on the multivariate datasets and averages the bands; "
                             "'multivariate' trains one model per cell over all bands; "
                             "'univariate_series' is the DET/PTV single-series sweep; "
                             "'hybrid' trains one model per cell with a k-cluster "
                             "channel-grouping head, sweeping k in addition to model/"
                             "dataset/window. If omitted in grid mode, you are prompted.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Grid mode only: list cells and targets without training.")
    parser.add_argument("--jobs", type=int, default=8,
                        help="Per-channel grid only: how many band trainings run "
                             "concurrently (default 8). Lower it if the GPU OOMs.")
    parser.add_argument("--window", type=str, default=None,
                        help="Grid mode only: restrict to specific window(s) "
                             "instead of all four. Format 'LxO' (e.g. '336x96') "
                             "or 'L336_O96'; comma-separate for several. Omit to "
                             "run every window.")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Grid mode only: how many seeds to train each cell "
                             "on (default 1). A count counts up from 69421, so "
                             "'--seeds 3' runs 69421, 69422, 69423 and reuses "
                             "whatever is already complete at 69421. A comma-"
                             "separated list (e.g. '69422,69423') selects those "
                             "seeds exactly.")
    parser.add_argument("--venv_test", action="store_true", help="Flag for venv test")
    parser.add_argument("--ablation", type=str, default=None, help="Ablation study configuration")
    parser.add_argument("--ablation_id", type=int, default=1, help="Ablation study configuration")
    parser.add_argument("--quick", action="store_true", help="Run a single seed for fast iteration")

    args = parser.parse_args()

    if args.venv_test:
        print("Running in venv test mode")
        # Add any venv test specific code here
    elif args.grid or args.job_name is None:
        from forecasting.grids import GRIDS, select_seeds, select_windows
        setup = args.setup or prompt_setup()
        # Validate --window/--seeds against this grid up front so a typo is a clean
        # error, not a mid-run traceback (or a silently empty sweep).
        try:
            windows = select_windows(args.window, GRIDS[setup]["windows"])
            seeds = select_seeds(args.seeds)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(2)
        # A per-channel grid fans each cell out into one run per band, so it needs
        # the process-pool driver rather than the in-process cell loop.
        if GRIDS[setup].get("per_channel"):
            from forecasting.perchannel import run_per_channel_grid
            failed = run_per_channel_grid(setup=setup, jobs=args.jobs,
                                          dry_run=args.dry_run, windows=windows,
                                          seeds=seeds)
        else:
            from forecasting.grid import run_grid
            failed = run_grid(setup, dry_run=args.dry_run, windows=windows,
                              seeds=seeds)
        sys.exit(1 if failed else 0)
    else:
        if args.ablation is not None:
            try:
                ablation = json.loads(args.ablation)
                main(job_name=args.job_name, ablation=ablation, ablation_id=args.ablation_id, quick=args.quick)
            except json.JSONDecodeError:
                print("Error: Invalid JSON string for ablation configuration")
                sys.exit(1)
        else:
            main(args.job_name, quick=args.quick)
