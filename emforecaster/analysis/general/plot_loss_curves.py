"""
Plot loss-vs-epoch curves (train / val) with the final test loss overlaid,
for one or more models, reading the offline ``log.json`` files written under
``logs/``.

Each run directory looks like ``logs/{ablation_id}_{ModelId}_{id}/{seed}/log.json``.
A ``log.json`` stores per-epoch lists:

    sl_train/loss   list[float]   one value per epoch
    sl_val/loss     list[float]   one value per epoch
    sl_val/mae      list[float]   one value per epoch
    sl_test/loss    list[float]   single value (best-model test, evaluated once)
    sl_test/mae     list[float]   single value

Because test is evaluated only once on the best-validation checkpoint, there is
no per-epoch test curve. We therefore draw the final test loss as a horizontal
reference line and mark the best-validation epoch (the checkpoint test was run on).

When a model has multiple seeds, curves are aggregated across seeds as a mean
with a +/-1 std shaded band. Early stopping can make seeds end at different
epochs, so aggregation is NaN-aware (shorter seeds contribute NaN past their end).

Usage
-----
    # Aggregate across all seeds, one figure per model, MSE loss:
    python emforecaster/analysis/general/plot_loss_curves.py \
        --models DLinear PatchTST TSMixer TimesNet

    # A single combined comparison figure instead of one-per-model:
    python emforecaster/analysis/general/plot_loss_curves.py --combined

    # Plot MAE instead of (or alongside) the loss:
    python emforecaster/analysis/general/plot_loss_curves.py --metric both

    # Pin an exact run directory and a single seed:
    python emforecaster/analysis/general/plot_loss_curves.py \
        --run-dir logs/1_DLinear_FbvXK7LEPR --seed 69425
"""

import os
import re
import json
import glob
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no display needed; write PNGs
import matplotlib.pyplot as plt


def find_run_dir(logs_dir, model, run_dir=None):
    """Return the run directory for ``model``.

    If ``run_dir`` is given it is used directly. Otherwise the most recently
    modified directory matching ``{ablation_id}_{model}_{id}`` is chosen.
    """
    if run_dir is not None:
        return run_dir if os.path.isdir(run_dir) else None

    pattern = re.compile(rf"^\d+_{re.escape(model)}_\w+$")
    candidates = [
        d
        for d in glob.glob(os.path.join(logs_dir, "*"))
        if os.path.isdir(d) and pattern.match(os.path.basename(d))
    ]
    if not candidates:
        return None
    # Most recently modified wins (latest run for this model).
    return max(candidates, key=os.path.getmtime)


def load_seed_logs(run_dir, seed=None):
    """Load ``log.json`` for each seed under ``run_dir``.

    Returns a dict ``{seed_name: log_dict}``.
    """
    logs = {}
    seed_dirs = (
        [os.path.join(run_dir, str(seed))]
        if seed is not None
        else sorted(glob.glob(os.path.join(run_dir, "*")))
    )
    for sd in seed_dirs:
        log_path = os.path.join(sd, "log.json")
        if os.path.isfile(log_path):
            with open(log_path) as f:
                logs[os.path.basename(sd)] = json.load(f)
    return logs


def stack_ragged(curves):
    """Stack a list of 1D arrays of differing length into a (n_seeds, max_len)
    array, padding the tail of shorter curves with NaN."""
    if not curves:
        return np.empty((0, 0))
    max_len = max(len(c) for c in curves)
    out = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        out[i, : len(c)] = c
    return out


def aggregate(logs, key):
    """Return (epochs, mean, std, n_per_epoch) for a per-epoch metric ``key``
    aggregated across seeds. ``std`` is 0 where only one seed contributes."""
    curves = [np.asarray(d[key], dtype=float) for d in logs.values() if key in d]
    stacked = stack_ragged(curves)
    if stacked.size == 0:
        return None
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
    n_per_epoch = np.sum(~np.isnan(stacked), axis=0)
    epochs = np.arange(1, len(mean) + 1)
    return epochs, mean, std, n_per_epoch


def scalar_mean_std(logs, key):
    """Mean and std of a scalar metric (stored as a length-1 list) across seeds."""
    vals = [float(d[key][0]) for d in logs.values() if key in d and len(d[key])]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


# Metric -> (per-epoch train key, per-epoch val key, test scalar key, axis label)
METRICS = {
    "loss": ("sl_train/loss", "sl_val/loss", "sl_test/loss", "MSE loss"),
    "mae": (None, "sl_val/mae", "sl_test/mae", "MAE"),
}


def plot_metric(ax, logs, model, metric):
    """Draw train/val curves (+ test reference line) for one metric on ``ax``."""
    train_key, val_key, test_key, ylabel = METRICS[metric]

    # Train (only loss is logged per-epoch for training).
    if train_key is not None:
        agg = aggregate(logs, train_key)
        if agg is not None:
            epochs, mean, std, _ = agg
            (line,) = ax.plot(epochs, mean, label="train", color="#1f77b4", lw=2)
            if len(logs) > 1:
                ax.fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=0.15)

    # Validation.
    val_agg = aggregate(logs, val_key)
    best_epoch = None
    if val_agg is not None:
        epochs, mean, std, _ = val_agg
        (line,) = ax.plot(epochs, mean, label="val", color="#ff7f0e", lw=2)
        if len(logs) > 1:
            ax.fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=0.15)
        # Best-val epoch = the checkpoint test was computed on.
        best_epoch = int(np.nanargmin(mean)) + 1
        ax.axvline(best_epoch, color="#ff7f0e", ls=":", lw=1, alpha=0.7)

    # Test (single final scalar) -> horizontal reference line.
    test_stat = scalar_mean_std(logs, test_key)
    if test_stat is not None:
        tmean, tstd, n = test_stat
        label = f"test (final) = {tmean:.4f}"
        if n > 1:
            label += f" ± {tstd:.4f}"
        ax.axhline(tmean, color="#2ca02c", ls="--", lw=1.8, label=label)

    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    title = model
    if best_epoch is not None:
        title += f"  (best val @ epoch {best_epoch})"
    ax.set_title(title)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs-dir", default="logs", help="Directory holding run subfolders (default: logs)")
    p.add_argument("--models", nargs="+", default=["DLinear", "PatchTST", "TSMixer", "TimesNet"],
                   help="Model ids to plot (matched against logs/<id>_<Model>_<hash>)")
    p.add_argument("--run-dir", default=None,
                   help="Explicit run directory (overrides auto-discovery; use with a single --models entry)")
    p.add_argument("--seed", type=int, default=None,
                   help="Plot a single seed instead of aggregating across all seeds")
    p.add_argument("--metric", choices=["loss", "mae", "both"], default="loss",
                   help="Which metric to plot (default: loss)")
    p.add_argument("--out", default="comparison", help="Output directory for PNGs (default: comparison)")
    p.add_argument("--combined", action="store_true",
                   help="One combined figure with a subplot per model instead of one figure each")
    p.add_argument("--logy", action="store_true", help="Use a log scale on the y-axis")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    metrics = ["loss", "mae"] if args.metric == "both" else [args.metric]

    # Resolve which models actually have logs.
    resolved = []
    for model in args.models:
        rd = find_run_dir(args.logs_dir, model, args.run_dir if len(args.models) == 1 else None)
        if rd is None:
            print(f"[skip] {model}: no run directory found under {args.logs_dir}/")
            continue
        logs = load_seed_logs(rd, args.seed)
        if not logs:
            print(f"[skip] {model}: no log.json found in {rd}")
            continue
        print(f"[ok]   {model}: {os.path.basename(rd)}  ({len(logs)} seed(s): {', '.join(logs)})")
        resolved.append((model, logs))

    if not resolved:
        print("Nothing to plot. Run some experiments first (python main.py rf_emf/<model>).")
        return

    if args.combined:
        nrows, ncols = len(resolved), len(metrics)
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
        for r, (model, logs) in enumerate(resolved):
            for c, metric in enumerate(metrics):
                ax = axes[r][c]
                plot_metric(ax, logs, model, metric)
                if args.logy:
                    ax.set_yscale("log")
        fig.tight_layout()
        out_path = os.path.join(args.out, f"loss_curves_combined_{args.metric}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved {out_path}")
    else:
        for model, logs in resolved:
            fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 4.5), squeeze=False)
            for c, metric in enumerate(metrics):
                ax = axes[0][c]
                plot_metric(ax, logs, model, metric)
                if args.logy:
                    ax.set_yscale("log")
            fig.tight_layout()
            out_path = os.path.join(args.out, f"loss_curve_{model}_{args.metric}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
