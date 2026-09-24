"""Plot train/val loss-vs-epoch curves with the final test loss overlaid,
reading the ``log.json`` files under
``logs/{setup}/{model}/{dataset}/L{L}_O{O}_{protocol}/{seed}/``.

Every seed is drawn as its own thin curve, with its best validation epoch
marked on it. The bold mean runs only to the epoch the shortest seed reached,
where every seed still contributes. Averaging past it -- the NaN-padded mean
this used to draw -- is survivorship: a seed that collapsed and stopped early
drops out, and the mean of the survivors looks converged (audit RA-9: the
series Transformer at L336/O96, where most seeds collapsed). Test loss is one
scalar per seed, evaluated on that seed's best-validation checkpoint, drawn as
a horizontal line at the seed mean.
Per-run figures go to ``{out}/{dataset}/{model}/``; ``--combined`` writes one
figure per (dataset, L, O) with a subplot per model to ``{out}/{dataset}/``.
"""

import argparse
import glob
import json
import os
import sys
import pathlib
import re

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The run-directory convention lives with the training code; add the repo
# root so this works whether it is run as a script or imported as emfres. The
# typeface and palette are results/'s own (emfres.style).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forecasting.run_path import iter_run_dirs
from emfres.style import GRIDLINE, INK_SECONDARY, SERIES_COLORS, rc_fonts

# train, val, test: Charcoal Blue, Verdigris, Burnt Peach.
COLORS = {"train": SERIES_COLORS[2], "val": SERIES_COLORS[0], "test": SERIES_COLORS[1]}

# metric -> (per-epoch train key, per-epoch val key, test scalar key, axis label)
METRICS = {
    "loss": ("sl_train/loss", "sl_val/loss", "sl_test/loss", "MSE loss"),
    "mae": (None, "sl_val/mae", "sl_test/mae", "MAE"),
}


def find_run_dirs(logs_dir, model, run_dir=None, latest=False, exclude=()):
    if run_dir is not None:
        return [run_dir] if os.path.isdir(run_dir) else []
    # `model` is a directory level in the tree; baselines are logged under
    # RF_EMF_<Model>, so accept either the bare name or the RF_EMF_ prefix.
    names = {model, f"RF_EMF_{model}"}
    dirs = [d for d in iter_run_dirs(logs_dir)
            if names & set(d.split(os.sep))
            and not any(x in d for x in exclude)]
    return dirs[-1:] if latest else dirs


def load_run_config(run_dir):
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "args.yaml"))):
        with open(path) as f:
            data = (yaml.safe_load(f) or {}).get("data", {}) or {}
        return {"dataset": data.get("dataset"), "L": data.get("seq_len"), "O": data.get("pred_len")}
    return {"dataset": None, "L": None, "O": None}


def load_seed_logs(run_dir, seed=None):
    seed_dirs = ([os.path.join(run_dir, str(seed))] if seed is not None
                 else sorted(glob.glob(os.path.join(run_dir, "*"))))
    logs = {}
    for sd in seed_dirs:
        path = os.path.join(sd, "log.json")
        if os.path.isfile(path):
            with open(path) as f:
                logs[os.path.basename(sd)] = json.load(f)
    return logs


def run_label(model, cfg):
    return f"{model}  L={cfg['L']} O={cfg['O']}  [{cfg['dataset']}]"


def dataset_slug(dataset):
    """`rf_emf_det` -> `det`: the same dataset spelling frames/ uses."""
    name = str(dataset or "unknown")
    if name.lower().startswith("rf_emf_"):
        name = name[len("rf_emf_"):]
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def window_slug(cfg):
    return f"l{cfg['L']}_o{cfg['O']}"


def out_dir(base, cfg, model=None):
    """One directory per dataset; the model is a filename token, not a level.

    The old layout nested <dataset>/<Model>/ and then repeated both in the
    filename, so every leaf said its dataset three times.
    """
    path = os.path.join(base, "loss-curves", dataset_slug(cfg["dataset"]))
    os.makedirs(path, exist_ok=True)
    return path


def aggregate(logs, key):
    """(epochs, mean, std, curves) over the epochs every seed reached.

    The mean stops at the shortest seed rather than NaN-padding past it: a
    seed that stopped early would otherwise leave the mean to the seeds that
    kept going. ``curves`` holds each seed's full curve, for drawing.
    """
    curves = [np.asarray(d[key], dtype=float) for d in logs.values() if key in d]
    curves = [c for c in curves if len(c)]
    if not curves:
        return None
    common = min(len(c) for c in curves)
    stacked = np.stack([c[:common] for c in curves])
    return np.arange(1, common + 1), stacked.mean(axis=0), stacked.std(axis=0), curves


def plot_curve(ax, logs, key, label, color):
    """Each seed thin, the mean over the common epochs bold; returns the curves."""
    agg = aggregate(logs, key)
    if agg is None:
        return None
    epochs, mean, std, curves = agg
    if len(curves) > 1:
        for c in curves:
            ax.plot(np.arange(1, len(c) + 1), c, color=color, lw=0.8, alpha=0.35)
        label += f" (mean of {len(curves)} to epoch {len(epochs)})"
    ax.plot(epochs, mean, label=label, color=color, lw=2)
    return curves


def plot_metric(ax, logs, title, metric):
    train_key, val_key, test_key, ylabel = METRICS[metric]
    if train_key:
        plot_curve(ax, logs, train_key, "train", COLORS["train"])
    val_curves = plot_curve(ax, logs, val_key, "val", COLORS["val"])
    if val_curves:
        # Best validation epoch per seed, on that seed's own curve: the
        # checkpoint each seed's test value comes from.
        best = [int(np.nanargmin(c)) for c in val_curves]
        ax.plot([b + 1 for b in best], [c[b] for c, b in zip(val_curves, best)],
                marker="v", linestyle="None", color=COLORS["val"], markersize=6,
                markeredgecolor="white", markeredgewidth=0.6, label="best val (per seed)")
        title += f"  (best val @ epoch {', '.join(str(b + 1) for b in best)})"

    test_vals = [float(d[test_key][0]) for d in logs.values() if d.get(test_key)]
    if test_vals:
        label = f"test (final) = {np.mean(test_vals):.4f}"
        if len(test_vals) > 1:
            label += f" ± {np.std(test_vals):.4f}"
        ax.axhline(np.mean(test_vals), color=COLORS["test"], ls="--", lw=1.8, label=label)

    ax.set_xlabel("epoch", color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, color=GRIDLINE, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--models", nargs="+",
                   default=["EMForecaster", "DLinear", "PatchTST", "TSMixer", "LSTM", "Transformer"])
    p.add_argument("--exclude", nargs="*", default=["cptest"],
                   help="skip run dirs whose name contains any of these substrings "
                        "(default: cptest smoke-test runs; pass --exclude with no value to keep all)")
    p.add_argument("--run-dir", default=None, help="explicit run directory (use with a single model)")
    p.add_argument("--seed", type=int, default=None, help="plot one seed instead of aggregating")
    p.add_argument("--datasets", nargs="+", default=None, help="only runs trained on these datasets")
    p.add_argument("--latest", action="store_true", help="only the most recent run per model")
    p.add_argument("--metric", choices=["loss", "mae", "both"], default="loss")
    p.add_argument("--out", default="results/figures")
    p.add_argument("--combined", action="store_true",
                   help="one figure per (dataset, L, O) with a subplot per model")
    p.add_argument("--logy", action="store_true")
    args = p.parse_args()
    plt.rcParams.update(rc_fonts("sans"))

    metrics = ["loss", "mae"] if args.metric == "both" else [args.metric]

    runs = []
    for model in args.models:
        run_dirs = find_run_dirs(args.logs_dir, model,
                                 args.run_dir if len(args.models) == 1 else None, args.latest,
                                 exclude=args.exclude)
        if not run_dirs:
            print(f"[skip] {model}: no run directory found under {args.logs_dir}/")
            continue
        for rd in run_dirs:
            logs = load_seed_logs(rd, args.seed)
            if not logs:
                print(f"[skip] {model}: no log.json found in {rd}")
                continue
            cfg = load_run_config(rd)
            if args.datasets and cfg["dataset"] not in args.datasets:
                continue
            print(f"[ok]   {run_label(model, cfg)}: {os.path.basename(rd)}  "
                  f"({len(logs)} seed(s): {', '.join(logs)})")
            runs.append((model, cfg, logs))

    if not runs:
        print("Nothing to plot. Run some experiments first (python main.py rf_emf/<model>).")
        return

    def save_figure(rows, path, suptitle=None, row_h=4.0):
        fig, axes = plt.subplots(len(rows), len(metrics),
                                 figsize=(7 * len(metrics), row_h * len(rows)), squeeze=False)
        for r, (model, cfg, logs) in enumerate(rows):
            for c, metric in enumerate(metrics):
                plot_metric(axes[r][c], logs, run_label(model, cfg), metric)
                if args.logy:
                    axes[r][c].set_yscale("log")
        if suptitle:
            fig.suptitle(suptitle, fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
        else:
            fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    if args.combined:
        groups = {}
        for run in runs:
            groups.setdefault(window_slug(run[1]) + dataset_slug(run[1]["dataset"]), []).append(run)
        for slug, group in groups.items():
            cfg = group[0][1]
            save_figure(group,
                        os.path.join(out_dir(args.out, cfg),
                                     f"all-models_{window_slug(cfg)}_{args.metric}.png"),
                        suptitle=f"dataset={cfg['dataset']}  L={cfg['L']}  O={cfg['O']}")
    else:
        for model, cfg, logs in runs:
            save_figure([(model, cfg, logs)],
                        os.path.join(out_dir(args.out, cfg),
                                     f"{model.lower()}_{window_slug(cfg)}_{args.metric}.png"),
                        row_h=4.5)


if __name__ == "__main__":
    main()
