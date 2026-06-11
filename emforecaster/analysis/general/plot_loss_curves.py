"""Plot train/val loss-vs-epoch curves with the final test loss overlaid,
reading the ``log.json`` files under ``logs/{ablation}_{Model}_{id}/{seed}/``.

Curves are averaged across seeds (mean with a +/-1 std band, NaN-aware so
early-stopped seeds can end at different epochs). Test loss is a single scalar
evaluated on the best-validation checkpoint, drawn as a horizontal line.
Per-run figures go to ``{out}/{dataset}/{model}/``; ``--combined`` writes one
figure per (dataset, L, O) with a subplot per model to ``{out}/{dataset}/``.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# metric -> (per-epoch train key, per-epoch val key, test scalar key, axis label)
METRICS = {
    "loss": ("sl_train/loss", "sl_val/loss", "sl_test/loss", "MSE loss"),
    "mae": (None, "sl_val/mae", "sl_test/mae", "MAE"),
}


def find_run_dirs(logs_dir, model, run_dir=None, latest=False):
    if run_dir is not None:
        return [run_dir] if os.path.isdir(run_dir) else []
    pattern = re.compile(rf"^\d+_{re.escape(model)}_\w+$")
    dirs = [d for d in glob.glob(os.path.join(logs_dir, "*"))
            if os.path.isdir(d) and pattern.match(os.path.basename(d))]
    dirs.sort(key=os.path.getmtime)
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


def run_slug(cfg):
    return f"{cfg['dataset']}_L{cfg['L']}_O{cfg['O']}"


def out_dir(base, cfg, model=None):
    path = os.path.join(base, str(cfg["dataset"] or "unknown_dataset"), *([model] if model else []))
    os.makedirs(path, exist_ok=True)
    return path


def aggregate(logs, key):
    """Per-epoch mean/std across seeds; shorter seeds are NaN-padded."""
    curves = [np.asarray(d[key], dtype=float) for d in logs.values() if key in d]
    max_len = max((len(c) for c in curves), default=0)
    if max_len == 0:
        return None
    stacked = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        stacked[i, :len(c)] = c
    with np.errstate(invalid="ignore"):
        mean, std = np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)
    return np.arange(1, max_len + 1), mean, std


def plot_curve(ax, logs, key, label, color):
    agg = aggregate(logs, key)
    if agg is None:
        return None
    epochs, mean, std = agg
    ax.plot(epochs, mean, label=label, color=color, lw=2)
    if len(logs) > 1:
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
    return mean


def plot_metric(ax, logs, title, metric):
    train_key, val_key, test_key, ylabel = METRICS[metric]
    if train_key:
        plot_curve(ax, logs, train_key, "train", "#1f77b4")
    val_mean = plot_curve(ax, logs, val_key, "val", "#ff7f0e")
    if val_mean is not None:
        best_epoch = int(np.nanargmin(val_mean)) + 1
        ax.axvline(best_epoch, color="#ff7f0e", ls=":", lw=1, alpha=0.7)
        title += f"  (best val @ epoch {best_epoch})"

    test_vals = [float(d[test_key][0]) for d in logs.values() if d.get(test_key)]
    if test_vals:
        label = f"test (final) = {np.mean(test_vals):.4f}"
        if len(test_vals) > 1:
            label += f" ± {np.std(test_vals):.4f}"
        ax.axhline(np.mean(test_vals), color="#2ca02c", ls="--", lw=1.8, label=label)

    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--models", nargs="+", default=["EMForecaster", "DLinear", "PatchTST", "TSMixer"])
    p.add_argument("--run-dir", default=None, help="explicit run directory (use with a single model)")
    p.add_argument("--seed", type=int, default=None, help="plot one seed instead of aggregating")
    p.add_argument("--datasets", nargs="+", default=None, help="only runs trained on these datasets")
    p.add_argument("--latest", action="store_true", help="only the most recent run per model")
    p.add_argument("--metric", choices=["loss", "mae", "both"], default="loss")
    p.add_argument("--out", default="comparison")
    p.add_argument("--combined", action="store_true",
                   help="one figure per (dataset, L, O) with a subplot per model")
    p.add_argument("--logy", action="store_true")
    args = p.parse_args()

    metrics = ["loss", "mae"] if args.metric == "both" else [args.metric]

    runs = []
    for model in args.models:
        run_dirs = find_run_dirs(args.logs_dir, model,
                                 args.run_dir if len(args.models) == 1 else None, args.latest)
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
            groups.setdefault(run_slug(run[1]), []).append(run)
        for slug, group in groups.items():
            cfg = group[0][1]
            save_figure(group,
                        os.path.join(out_dir(args.out, cfg),
                                     f"loss_curves_combined_{slug}_{args.metric}.png"),
                        suptitle=f"dataset={cfg['dataset']}  L={cfg['L']}  O={cfg['O']}")
    else:
        for model, cfg, logs in runs:
            save_figure([(model, cfg, logs)],
                        os.path.join(out_dir(args.out, cfg, model),
                                     f"loss_curve_{model}_{run_slug(cfg)}_{args.metric}.png"),
                        row_h=4.5)


if __name__ == "__main__":
    main()
