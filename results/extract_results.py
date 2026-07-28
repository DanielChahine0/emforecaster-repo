#!/usr/bin/env python3
"""
Extract point-forecasting results from a `logs/` tree of `log.json` run files
and render them as Markdown tables, in the same style as the hand-written
results/{univariate,multivariate}/point_forecasting_results.md files.

Each run directory under --logs-root is expected to contain a log.json with
flat `parameters/...` config keys plus metric arrays (sl_test/loss,
sl_test/mae, and, for conformal runs, sl_test/ic, sl_test/jc,
sl_test/interval_width_mean). Runs are grouped into cells by
(model, dataset, seq_len, pred_len, conf) and averaged over every replicate
found in that group. "Replicate" is deliberately generic: for a normal grid
that's seeds (logs/<model>/<dataset>/L*_O*[_cp]/<seed>/log.json); for a
per-channel tree (logs/univariate (per channel)/...) each channel directory
is one more replicate, since it also ends in a log.json two levels down.
Point this script at whichever logs directory holds the runs you want
averaged together -- it does not merge across multiple --logs-root trees.

This script does NOT reproduce two judgment calls that exist in the current
hand-written point_forecasting_results.md files: it does not impute
individual anomalous running_time outliers (contended/throttled machines),
and it does not exclude smoke-test runs. Use --exclude to drop specific
runs by path substring, and watch the "outlier" warnings printed to stderr.

Usage:
    python results/extract_results.py \\
        --logs-root logs/univariate \\
        --title "Univariate Point-Forecasting Performance" \\
        --output results/univariate/point_forecasting_results_generated.md

    python results/extract_results.py \\
        --logs-root logs/multivariate \\
        --title "Multivariate Point-Forecasting Performance" \\
        --output results/multivariate/point_forecasting_results_generated.md
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

MODEL_DISPLAY = {
    "EMForecaster": "EMForecaster (2025)",
    "DLinear": "DLinear (2023)",
    "TSMixer": "TSMixer (2023)",
    "PatchTST": "PatchTST (2023)",
    "Transformer": "Transformer (2017)",
    "RF_EMF_Transformer": "Transformer (2017)",
    "LSTM": "LSTM (1997)",
    "RF_EMF_LSTM": "LSTM (1997)",
    "ARIMA": "ARIMA (1970)",
}
MODEL_ORDER = [
    "EMForecaster", "DLinear", "TSMixer", "PatchTST",
    "Transformer", "RF_EMF_Transformer", "LSTM", "RF_EMF_LSTM", "ARIMA",
]
DATASET_ORDER = ["det", "ptv", "ptv2", "ed2022", "uh2022", "emf2023"]

# Every running_time string observed in the current logs is either "<N>s" or
# "<N>min <N>s" -- no hour-scale values, unlike the imputed times narrated in
# the hand-written notes (those were fixed in the log.json itself already).
_TIME_RE = re.compile(r"^(?:(?P<min>\d+)min\s*)?(?:(?P<sec>\d+)s)?$")


def parse_running_time(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    m = _TIME_RE.match(raw.strip())
    if not m or (m.group("min") is None and m.group("sec") is None):
        raise ValueError(f"unrecognized running_time format: {raw!r}")
    return int(m.group("min") or 0) * 60 + int(m.group("sec") or 0)


def format_seconds(sec: Optional[float]) -> str:
    if sec is None:
        return "X"
    sec = round(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if h or m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def format_seconds_fine(sec: Optional[float]) -> str:
    """One-decimal seconds for sub-minute per-epoch times; falls back to format_seconds above a minute."""
    if sec is None:
        return "X"
    if sec < 60:
        return f"{sec:.1f}s"
    return format_seconds(sec)


@dataclass
class Run:
    model: str
    dataset: str
    seq_len: int
    pred_len: int
    conf: bool
    mse: Optional[float]
    mae: Optional[float]
    running_time: Optional[float]
    per_epoch: Optional[float]
    num_params: Optional[int]
    ic: Optional[float]
    jc: Optional[float]
    interval_width: Optional[float]
    path: Path


@dataclass
class Cell:
    mse: Optional[float]
    mae: Optional[float]
    running_time: Optional[float]
    per_epoch: Optional[float]
    ic: Optional[float]
    jc: Optional[float]
    interval_width: Optional[float]
    num_params: Optional[int]
    n: int


def load_runs(logs_root: Path, exclude: list[str]) -> list[Run]:
    runs = []
    for log_path in sorted(logs_root.rglob("log.json")):
        if any(pat in str(log_path) for pat in exclude):
            continue
        with open(log_path) as f:
            d = json.load(f)

        model = d.get("parameters/exp/model_id")
        dataset_raw = d.get("parameters/data/dataset")
        seq_len = d.get("parameters/data/seq_len")
        pred_len = d.get("parameters/data/pred_len")
        if model is None or dataset_raw is None or seq_len is None or pred_len is None:
            print(f"WARNING: skipping {log_path} (missing model/dataset/seq_len/pred_len)", file=sys.stderr)
            continue

        criterion = d.get("parameters/sl/criterion")
        if criterion not in (None, "MSE"):
            print(f"WARNING: {log_path} uses criterion={criterion!r}, not MSE -- "
                  f"its 'MSE' column value is actually sl_test/loss under that criterion", file=sys.stderr)

        conf = bool(d.get("parameters/conf/conf"))
        dataset = dataset_raw.lower()
        if dataset.startswith("rf_emf_"):
            dataset = dataset[len("rf_emf_"):]

        test_loss = d.get("sl_test/loss")
        mse = test_loss[0] if test_loss else None
        test_mae = d.get("sl_test/mae")
        mae = test_mae[0] if test_mae else None

        train_loss = d.get("sl_train/loss")
        epochs = len(train_loss) if train_loss else None
        try:
            running_time = parse_running_time(d.get("parameters/running_time"))
        except ValueError as exc:
            print(f"WARNING: {log_path}: {exc}", file=sys.stderr)
            running_time = None
        per_epoch = running_time / epochs if running_time is not None and epochs else None

        num_params = d.get("parameters/sl/num_params") or None

        ic = jc = interval_width = None
        if conf:
            v = d.get("sl_test/ic")
            ic = v[0] if v else None
            v = d.get("sl_test/jc")
            jc = v[0] if v else None
            v = d.get("sl_test/interval_width_mean")
            interval_width = v[0] if v else None

        runs.append(Run(model, dataset, seq_len, pred_len, conf, mse, mae,
                         running_time, per_epoch, num_params, ic, jc, interval_width, log_path))
    return runs


def mean_of(runs: list[Run], attr: str) -> Optional[float]:
    vals = [getattr(r, attr) for r in runs if getattr(r, attr) is not None]
    return statistics.mean(vals) if vals else None


def aggregate(runs: list[Run]) -> dict[tuple, Cell]:
    groups: dict[tuple, list[Run]] = defaultdict(list)
    for r in runs:
        groups[(r.model, r.dataset, r.seq_len, r.pred_len, r.conf)].append(r)

    cells = {}
    for key, rs in groups.items():
        cells[key] = Cell(
            mse=mean_of(rs, "mse"),
            mae=mean_of(rs, "mae"),
            running_time=mean_of(rs, "running_time"),
            per_epoch=mean_of(rs, "per_epoch"),
            ic=mean_of(rs, "ic"),
            jc=mean_of(rs, "jc"),
            interval_width=mean_of(rs, "interval_width"),
            num_params=next((r.num_params for r in rs if r.num_params), None),
            n=len(rs),
        )
    return cells


def diagnose(runs: list[Run], cells: dict[tuple, Cell]) -> None:
    """Print (not fix) anomalies to stderr: e.g. a single seed 3x+ slower than its siblings."""
    groups: dict[tuple, list[Run]] = defaultdict(list)
    for r in runs:
        groups[(r.model, r.dataset, r.seq_len, r.pred_len, r.conf)].append(r)
    for key, rs in groups.items():
        times = [r.running_time for r in rs if r.running_time is not None]
        if len(times) < 2:
            continue
        lo, hi = min(times), max(times)
        if lo > 0 and hi / lo >= 3:
            model, dataset, seq_len, pred_len, conf = key
            print(f"WARNING: wide running_time spread for {model}/{dataset} L{seq_len}_O{pred_len} "
                  f"conf={conf}: {[format_seconds(t) for t in times]} -- possible contended/throttled "
                  f"outlier seed; consider --exclude", file=sys.stderr)


def model_rank(m: str) -> tuple:
    return (0, MODEL_ORDER.index(m)) if m in MODEL_ORDER else (1, m)


def dataset_rank(ds: str) -> tuple:
    return (0, DATASET_ORDER.index(ds)) if ds in DATASET_ORDER else (1, ds)


def sort_models(models: set[str]) -> list[str]:
    return sorted(models, key=model_rank)


def row_keys_for(cells: dict[tuple, Cell], conf: bool) -> list[tuple]:
    keys = {(ds, sl, pl) for (_, ds, sl, pl, c) in cells if c == conf}
    return sorted(keys, key=lambda k: (dataset_rank(k[0]), k[1], k[2]))


def fmt_num(v: Optional[float], decimals: int) -> str:
    return f"{v:.{decimals}f}" if v is not None else ""


def render_metric_table(cells, models, row_keys, conf, attr, decimals, bold_lowest) -> str:
    lines = [
        "| Dataset | $L$ | $O$ | " + " | ".join(MODEL_DISPLAY.get(m, m) for m in models) + " |",
        "|---|---|---|" + "---|" * len(models),
    ]
    for ds, sl, pl in row_keys:
        row_vals = [getattr(cells.get((m, ds, sl, pl, conf)), attr, None) for m in models]
        strs = [fmt_num(v, decimals) for v in row_vals]
        if bold_lowest:
            populated = [(i, v) for i, v in enumerate(row_vals) if v is not None]
            if populated:
                min_i = min(populated, key=lambda p: p[1])[0]
                strs[min_i] = f"**{strs[min_i]}**"
        lines.append(f"| {ds.upper()} | {sl} | {pl} | " + " | ".join(strs) + " |")
    return "\n".join(lines)


def render_time_table(cells, models, row_keys, conf, attr, formatter) -> str:
    lines = [
        "| Dataset | $L$ | $O$ | " + " | ".join(MODEL_DISPLAY.get(m, m) for m in models) + " |",
        "|---|---|---|" + "---|" * len(models),
    ]
    for ds, sl, pl in row_keys:
        row_vals = [getattr(cells.get((m, ds, sl, pl, conf)), attr, None) for m in models]
        strs = [formatter(v) for v in row_vals]
        populated = [(i, v) for i, v in enumerate(row_vals) if v is not None]
        if populated:
            min_i = min(populated, key=lambda p: p[1])[0]
            strs[min_i] = f"**{strs[min_i]}**"
        lines.append(f"| {ds.upper()} | {sl} | {pl} | " + " | ".join(strs) + " |")
    return "\n".join(lines)


def render_summary_table(runs, models, conf) -> str:
    lines = [
        "| Model | Mean / epoch | Estimation for 100 ep |",
        "|---|---:|---:|",
    ]
    rows = []
    for m in models:
        vals = [r.per_epoch for r in runs if r.model == m and r.conf == conf and r.per_epoch is not None]
        if not vals:
            continue
        mean_epoch = statistics.mean(vals)
        rows.append((m, mean_epoch))
    for m, mean_epoch in sorted(rows, key=lambda r: r[1]):
        lines.append(f"| {MODEL_DISPLAY.get(m, m)} | {format_seconds_fine(mean_epoch)} "
                      f"| ~{format_seconds(mean_epoch * 100)} |")
    return "\n".join(lines)


def render_model_sizes(cells, models) -> str:
    lo_cols = sorted({(sl, pl) for (_, _, sl, pl, _) in cells}, key=lambda p: (p[0], p[1]))
    lines = [
        "| Model | " + " | ".join(f"L{sl} / O{pl}" for sl, pl in lo_cols) + " |",
        "|---|" + "---:|" * len(lo_cols),
    ]
    for m in models:
        row = []
        for sl, pl in lo_cols:
            found = None
            for (mm, _, ssl, ppl, _), cell in cells.items():
                if mm == m and ssl == sl and ppl == pl and cell.num_params:
                    found = cell.num_params
                    break
            row.append(f"{found:,}" if found else "")
        if any(row):
            lines.append(f"| {MODEL_DISPLAY.get(m, m)} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def replicate_note(cells: dict[tuple, Cell], conf: bool) -> str:
    ns = sorted({cell.n for key, cell in cells.items() if key[4] == conf})
    if len(ns) <= 1:
        n = ns[0] if ns else 0
        return f"Every cell in this section averages **{n}** replicate run(s)."
    return (f"Replicate counts per cell in this section range from **{min(ns)}** to **{max(ns)}**, "
            f"so rows mix cells with different amounts of averaging.")


def render_section(title: str, runs: list[Run], cells: dict[tuple, Cell], models: list[str], conf: bool) -> str:
    row_keys = row_keys_for(cells, conf)
    if not row_keys:
        return ""

    out = [f"# {title}", ""]
    out.append("## Table A | MSE (`sl_test/loss`)")
    out.append("")
    out.append("**Bold** = best (lowest) value in the row, among populated cells.")
    out.append("")
    out.append(render_metric_table(cells, models, row_keys, conf, "mse", 4, bold_lowest=True))
    out.append("")

    if any(cells.get((m, ds, sl, pl, conf)) and cells[(m, ds, sl, pl, conf)].mae is not None
           for m in models for ds, sl, pl in row_keys):
        out.append("## Table A2 | MAE (`sl_test/mae`)")
        out.append("")
        out.append(render_metric_table(cells, models, row_keys, conf, "mae", 4, bold_lowest=True))
        out.append("")

    out.append("## Table B | Training Time (`parameters/running_time`)")
    out.append("")
    out.append("**Bold** = fastest in the row. `X` = not logged (e.g. ARIMA is not gradient-trained).")
    out.append("")
    out.append(render_time_table(cells, models, row_keys, conf, "running_time", format_seconds))
    out.append("")

    out.append("## Table C | Per-Epoch Training Time (`running_time / len(sl_train/loss)`, per run)")
    out.append("")
    out.append(render_time_table(cells, models, row_keys, conf, "per_epoch", format_seconds_fine))
    out.append("")
    out.append("### Per-model summary")
    out.append("")
    out.append(render_summary_table(runs, models, conf))
    out.append("")

    if conf:
        if any(cells.get((m, ds, sl, pl, True)) and cells[(m, ds, sl, pl, True)].ic is not None
               for m in models for ds, sl, pl in row_keys):
            out.append("## Table D | Marginal Coverage % (`sl_test/ic`)")
            out.append("")
            out.append("Not bolded -- \"best\" means closest to the target coverage, not highest.")
            out.append("")
            out.append(render_metric_table(cells, models, row_keys, True, "ic", 2, bold_lowest=False))
            out.append("")
        if any(cells.get((m, ds, sl, pl, True)) and cells[(m, ds, sl, pl, True)].jc is not None
               for m in models for ds, sl, pl in row_keys):
            out.append("## Table E | Joint Coverage % (`sl_test/jc`)")
            out.append("")
            out.append(render_metric_table(cells, models, row_keys, True, "jc", 2, bold_lowest=False))
            out.append("")
        if any(cells.get((m, ds, sl, pl, True)) and cells[(m, ds, sl, pl, True)].interval_width is not None
               for m in models for ds, sl, pl in row_keys):
            out.append("## Table F | Mean Interval Width (`sl_test/interval_width_mean`)")
            out.append("")
            out.append("Not bolded -- narrower is only better when the target coverage (Table D) is met.")
            out.append("")
            out.append(render_metric_table(cells, models, row_keys, True, "interval_width", 3, bold_lowest=False))
            out.append("")

    out.append(replicate_note(cells, conf))
    out.append("")
    return "\n".join(out)


def render_report(title: str, logs_root: Path, runs: list[Run], cells: dict[tuple, Cell]) -> str:
    models = sort_models({k[0] for k in cells})
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    out = [
        f"# {title}",
        "",
        f"*Auto-generated by `results/extract_results.py` from `{logs_root}` on {generated}. "
        f"{len(runs)} runs found, {len(cells)} (model, dataset, L, O, conf) cells.*",
        "",
        "Test-set metrics, averaged over every replicate run found in each cell "
        "(seeds, or channels for a per-channel logs tree). "
        "$L$ = input/lookback length (`seq_len`), $O$ = forecast horizon (`pred_len`).",
        "",
        "---",
        "",
    ]

    without_cp = render_section("1 · Point-Forecasting Performance (without conformal prediction)",
                                 runs, cells, models, conf=False)
    if without_cp:
        out.append(without_cp)
        out.append("---")
        out.append("")

    with_cp = render_section("2 · Point-Forecasting Performance (with conformal prediction)",
                              runs, cells, models, conf=True)
    if with_cp:
        out.append(with_cp)
        out.append("---")
        out.append("")

    sizes = render_model_sizes(cells, models)
    if sizes.count("\n") > 1:
        out.append("# Model sizes")
        out.append("")
        out.append("Weights come from `parameters/sl/num_params` in each log. For multivariate logs this can "
                    "vary by dataset (different channel counts), not just by (model, L, O); this table shows "
                    "the first value found per (model, L, O) -- cross-check if that matters for your use case.")
        out.append("")
        out.append(sizes)
        out.append("")

    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-root", required=True, type=Path,
                         help="Directory to walk for log.json files, e.g. 'logs/univariate' or 'logs/multivariate'.")
    parser.add_argument("--output", required=True, type=Path,
                         help="Markdown file to write.")
    parser.add_argument("--title", default=None,
                         help="Report title. Defaults to the --logs-root directory name.")
    parser.add_argument("--exclude", action="append", default=[], metavar="SUBSTRING",
                         help="Substring to match against a log.json path to exclude that run. Repeatable.")
    args = parser.parse_args()

    if not args.logs_root.is_dir():
        parser.error(f"--logs-root {args.logs_root} is not a directory")

    runs = load_runs(args.logs_root, args.exclude)
    if not runs:
        parser.error(f"no usable log.json files found under {args.logs_root}")

    cells = aggregate(runs)
    diagnose(runs, cells)

    title = args.title or args.logs_root.name.replace("_", " ").title()
    md = render_report(title, args.logs_root, runs, cells)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    print(f"Wrote {args.output} ({len(runs)} runs, {len(cells)} aggregated cells).")


if __name__ == "__main__":
    main()
