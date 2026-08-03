#!/usr/bin/env python3
"""Build point-forecasting result tables from a logs/ tree of log.json runs.

    python results/extract_results.py --logs-root logs/multivariate
    python results/extract_results.py --logs-root "logs/univariate (per channel)" --output out.md
    python results/extract_results.py   # no --logs-root: rebuild every section of results/tables.md
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

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
DATASET_DISPLAY = {
    "emf_2023_university_5_with_ptv": "EMF2023 (University-5 w/ PTV)",
}
DATASET_ORDER = ["det", "ptv", "ptv2", "ed2022", "uh2022", "emf_2023_university_5_with_ptv"]

MISSING = "—"

# format is "<N>s", "<N>min <N>s" or "<N>hr <N>min <N>s"
_TIME_RE = re.compile(
    r"^(?:(?P<hr>\d+)h(?:r|rs)?\s*)?(?:(?P<min>\d+)min\s*)?(?:(?P<sec>\d+)s)?$")
_CHANNEL_RE = re.compile(r"^ch(\d+)$")


def parse_running_time(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    m = _TIME_RE.match(raw.strip())
    if not m or not any(m.group(g) for g in ("hr", "min", "sec")):
        raise ValueError(f"unrecognized running_time format: {raw!r}")
    return (int(m.group("hr") or 0) * 3600
            + int(m.group("min") or 0) * 60
            + int(m.group("sec") or 0))


def format_seconds(sec: Optional[float]) -> str:
    if sec is None:
        return MISSING
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
    if sec is None:
        return MISSING
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
    channel: Optional[int]
    seed: Optional[str]
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
    n_channels: Optional[int]
    n_seeds: Optional[int]


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

        channel = None
        for part in log_path.parts:
            m = _CHANNEL_RE.fullmatch(part)
            if m:
                channel = int(m.group(1))
        seed = log_path.parent.name if log_path.parent.name.isdigit() else None

        runs.append(Run(model, dataset, seq_len, pred_len, conf, mse, mae,
                         running_time, per_epoch, num_params, ic, jc, interval_width,
                         channel, seed, log_path))
    return runs


def mean_of(runs: list[Run], attr: str) -> Optional[float]:
    vals = [getattr(r, attr) for r in runs if getattr(r, attr) is not None]
    return statistics.mean(vals) if vals else None


def group_runs(runs: list[Run]) -> dict[tuple, list[Run]]:
    groups: dict[tuple, list[Run]] = defaultdict(list)
    for r in runs:
        groups[(r.model, r.dataset, r.seq_len, r.pred_len, r.conf)].append(r)
    return groups


def aggregate(runs: list[Run]) -> dict[tuple, Cell]:
    cells = {}
    for key, rs in group_runs(runs).items():
        channels = {r.channel for r in rs if r.channel is not None}
        seeds = {r.seed for r in rs if r.seed is not None}
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
            n_channels=len(channels) or None,
            n_seeds=len(seeds) or None,
        )
    return cells


def diagnose(runs: list[Run], cells: dict[tuple, Cell]) -> None:
    """Flag (don't fix) a seed running 3x+ slower than its siblings in the same cell."""
    for key, rs in group_runs(runs).items():
        times = [r.running_time for r in rs if r.running_time is not None]
        if len(times) < 2:
            continue
        lo, hi = min(times), max(times)
        if lo > 0 and hi / lo >= 3:
            model, dataset, seq_len, pred_len, conf = key
            print(f"WARNING: wide running_time spread for {model}/{dataset} L{seq_len}_O{pred_len} "
                  f"conf={conf}: {[format_seconds(t) for t in times]} -- possible contended/throttled "
                  f"outlier seed; consider --exclude", file=sys.stderr)


# ---------------------------------------------------------------- ordering

def model_rank(m: str) -> tuple:
    return (0, MODEL_ORDER.index(m)) if m in MODEL_ORDER else (1, m)


def dataset_rank(ds: str) -> tuple:
    return (0, DATASET_ORDER.index(ds)) if ds in DATASET_ORDER else (1, ds)


def sort_models(models: set[str]) -> list[str]:
    return sorted(models, key=model_rank)


def model_label(m: str) -> str:
    return MODEL_DISPLAY.get(m, m)


def dataset_label(ds: str) -> str:
    return DATASET_DISPLAY.get(ds, ds.upper())


DISPLAY_ORDER = list(dict.fromkeys(model_label(m) for m in MODEL_ORDER))


def display_rank(label: str) -> tuple:
    return (0, DISPLAY_ORDER.index(label)) if label in DISPLAY_ORDER else (1, label)


def row_keys_for(cells: dict[tuple, Cell], conf: bool) -> list[tuple]:
    keys = {(ds, sl, pl) for (_, ds, sl, pl, c) in cells if c == conf}
    return sorted(keys, key=lambda k: (dataset_rank(k[0]), k[1], k[2]))


# ---------------------------------------------------------------- markdown

def fmt_num(v: Optional[float], decimals: int) -> str:
    return f"{v:.{decimals}f}" if v is not None else MISSING


def num_fmt(decimals: int) -> Callable[[Optional[float]], str]:
    return lambda v: fmt_num(v, decimals)


def md_table(headers: list[str], aligns: str, rows: list[list[str]]) -> list[str]:
    """`aligns` is one char per column: 'l' or 'r'."""
    rule = {"l": ":---", "r": "---:"}
    return (
        ["| " + " | ".join(headers) + " |",
         "|" + "|".join(rule[a] for a in aligns) + "|"]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )


class Doc:
    """Markdown builder that keeps heading levels and numbering consistent."""

    def __init__(self, title: str):
        self.lines = [f"# {title}", ""]
        self.group_no = 0
        self.table_no = 0

    def para(self, text: str) -> None:
        self.lines.extend([text, ""])

    def quote(self, items: list[tuple[str, str]]) -> None:
        for i, (label, value) in enumerate(items):
            if i:
                self.lines.append(">")
            self.lines.append(f"> **{label}** · {value}")
        self.lines.append("")

    def group(self, title: str) -> None:
        self.group_no += 1
        self.table_no = 0
        self.lines.extend([f"## {self.group_no} · {title}", ""])

    def table(self, title: str, lines: list[str], note: Optional[str] = None) -> None:
        self.table_no += 1
        self.lines.extend([f"### {self.group_no}.{self.table_no} · {title}", ""])
        if note:
            self.lines.extend([note, ""])
        self.lines.extend(lines + [""])

    def render(self) -> str:
        return "\n".join(self.lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------- per-root tables

def mark_ranks(strs: list[str], vals: list[Optional[float]]) -> None:
    """Bold the lowest value in the row and underline the runner-up, in place."""
    ranked = sorted((v, i) for i, v in enumerate(vals) if v is not None)
    if ranked:
        best = ranked[0][1]
        strs[best] = f"**{strs[best]}**"
    if len(ranked) > 1:
        second = ranked[1][1]
        strs[second] = f"<u>{strs[second]}</u>"


def metric_rows(cells, models, row_keys, conf, attr, fmt, mark_best: bool) -> list[list[str]]:
    rows = []
    for ds, sl, pl in row_keys:
        vals = [getattr(cells.get((m, ds, sl, pl, conf)), attr, None) for m in models]
        strs = [fmt(v) for v in vals]
        if mark_best:
            mark_ranks(strs, vals)
        rows.append([dataset_label(ds), str(sl), str(pl)] + strs)
    return rows


def metric_table(cells, models, row_keys, conf, attr, fmt, mark_best=True) -> list[str]:
    return md_table(
        ["Dataset", "$L$", "$O$"] + [model_label(m) for m in models],
        "lrr" + "r" * len(models),
        metric_rows(cells, models, row_keys, conf, attr, fmt, mark_best),
    )


def has_metric(cells, models, row_keys, conf, attr) -> bool:
    return any(getattr(cells.get((m, ds, sl, pl, conf)), attr, None) is not None
               for m in models for ds, sl, pl in row_keys)


def epoch_summary_table(runs, models, conf) -> list[str]:
    rows = []
    for m in models:
        vals = [r.per_epoch for r in runs if r.model == m and r.conf == conf and r.per_epoch is not None]
        if vals:
            rows.append((m, statistics.mean(vals), len(vals)))
    return md_table(
        ["Model", "Mean / epoch", "Projected 100 epochs", "Runs"],
        "lrrr",
        [[model_label(m), format_seconds_fine(mean), f"~{format_seconds(mean * 100)}", str(n)]
         for m, mean, n in sorted(rows, key=lambda r: r[1])],
    )


def model_size_table(cells, models) -> list[str]:
    settings = sorted({(sl, pl) for (_, _, sl, pl, _) in cells})
    rows = []
    for m in models:
        row = []
        for sl, pl in settings:
            found = next((c.num_params for (mm, _, ssl, ppl, _), c in cells.items()
                          if mm == m and ssl == sl and ppl == pl and c.num_params), None)
            row.append(f"{found:,}" if found else MISSING)
        if any(v != MISSING for v in row):
            rows.append([model_label(m)] + row)
    if not rows:
        return []
    return md_table(["Model"] + [f"L{sl} → O{pl}" for sl, pl in settings], "l" + "r" * len(settings), rows)


def coverage_summary(runs: list[Run], cells: dict[tuple, Cell]) -> str:
    per_channel = runs and all(r.channel is not None for r in runs)
    ns = sorted({c.n for c in cells.values()})
    span = f"**{ns[0]}**" if len(ns) == 1 else f"**{min(ns)}–{max(ns)}**"
    if per_channel:
        seeds = sorted({c.n_seeds for c in cells.values() if c.n_seeds})
        seed_txt = (f"{seeds[0]} seed" if len(seeds) == 1 and seeds[0] == 1
                    else f"{min(seeds)}–{max(seeds)} seeds" if seeds else "unknown seed count")
        return (f"{len(runs)} logs · each table cell is a mean over {span} single-channel runs "
                f"({seed_txt} per channel)")
    return f"{len(runs)} logs · each table cell averages {span} replicate run(s)"


def render_report(title: str, runs: list[Run], cells: dict[tuple, Cell]) -> str:
    models = sort_models({k[0] for k in cells})
    confs = [c for c in (False, True) if row_keys_for(cells, c)]
    settings = sorted({(r.seq_len, r.pred_len) for r in runs})
    datasets = sorted({r.dataset for r in runs}, key=dataset_rank)

    doc = Doc(title)
    doc.quote([
        ("Models", ", ".join(model_label(m) for m in models)),
        ("Datasets", ", ".join(dataset_label(d) for d in datasets)),
        ("Grid", ", ".join(f"L{sl} → O{pl}" for sl, pl in settings)),
        ("Runs", coverage_summary(runs, cells)),
        ("Conformal prediction",
         "on for every run" if confs == [True] else
         "off for every run" if confs == [False] else
         "reported separately below for runs with and without it"),
    ])

    for conf in confs:
        suffix = ""
        if len(confs) > 1:
            suffix = " — with conformal prediction" if conf else " — without conformal prediction"
        row_keys = row_keys_for(cells, conf)

        doc.group("Forecast accuracy" + suffix)
        doc.para("**Bold** marks the best (lowest) value in each row and <u>underline</u> the "
                 f"runner-up, among populated cells. `{MISSING}` means the run was not logged.")
        doc.table("MSE — `sl_test/loss`",
                  metric_table(cells, models, row_keys, conf, "mse", num_fmt(4)))
        if has_metric(cells, models, row_keys, conf, "mae"):
            doc.table("MAE — `sl_test/mae`",
                      metric_table(cells, models, row_keys, conf, "mae", num_fmt(4)))

        doc.group("Training cost" + suffix)
        doc.para("**Bold** marks the fastest model in each row, <u>underline</u> the runner-up."
                 + (" ARIMA is not gradient-trained, so it logs no training time or epochs."
                    if "ARIMA" in models else ""))
        doc.table("Total training time — `parameters/running_time`",
                  metric_table(cells, models, row_keys, conf, "running_time", format_seconds))
        doc.table("Per-epoch training time — `running_time / len(sl_train/loss)`",
                  metric_table(cells, models, row_keys, conf, "per_epoch", format_seconds_fine))
        doc.table("Mean epoch cost by model", epoch_summary_table(runs, models, conf),
                  note="Averaged over every run of that model in this section, all datasets and "
                       "settings pooled.")

        if conf:
            doc.group("Conformal prediction")
            doc.para("Nothing is bolded here: the best coverage is the one closest to the target, "
                     "not the highest, and a narrower interval only helps when the target coverage "
                     "is actually met.")
            if has_metric(cells, models, row_keys, True, "ic"):
                doc.table("Marginal coverage % — `sl_test/ic`",
                          metric_table(cells, models, row_keys, True, "ic", num_fmt(2), mark_best=False))
            if has_metric(cells, models, row_keys, True, "jc"):
                doc.table("Joint coverage % — `sl_test/jc`",
                          metric_table(cells, models, row_keys, True, "jc", num_fmt(2), mark_best=False))
            if has_metric(cells, models, row_keys, True, "interval_width"):
                doc.table("Mean interval width — `sl_test/interval_width_mean`",
                          metric_table(cells, models, row_keys, True, "interval_width",
                                       num_fmt(3), mark_best=False))

    sizes = model_size_table(cells, models)
    if sizes:
        doc.group("Model size")
        doc.table("Trainable weights — `parameters/sl/num_params`", sizes,
                  note="Multivariate models size with the channel count, so a value can vary by "
                       "dataset as well as by (model, $L$, $O$); this table shows the first value "
                       "found per (model, $L$, $O$).")

    return doc.render()


# ---------------------------------------------------------------- comparison

def canonical_cells(cells: dict[tuple, Cell]) -> dict[tuple, Cell]:
    """Re-key cells by display name so `RF_EMF_Transformer` and `Transformer` line up."""
    out: dict[tuple, Cell] = {}
    for (m, ds, sl, pl, conf), cell in sorted(cells.items(), key=lambda kv: model_rank(kv[0][0])):
        out.setdefault((model_label(m), ds, sl, pl, conf), cell)
    return out


def shared_keys(a: dict[tuple, Cell], b: dict[tuple, Cell]) -> list[tuple]:
    return sorted(set(a) & set(b),
                  key=lambda k: (dataset_rank(k[1]), k[2], k[3], display_rank(k[0])))


def fmt_delta(value: float, decimals: int, unit: str) -> str:
    if round(value, decimals) == 0:
        return f"{0:.{decimals}f}{unit}"
    return f"{value:+.{decimals}f}{unit}"


def pair_rows(keys, left, right, attr, fmt, lower_is_better: bool, delta: str):
    rows, wins = [], [0, 0, 0]  # left, right, tie
    for key in keys:
        a = getattr(left.get(key), attr, None)
        b = getattr(right.get(key), attr, None)
        sa, sb, sd = fmt(a), fmt(b), MISSING
        if a is not None and b is not None:
            if delta == "pct" and a:
                sd = fmt_delta((b - a) / a * 100, 1, "%")
            elif delta == "pp":
                sd = fmt_delta(b - a, 2, " pp")
            if lower_is_better:
                if b < a:
                    sb, wins[1] = f"**{sb}**", wins[1] + 1
                elif a < b:
                    sa, wins[0] = f"**{sa}**", wins[0] + 1
                else:
                    wins[2] += 1
        model, ds, sl, pl, _ = key
        rows.append([dataset_label(ds), str(sl), str(pl), model, sa, sb, sd])
    return rows, wins


def pair_table(keys, left, right, left_label, right_label, attr, fmt,
               lower_is_better=True, delta="pct"):
    rows, wins = pair_rows(keys, left, right, attr, fmt, lower_is_better, delta)
    return md_table(["Dataset", "$L$", "$O$", "Model", left_label, right_label, "Δ"],
                    "lrrlrrr", rows), wins


def render_comparison(title: str, mv_title: str, mv_cells: dict[tuple, Cell],
                      uv_roots: list[tuple[str, dict[tuple, Cell]]]) -> str:
    mv = canonical_cells(mv_cells)
    matched = [(label, canonical_cells(c)) for label, c in uv_roots]
    matched = [(label, c, shared_keys(mv, c)) for label, c in matched]
    matched = [(label, c, keys) for label, c, keys in matched if keys]
    if not matched:
        return ""

    left = short_label(mv_title)
    doc = Doc(title)
    doc.quote([
        ("Compares", " · ".join(f"{left} vs {short_label(label)}" for label, _, _ in matched)),
        ("Matched on", "(dataset, $L$, $O$, model, conformal setting) — only cells present in "
                       "both trees appear below"),
        ("Reading Δ", f"Δ is the univariate value relative to {left}, so a negative Δ favours the "
                      "univariate pipeline; **bold** marks the better of the two wherever lower "
                      "is unambiguously better"),
    ])

    all_wins: list[tuple[str, str, list[int]]] = []
    for label, uv, keys in matched:
        right = short_label(label)
        pair = f"{left} vs {right}"

        doc.group(f"Accuracy — {pair}")
        doc.para(f"{len(keys)} matched cells, from `{mv_title}` and `{label}`.")
        for metric, attr, decimals in [("MSE — `sl_test/loss`", "mse", 4),
                                       ("MAE — `sl_test/mae`", "mae", 4)]:
            table, wins = pair_table(keys, mv, uv, left, right, attr, num_fmt(decimals))
            doc.table(metric, table)
            all_wins.append((right, metric.split(" —")[0], wins))

        doc.group(f"Training cost — {pair}")
        doc.para(f"`{right}` is the mean cost of one channel's model; `all channels` multiplies it "
                 "by the channel count, which is what the whole forecast costs when every channel "
                 f"needs its own model. Δ compares that total against {left}.")
        rows = []
        for key in keys:
            a = getattr(mv.get(key), "running_time", None)
            b = getattr(uv.get(key), "running_time", None)
            chans = getattr(uv.get(key), "n_channels", None)
            total = b * chans if b is not None and chans else None
            sd = fmt_delta((total - a) / a * 100, 1, "%") if a and total is not None else MISSING
            model, ds, sl, pl, _ = key
            rows.append([dataset_label(ds), str(sl), str(pl), model, format_seconds(a),
                         format_seconds(b), str(chans) if chans else MISSING,
                         format_seconds(total), sd])
        doc.table("Total training time — `parameters/running_time`",
                  md_table(["Dataset", "$L$", "$O$", "Model", left, right,
                            "Channels", "All channels", "Δ"], "lrrlrrrrr", rows))

        conformal = [k for k in keys if k[4]]
        if conformal:
            doc.group(f"Conformal prediction — {pair}")
            doc.para("Δ is in percentage points for coverage and relative for interval width. "
                     "Coverage is not bolded: closest to target wins, not highest.")
            for metric, attr, decimals, lower, kind in [
                ("Marginal coverage % — `sl_test/ic`", "ic", 2, False, "pp"),
                ("Joint coverage % — `sl_test/jc`", "jc", 2, False, "pp"),
                ("Mean interval width — `sl_test/interval_width_mean`", "interval_width", 3, True, "pct"),
            ]:
                if any(getattr(mv.get(k), attr, None) is not None for k in conformal):
                    table, wins = pair_table(conformal, mv, uv, left, right, attr,
                                             num_fmt(decimals), lower_is_better=lower, delta=kind)
                    doc.table(metric, table)
                    if lower:
                        all_wins.append((right, "Interval width", wins))

    doc.group("Scoreboard")
    doc.table("Matched-cell wins by metric",
              md_table(["Comparison", "Metric", left, "Univariate", "Tie"], "llrrr",
                       [[f"{left} vs {label}", metric, str(w[0]), str(w[1]), str(w[2])]
                        for label, metric, w in all_wins]),
              note="How many matched cells each pipeline wins, counted only for the metrics where "
                   "lower is unambiguously better.")
    return doc.render()


# ---------------------------------------------------------------- file I/O

DEFAULT_TABLES_FILE = Path(__file__).resolve().parent / "tables.md"
DEFAULT_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

MULTIVARIATE_TITLE = "Multivariate"
COMPARISON_TITLE = "Multivariate vs Univariate"
ROOT_TITLES = {
    "univariate": "Univariate Series",
    "univariate (per channel)": "Univariate (Per Channel)",
    "multivariate": MULTIVARIATE_TITLE,
}
# compact forms used as column headers in the side-by-side comparison
SHORT_TITLES = {
    "Univariate Series": "Univariate series",
    "Univariate (Per Channel)": "Per-channel",
    MULTIVARIATE_TITLE: "Multivariate",
}


def short_label(title: str) -> str:
    return SHORT_TITLES.get(title, title)
SECTION_ORDER = [
    "Univariate Series",
    "Univariate (Per Channel)",
    MULTIVARIATE_TITLE,
    COMPARISON_TITLE,
]
# headings written by older versions of this script
HEADING_ALIASES = {
    "Univariate": "Univariate Series",
    "Univariate Per Channel": "Univariate (Per Channel)",
}

# only top-level '# ' headings; everything a report emits is '##' or deeper
_SECTION_RE = re.compile(r"^# (.+)$", re.MULTILINE)


def derive_heading(logs_root: Path) -> str:
    key = logs_root.name.lower()
    if key in ROOT_TITLES:
        return ROOT_TITLES[key]
    name = re.sub(r"[_\s]+", " ", logs_root.name.replace("(", " ").replace(")", " ")).strip()
    return name.title()


def split_sections(text: str) -> list[tuple[str, str]]:
    matches = list(_SECTION_RE.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = m.group(1).strip()
        body = text[start:end].rstrip("\n")
        renamed = HEADING_ALIASES.get(heading)
        if renamed:
            heading = renamed
            body = _SECTION_RE.sub(f"# {renamed}", body, count=1)
        sections.append((heading, body))
    return sections


def upsert_sections(tables_path: Path, updates: list[tuple[str, str]]) -> list[tuple[str, bool]]:
    """Replace or append each section, then reorder the file to SECTION_ORDER."""
    existing = tables_path.read_text() if tables_path.exists() else ""
    sections: list[tuple[str, str]] = []
    for heading, body in split_sections(existing):
        if not any(heading == h for h, _ in sections):
            sections.append((heading, body))

    outcome = []
    for heading, body in updates:
        block = body.rstrip("\n")
        for i, (h, _) in enumerate(sections):
            if h == heading:
                sections[i] = (heading, block)
                outcome.append((heading, True))
                break
        else:
            sections.append((heading, block))
            outcome.append((heading, False))

    rank = {h: i for i, h in enumerate(SECTION_ORDER)}
    ordered = sorted(enumerate(sections), key=lambda p: (rank.get(p[1][0], len(rank)), p[0]))
    tables_path.write_text("\n\n".join(b for _, (_, b) in ordered) + "\n")
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-root", type=Path, default=None,
                         help=f"directory to scan for log.json files; omit to process every "
                              f"subdirectory of {DEFAULT_LOGS_DIR} in turn")
    parser.add_argument("--output", type=Path, default=None,
                         help=f"write the full report here; omit to upsert a section into "
                              f"{DEFAULT_TABLES_FILE.name} (requires an explicit --logs-root)")
    parser.add_argument("--title", default=None,
                         help="report/section heading, default derived from --logs-root "
                              "(requires an explicit --logs-root)")
    parser.add_argument("--exclude", action="append", default=[], metavar="SUBSTRING",
                         help="skip log.json paths containing this substring, repeatable")
    args = parser.parse_args()

    if args.logs_root is not None:
        logs_roots = [args.logs_root]
    else:
        if args.output is not None or args.title is not None:
            parser.error("--output/--title require an explicit --logs-root")
        if not DEFAULT_LOGS_DIR.is_dir():
            parser.error(f"no --logs-root given and {DEFAULT_LOGS_DIR} is not a directory")
        logs_roots = sorted(p for p in DEFAULT_LOGS_DIR.iterdir() if p.is_dir())
        if not logs_roots:
            parser.error(f"no subdirectories found under {DEFAULT_LOGS_DIR}")

    loaded: list[tuple[str, list[Run], dict[tuple, Cell]]] = []
    for logs_root in logs_roots:
        if not logs_root.is_dir():
            parser.error(f"--logs-root {logs_root} is not a directory")

        runs = load_runs(logs_root, args.exclude)
        if not runs:
            if len(logs_roots) == 1:
                parser.error(f"no usable log.json files found under {logs_root}")
            print(f"WARNING: no usable log.json files found under {logs_root}, skipping", file=sys.stderr)
            continue

        cells = aggregate(runs)
        diagnose(runs, cells)
        loaded.append((args.title or derive_heading(logs_root), runs, cells))

    if args.output is not None:
        title, runs, cells = loaded[0]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_report(title, runs, cells))
        print(f"Wrote {args.output} ({len(runs)} runs, {len(cells)} aggregated cells).", file=sys.stderr)
        return

    updates = [(title, render_report(title, runs, cells)) for title, runs, cells in loaded]

    by_title = {title: cells for title, _, cells in loaded}
    mv_cells = by_title.get(MULTIVARIATE_TITLE)
    uv_roots = [(t, c) for t, c in by_title.items() if t != MULTIVARIATE_TITLE]
    if mv_cells and uv_roots:
        comparison = render_comparison(COMPARISON_TITLE, MULTIVARIATE_TITLE, mv_cells, uv_roots)
        if comparison:
            updates.append((COMPARISON_TITLE, comparison))
        else:
            print("WARNING: no (dataset, L, O, model) cells shared between the multivariate and "
                  "univariate trees; skipping the comparison section", file=sys.stderr)
    elif len(loaded) < 2:
        print(f"NOTE: '{COMPARISON_TITLE}' needs every logs/ subdirectory; run without "
              f"--logs-root to rebuild it.", file=sys.stderr)

    for title, replaced in upsert_sections(DEFAULT_TABLES_FILE, updates):
        print(f"{'Updated' if replaced else 'Added'} '# {title}' section in {DEFAULT_TABLES_FILE}.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
