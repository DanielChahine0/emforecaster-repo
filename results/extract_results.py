#!/usr/bin/env python3
"""Build point-forecasting result tables from a logs/ tree of log.json runs.

    python results/extract_results.py --logs-root logs/multivariate
    python results/extract_results.py --logs-root "logs/univariate (per channel)" --output out.md
    python results/extract_results.py --logs-root logs/cluster
    python results/extract_results.py   # no --logs-root: processes every logs/* subdirectory

Cluster runs (forecasting/grids.py GRIDS["cluster"]) sweep a k-cluster dimension on
top of the usual (model, dataset, L, O) grid, so k is part of a cell's identity
here: without it the k values of one cell (1, 2, 3, 5 and 9 on EMF; 1, 2, 3 and 6
on VoIP) would be averaged into a single meaningless number. It shows up as an extra column, and only for trees that have
it -- every non-cluster tree's tables are byte-for-byte what they were before.

The last section, '# Ultimate Comparison', puts every pipeline (per-channel,
multivariate, and one column per cluster count) into a single MSE / MAE /
training-time table at ULTIMATE_SETTING, the one window all of them share. It
needs every logs/ subdirectory, so it is only built on a no---logs-root run.
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

from emfres.vocab import (
    MISSING, format_seconds, format_seconds_fine, parse_duration,
)

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
    # The VoIP grids' last-value baseline; no EMF grid carries it.
    "Persistence",
]
DATASET_DISPLAY = {
    "emf_2023_university_5_with_ptv": "EMF2023 (University-5 w/ PTV)",
    "voip_mobile": "VoIP Mobile",
    "voip_fixed": "VoIP Fixed",
}
DATASET_ORDER = ["det", "ptv", "ptv2", "ed2022", "uh2022", "emf_2023_university_5_with_ptv",
                 "voip_mobile", "voip_fixed"]

# MISSING now comes from emfres.vocab, so the tables and the duration
# formatters cannot disagree about what an absent value looks like.

# The single (L, O) window the '# Ultimate Comparison' and '# CCM vs Base'
# sections are built for: the one window where per-channel, multivariate and
# clustered runs can all be put in the same row. It is per corpus -- the EMF
# benchmark is L336/O96 and the VoIP one L48/O12 -- so emfres.tables rebinds it
# around a render rather than every caller agreeing on a constant.
ULTIMATE_SETTING = (336, 96)

from emfres.runlogs import (  # noqa: E402
    Cell, Run, excluded, load_runs, parse_running_time,
)
from emfres.frames import family_of, is_archive  # noqa: E402




def mean_of(runs: list[Run], attr: str) -> Optional[float]:
    vals = [getattr(r, attr) for r in runs if getattr(r, attr) is not None]
    return statistics.mean(vals) if vals else None


def group_runs(runs: list[Run]) -> dict[tuple, list[Run]]:
    # k is last so the existing positional reads of a cell key (key[0] is the
    # model, key[4] the conformal flag) keep working; it is None for every
    # non-cluster run, which leaves those trees grouped exactly as before.
    groups: dict[tuple, list[Run]] = defaultdict(list)
    for r in runs:
        groups[(r.model, r.dataset, r.seq_len, r.pred_len, r.conf, r.k)].append(r)
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
            model, dataset, seq_len, pred_len, conf, k = key
            k_txt = f" k={k}" if k is not None else ""
            print(f"WARNING: wide running_time spread for {model}/{dataset} L{seq_len}_O{pred_len}"
                  f"{k_txt} conf={conf}: {[format_seconds(t) for t in times]} -- possible "
                  f"contended/throttled outlier seed; consider --exclude", file=sys.stderr)


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
    """The (dataset, L, O, k) row identity of every cell in this conformal section."""
    keys = {(ds, sl, pl, k) for (_, ds, sl, pl, c, k) in cells if c == conf}
    return sorted(keys, key=lambda r: (dataset_rank(r[0]), r[1], r[2],
                                       -1 if r[3] is None else r[3]))


def setting_label(sl: int, pl: int) -> str:
    return f"L{sl}/O{pl}"


def fmt_num(v: Optional[float], decimals: int) -> str:
    return f"{v:.{decimals}f}" if v is not None else MISSING


def num_fmt(decimals: int) -> Callable[[Optional[float]], str]:
    return lambda v: fmt_num(v, decimals)


def md_table(headers: list[str], aligns: str, rows: list[list[str]]) -> list[str]:
    rule = {"l": ":---", "r": "---:"}   # aligns is one char per column
    return (
        ["| " + " | ".join(headers) + " |",
         "|" + "|".join(rule[a] for a in aligns) + "|"]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )


class Doc:
    def __init__(self, title: str):
        self.lines = [f"# {title}", ""]
        self.group_no = 0
        self.table_no = 0

    def para(self, text: str) -> None:
        self.lines.extend([text, ""])

    def preamble(self, items: list[tuple[str, str]]) -> None:
        self.lines.extend([f"- {label}: {value}" for label, value in items] + [""])

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


def mark_ranks(strs: list[str], vals: list[Optional[float]]) -> None:
    ranked = sorted((v, i) for i, v in enumerate(vals) if v is not None)
    if ranked:
        best = ranked[0][1]
        strs[best] = f"**{strs[best]}**"
    if len(ranked) > 1:
        second = ranked[1][1]
        strs[second] = f"<u>{strs[second]}</u>"


def shows_k(row_keys: list[tuple]) -> bool:
    """Whether this section needs a k column at all (cluster trees only)."""
    return any(k is not None for _, _, _, k in row_keys)


def metric_rows(cells, models, row_keys, conf, attr, fmt, mark_best: bool) -> list[list[str]]:
    show_k = shows_k(row_keys)
    rows = []
    for ds, sl, pl, k in row_keys:
        vals = [getattr(cells.get((m, ds, sl, pl, conf, k)), attr, None) for m in models]
        strs = [fmt(v) for v in vals]
        if mark_best:
            # Ranked within one k: the models of a cluster row compete at the same
            # cluster count, not against the other k's of the same window.
            mark_ranks(strs, vals)
        head = [dataset_label(ds), str(sl), str(pl)]
        if show_k:
            head.append(MISSING if k is None else str(k))
        rows.append(head + strs)
    return rows


def metric_table(cells, models, row_keys, conf, attr, fmt, mark_best=True) -> list[str]:
    show_k = shows_k(row_keys)
    return md_table(
        ["Dataset", "$L$", "$O$"] + (["$k$"] if show_k else []) + [model_label(m) for m in models],
        "lrr" + ("r" if show_k else "") + "r" * len(models),
        metric_rows(cells, models, row_keys, conf, attr, fmt, mark_best),
    )


def has_metric(cells, models, row_keys, conf, attr) -> bool:
    return any(getattr(cells.get((m, ds, sl, pl, conf, k)), attr, None) is not None
               for m in models for ds, sl, pl, k in row_keys)


def variant_label(m: str, k: Optional[int]) -> str:
    """Model name, disambiguated by cluster count on a cluster tree."""
    return model_label(m) if k is None else f"{model_label(m)} (k={k})"


def model_variants(keys, models) -> list[tuple[str, Optional[int]]]:
    """Every (model, k) pair present, in model order then ascending k."""
    return sorted({(m, k) for (m, _, _, _, _, k) in keys if m in models},
                  key=lambda v: (model_rank(v[0]), -1 if v[1] is None else v[1]))


def epoch_summary_table(runs, models, conf) -> list[str]:
    # Split by k as well as model: CCM builds k parallel per-cluster layers, so
    # epoch cost grows with k and pooling k=2 with k=9 would report neither.
    rows = []
    for m, k in model_variants({(r.model, None, None, None, None, r.k) for r in runs}, models):
        vals = [r.per_epoch for r in runs
                if r.model == m and r.k == k and r.conf == conf and r.per_epoch is not None]
        if vals:
            rows.append((variant_label(m, k), statistics.mean(vals), len(vals)))
    return md_table(
        ["Model", "Mean / epoch", "Estimation for 100 ep", "Runs"],
        "lrrr",
        [[label, format_seconds_fine(mean), f"~{format_seconds(mean * 100)}", str(n)]
         for label, mean, n in sorted(rows, key=lambda r: r[1])],
    )


def model_size_table(cells, models) -> list[str]:
    settings = sorted({(sl, pl) for (_, _, sl, pl, _, _) in cells})
    rows = []
    for m, k in model_variants(cells, models):
        row = []
        for sl, pl in settings:
            found = next((c.num_params for (mm, _, ssl, ppl, _, kk), c in cells.items()
                          if mm == m and kk == k and ssl == sl and ppl == pl and c.num_params), None)
            row.append(f"{found:,}" if found else MISSING)
        if any(v != MISSING for v in row):
            rows.append([variant_label(m, k)] + row)
    if not rows:
        return []
    return md_table(["Model"] + [setting_label(sl, pl) for sl, pl in settings],
                    "l" + "r" * len(settings), rows)


def coverage_summary(runs: list[Run], cells: dict[tuple, Cell]) -> str:
    per_channel = runs and all(r.channel is not None for r in runs)
    ns = sorted({c.n for c in cells.values()})
    span = f"**{ns[0]}**" if len(ns) == 1 else f"**{min(ns)}-{max(ns)}**"
    if per_channel:
        seeds = sorted({c.n_seeds for c in cells.values() if c.n_seeds})
        seed_txt = (f"{seeds[0]} seed" if len(seeds) == 1 and seeds[0] == 1
                    else f"{min(seeds)}-{max(seeds)} seeds" if seeds else "unknown seed count")
        return f"{len(runs)} logs; each cell is a mean over {span} single-channel runs ({seed_txt} each)"
    return f"{len(runs)} logs; each cell averages {span} replicate run(s)"


def render_report(title: str, runs: list[Run], cells: dict[tuple, Cell]) -> str:
    models = sort_models({k[0] for k in cells})
    confs = [c for c in (False, True) if row_keys_for(cells, c)]
    settings = sorted({(r.seq_len, r.pred_len) for r in runs})
    datasets = sorted({r.dataset for r in runs}, key=dataset_rank)

    ks = sorted({r.k for r in runs if r.k is not None})

    doc = Doc(title)
    items = [
        ("Models", ", ".join(model_label(m) for m in models)),
        ("Datasets", ", ".join(dataset_label(d) for d in datasets)),
        ("Grid", ", ".join(setting_label(sl, pl) for sl, pl in settings)),
    ]
    if ks:
        items.append(("Channel clusters",
                      f"k = {', '.join(str(k) for k in ks)} "
                      f"(each k is its own cell, never averaged together)"))
    items.append(("Runs", coverage_summary(runs, cells)))
    items.append(("Conformal prediction",
                  "on for every run" if confs == [True] else
                  "off for every run" if confs == [False] else
                  "reported separately below"))
    doc.preamble(items)

    for conf in confs:
        suffix = ""
        if len(confs) > 1:
            suffix = " (with conformal prediction)" if conf else " (without conformal prediction)"
        row_keys = row_keys_for(cells, conf)

        doc.group("Forecast accuracy" + suffix)
        doc.para(f"**Bold** = best (lowest) in the row, <u>underline</u> = runner-up. "
                 f"`{MISSING}` = not logged.")
        doc.table("MSE (`sl_test/loss`)",
                  metric_table(cells, models, row_keys, conf, "mse", num_fmt(4)))
        if has_metric(cells, models, row_keys, conf, "mae"):
            doc.table("MAE (`sl_test/mae`)",
                      metric_table(cells, models, row_keys, conf, "mae", num_fmt(4)))

        doc.group("Training cost" + suffix)
        doc.para("**Bold** = fastest in the row, <u>underline</u> = runner-up."
                 + (f" `{MISSING}` = not logged (ARIMA is not gradient-trained)."
                    if "ARIMA" in models else ""))
        doc.table("Total training time (`parameters/running_time`)",
                  metric_table(cells, models, row_keys, conf, "running_time", format_seconds))
        doc.table("Per-epoch training time (`running_time / len(sl_train/loss)`)",
                  metric_table(cells, models, row_keys, conf, "per_epoch", format_seconds_fine))
        doc.table("Mean epoch cost by model", epoch_summary_table(runs, models, conf),
                  note="Pooled over every run of that model in this section.")

        if conf:
            doc.group("Conformal prediction")
            doc.para('Not marked -- "best" coverage means closest to the target, not highest, and '
                     "a narrower interval only counts if that target is met.")
            if has_metric(cells, models, row_keys, True, "ic"):
                doc.table("Marginal coverage % (`sl_test/ic`)",
                          metric_table(cells, models, row_keys, True, "ic", num_fmt(2), mark_best=False))
            if has_metric(cells, models, row_keys, True, "jc"):
                doc.table("Joint coverage % (`sl_test/jc`)",
                          metric_table(cells, models, row_keys, True, "jc", num_fmt(2), mark_best=False))
            if has_metric(cells, models, row_keys, True, "interval_width"):
                doc.table("Mean interval width (`sl_test/interval_width_mean`)",
                          metric_table(cells, models, row_keys, True, "interval_width",
                                       num_fmt(3), mark_best=False))

    sizes = model_size_table(cells, models)
    if sizes:
        by = "(model, k, L, O)" if ks else "(model, L, O)"
        doc.group("Model size")
        doc.table("Trainable weights (`parameters/sl/num_params`)", sizes,
                  note=f"For multivariate logs this varies by dataset (different channel counts), "
                       f"not just by {by}; shown here is the first value found per {by}.")

    return doc.render()


def canonical_cells(cells: dict[tuple, Cell]) -> dict[tuple, Cell]:
    # re-key by display name so RF_EMF_Transformer and Transformer line up
    out: dict[tuple, Cell] = {}
    for (m, ds, sl, pl, conf, k), cell in sorted(cells.items(), key=lambda kv: model_rank(kv[0][0])):
        out.setdefault((model_label(m), ds, sl, pl, conf, k), cell)
    return out


def shared_keys(a: dict[tuple, Cell], b: dict[tuple, Cell]) -> list[tuple]:
    return sorted(set(a) & set(b),
                  key=lambda k: (dataset_rank(k[1]), k[2], k[3], display_rank(k[0])))


def fmt_delta(value: float, decimals: int, unit: str) -> str:
    if round(value, decimals) == 0:
        return f"{0:.{decimals}f}{unit}"
    return f"{value:+.{decimals}f}{unit}"


def pair_rows(keys, left, right, attr, fmt, lower_is_better: bool, delta: str):
    rows, wins = [], [0, 0, 0]
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
        model, ds, sl, pl = key[0], key[1], key[2], key[3]
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
    doc.preamble([
        ("Compares", " · ".join(f"{left} vs {short_label(label)}" for label, _, _ in matched)),
        ("Matched on", "(dataset, L, O, model, conformal setting); only cells present in both "
                       "trees appear below"),
        ("Delta", f"Δ is the univariate value relative to {left}, so negative Δ means the "
                  "univariate value is lower; **bold** = better of the two where lower is better"),
    ])

    all_wins: list[tuple[str, str, list[int]]] = []
    for label, uv, keys in matched:
        right = short_label(label)
        pair = f"{left} vs {right}"

        doc.group(f"Forecast accuracy ({pair})")
        doc.para(f"{len(keys)} matched cells, from `{mv_title}` and `{label}`.")
        for name, log_key, attr, decimals in [("MSE", "sl_test/loss", "mse", 4),
                                              ("MAE", "sl_test/mae", "mae", 4)]:
            table, wins = pair_table(keys, mv, uv, left, right, attr, num_fmt(decimals))
            doc.table(f"{name} (`{log_key}`)", table)
            all_wins.append((right, name, wins))

        doc.group(f"Training cost ({pair})")
        doc.para(f"`{right}` is the mean cost of one channel's model; `all channels` multiplies it "
                 f"by the channel count. Δ compares that total against {left}.")
        rows = []
        for key in keys:
            a = getattr(mv.get(key), "running_time", None)
            b = getattr(uv.get(key), "running_time", None)
            chans = getattr(uv.get(key), "n_channels", None)
            total = b * chans if b is not None and chans else None
            sd = fmt_delta((total - a) / a * 100, 1, "%") if a and total is not None else MISSING
            model, ds, sl, pl = key[0], key[1], key[2], key[3]
            rows.append([dataset_label(ds), str(sl), str(pl), model, format_seconds(a),
                         format_seconds(b), str(chans) if chans else MISSING,
                         format_seconds(total), sd])
        doc.table("Total training time (`parameters/running_time`)",
                  md_table(["Dataset", "$L$", "$O$", "Model", left, right,
                            "Channels", "All channels", "Δ"], "lrrlrrrrr", rows))

        conformal = [key for key in keys if key[4]]  # key[4] is the conf flag, not k
        if conformal:
            doc.group(f"Conformal prediction ({pair})")
            doc.para("Δ is in percentage points for coverage, relative for interval width. "
                     'Coverage is not marked -- "best" means closest to the target.')
            for name, log_key, attr, decimals, lower, kind in [
                ("Marginal coverage %", "sl_test/ic", "ic", 2, False, "pp"),
                ("Joint coverage %", "sl_test/jc", "jc", 2, False, "pp"),
                ("Mean interval width", "sl_test/interval_width_mean", "interval_width", 3, True, "pct"),
            ]:
                if any(getattr(mv.get(key), attr, None) is not None for key in conformal):
                    table, wins = pair_table(conformal, mv, uv, left, right, attr,
                                             num_fmt(decimals), lower_is_better=lower, delta=kind)
                    doc.table(f"{name} (`{log_key}`)", table)
                    if lower:
                        all_wins.append((right, "Interval width", wins))

    doc.group("Scoreboard")
    doc.table("Matched-cell wins by metric",
              md_table(["Comparison", "Metric", left, "Univariate", "Tie"], "llrrr",
                       [[f"{left} vs {label}", metric, str(w[0]), str(w[1]), str(w[2])]
                        for label, metric, w in all_wins]),
              note="How many matched cells each pipeline wins, counted only where lower is better.")
    return doc.render()


@dataclass
class Pipeline:
    """One row-group of the ultimate table: a whole training strategy.

    `cells` is always display-name-keyed (canonical_cells), so RF_EMF_Transformer
    and Transformer line up across trees. `k` selects one cluster count out of a
    cluster tree and is None for the trees that have no k dimension.
    """
    label: str
    cells: dict[tuple, Cell]
    k: Optional[int] = None
    per_channel: bool = False


def pipeline_value(p: Pipeline, model: str, ds: str, sl: int, pl: int,
                   conf: bool, attr: str) -> Optional[float]:
    cell = p.cells.get((model, ds, sl, pl, conf, p.k))
    v = getattr(cell, attr, None)
    if v is not None and p.per_channel and attr == "running_time":
        # One per-channel log is one channel's model; the strategy's real cost is
        # all C of them, which is what the other pipelines' single run competes with.
        return v * cell.n_channels if cell.n_channels else None
    return v


def ultimate_axes(pipelines: list[Pipeline], sl: int, pl: int,
                  conf: bool) -> tuple[list[str], list[str]]:
    """The (models, datasets) any pipeline covers at this window, in report order."""
    keys = [key for p in pipelines for key in p.cells
            if key[2] == sl and key[3] == pl and key[4] == conf and key[5] == p.k]
    return (sorted({key[0] for key in keys}, key=display_rank),
            sorted({key[1] for key in keys}, key=dataset_rank))


def ultimate_rows(pipelines, models, datasets, sl, pl, conf, attr, fmt) -> list[list[str]]:
    rows = []
    for ds in datasets:
        for p in pipelines:
            vals = [pipeline_value(p, m, ds, sl, pl, conf, attr) for m in models]
            if all(v is None for v in vals):
                continue
            strs = [fmt(v) for v in vals]
            # Ranked across the row, i.e. model vs model under one strategy; the
            # strategies are compared by reading down a dataset block instead.
            mark_ranks(strs, vals)
            rows.append([dataset_label(ds), p.label] + strs)
    return rows


def ultimate_table(pipelines, models, datasets, sl, pl, conf, attr, fmt) -> list[str]:
    rows = ultimate_rows(pipelines, models, datasets, sl, pl, conf, attr, fmt)
    if not rows:
        return []
    return md_table(["Dataset", "Pipeline"] + models, "ll" + "r" * len(models), rows)


def channel_counts(pipelines, datasets, sl, pl, conf) -> str:
    """'ED2022 C=27, ...' for the per-channel pipeline, for the cost-table note."""
    counts = {}
    for p in pipelines:
        if not p.per_channel:
            continue
        for (_, ds, ssl, ppl, c, k), cell in p.cells.items():
            if (ssl, ppl, c, k) == (sl, pl, conf, p.k) and cell.n_channels:
                counts[ds] = max(counts.get(ds, 0), cell.n_channels)
    return ", ".join(f"{dataset_label(ds)} C={counts[ds]}"
                     for ds in datasets if ds in counts)


def render_ultimate(title: str, pipelines: list[Pipeline]) -> str:
    sl, pl = ULTIMATE_SETTING
    axes = [(conf,) + ultimate_axes(pipelines, sl, pl, conf) for conf in (False, True)]
    axes = [a for a in axes if a[1]]
    if not axes:
        return ""

    doc = Doc(title)
    items = [
        ("Configuration", f"{setting_label(sl, pl)} only -- the one window every pipeline shares"),
        ("Pipelines", " · ".join(p.label for p in pipelines)),
        ("Datasets", ", ".join(dataset_label(d) for d in axes[0][2])),
        ("Models", ", ".join(axes[0][1])),
    ]
    if any(p.k == 1 for p in pipelines):
        items.append(
            ("Caveat", f"`Cluster (k=1)` is the multivariate network with CCM's "
                       f"clustering switched off (an all-ones sinkhorn assignment, one "
                       f"shared cluster-wise linear, no similarity gradient), so it is "
                       f"the controlled no-clustering baseline. It still carries the "
                       f"cluster assigner's parameters, which is why its weight count "
                       f"exceeds `{short_label(MULTIVARIATE_TITLE)}`'s. The two agree "
                       f"within seed noise, not bit for bit (the per-seed gaps are in "
                       f"`Channel Clustering (CCM) vs. plain`). Cluster rows are the "
                       f"logged batch protocol; the fixed-membership numbers, the "
                       f"primary protocol, are in `Paradigm Comparison`."
                       + (" Runs made before 2026-08-25 by the upstream wrappers are in "
                          "logs/_archive and are not in these tables."
                          if any(family_of(d) == "emf" for d in axes[0][2]) else "")))
    items.append(
        ("Conformal prediction",
         "on for every run" if [a[0] for a in axes] == [True] else
         "off for every run" if [a[0] for a in axes] == [False] else
         "reported separately below"))
    doc.preamble(items)

    for conf, models, datasets in axes:
        suffix = ""
        if len(axes) > 1:
            suffix = " (with conformal prediction)" if conf else " (without conformal prediction)"

        doc.group("Every pipeline side by side" + suffix)
        doc.para(f"Read **across** a row to rank the models under one pipeline "
                 f"(**bold** = best, <u>underline</u> = runner-up); read **down** a dataset "
                 f"block to see what the same model does per channel, jointly, or with a "
                 f"k-cluster head. `{MISSING}` = that combination was not run "
                 f"(ARIMA and Persistence never are: they fit one model per band, so "
                 f"there is nothing to cluster).")

        for name, log_key, attr, fmt in [("MSE", "sl_test/loss", "mse", num_fmt(4)),
                                         ("MAE", "sl_test/mae", "mae", num_fmt(4))]:
            table = ultimate_table(pipelines, models, datasets, sl, pl, conf, attr, fmt)
            if table:
                doc.table(f"{name} (`{log_key}`)", table)

        table = ultimate_table(pipelines, models, datasets, sl, pl, conf,
                               "running_time", format_seconds)
        if table:
            chans = channel_counts(pipelines, datasets, sl, pl, conf)
            doc.table("Total training time (`parameters/running_time`)", table,
                      note=f"Cost of the whole strategy, so the per-channel row is the mean "
                           f"single-channel run times the channel count"
                           + (f" ({chans})" if chans else "") + "; every other row is one "
                           f"model's training run, averaged over seeds. ARIMA is not gradient-trained and "
                           f"logs no time.")

    return doc.render()


# The cluster cell whose logged parameters stand for a family's sweep in the
# CCM paper's configuration table (Table 10): the first dataset at a mid-range
# k. The EMF sweep runs k 1, 2, 3, 5, 9 over 27-29 bands; VoIP k 1, 2, 3, 6
# over 6 signals, where k = 5 was never run.
CCM_REPRESENTATIVE = {"emf": ("ed2022", 5), "voip": ("voip_mobile", 3)}


def render_ccm_vs_base(title: str, cluster_cells: dict[tuple, Cell],
                       mv_cells: dict[tuple, Cell], cluster_runs: list[Run]) -> str:
    """The CCM paper's Table 4 shape: base | + CCM per backbone, IMP(%) last.

    Both inputs are canonical_cells-keyed. "Base" is the plain multivariate
    cell from the '# Multivariate' tree - the same job config, byte-identical
    training hyperparameters, only the network swapped for upstream's CCM
    wrapper. The + CCM column takes, per (dataset, model), the k with the best
    test MSE and reuses that k for the MAE row; the note flags this as a
    post-hoc (oracle) choice, unlike the paper's validation-tuned k. A second
    table mirrors the paper's Table 10 configuration summary, read from the
    cluster runs' own logs.
    """
    sl, pl = ULTIMATE_SETTING
    conf = True
    at = lambda key: key[2] == sl and key[3] == pl and key[4] == conf
    models = sorted({k[0] for k in cluster_cells if at(k) and k[5] is not None},
                    key=display_rank)
    datasets = sorted({k[1] for k in cluster_cells if at(k) and k[5] is not None},
                      key=dataset_rank)
    if not models or not datasets:
        return ""

    family = family_of(datasets[0])
    rep_ds, rep_k = CCM_REPRESENTATIVE.get(family, (datasets[0], None))
    if rep_ds not in datasets:
        rep_ds = datasets[0]
    windows = sorted({(r.seq_len, r.pred_len) for r in cluster_runs
                      if family_of(r.dataset) == family}, reverse=True)
    others = [setting_label(*w) for w in windows if w != (sl, pl)]

    doc = Doc(title)
    doc.preamble([
        ("Format", "the CCM paper's Table 4 - per backbone a base | + CCM column "
                   "pair, **bold** = better of the pair, <u>underline</u> = best in "
                   "the row, last column the mean % improvement of CCM over the "
                   "base models"),
        ("Base", f"the plain multivariate cell from '# {MULTIVARIATE_TITLE}' - same "
                 f"job config and training hyperparameters, no CCM"),
        ("Configuration", f"{setting_label(sl, pl)}"
                          + (f"; the k sweep also ran at {', '.join(others)}, in "
                             f"`Channel Clustering (CCM) vs. plain`" if others
                             else " -- the window the k sweep covers")),
        ("Protocol", "the logged batch protocol (cluster membership averaged over each "
                     "evaluation batch); the primary fixed-membership numbers, with k "
                     "chosen on validation, are in `Paradigm Comparison`"),
        ("Caveat", "the + CCM cell's k is chosen post hoc by test MSE per (dataset, "
                   "model) and reused for MAE - an oracle bound, where the paper "
                   "tunes k on validation data. The base column is the plain "
                   "multivariate cell; `Cluster (k=1)` in the Ultimate Comparison is "
                   "the same network with clustering off and is the stricter baseline."),
        ("Conformal prediction", "on for every run"),
    ])

    doc.group("Base vs + CCM")

    headers = ["Dataset", "Metric"]
    for m in models:
        headers += [m, "+ CCM"]
    headers.append("IMP (%)")
    rows = []
    for ds in datasets:
        ks_of = lambda m: sorted({k[5] for k in cluster_cells
                                  if k[0] == m and k[1] == ds and at(k)
                                  and k[5] is not None})
        best_k = {}
        for m in models:
            scored = [(cluster_cells[(m, ds, sl, pl, conf, k)].mse, k)
                      for k in ks_of(m)
                      if cluster_cells[(m, ds, sl, pl, conf, k)].mse is not None]
            if scored:
                best_k[m] = min(scored)[1]
        for attr, name in (("mse", "MSE"), ("mae", "MAE")):
            strs: list[str] = []
            vals: list[Optional[float]] = []
            imps = []
            for m in models:
                base_cell = mv_cells.get((m, ds, sl, pl, conf, None))
                base = getattr(base_cell, attr, None)
                k = best_k.get(m)
                ccm_cell = cluster_cells.get((m, ds, sl, pl, conf, k)) if k else None
                ccm = getattr(ccm_cell, attr, None)
                bs, cs = fmt_num(base, 4), fmt_num(ccm, 4)
                if ccm is not None:
                    cs = f"{cs} (k={k})"
                if base is not None and ccm is not None:
                    if ccm < base:
                        cs = f"**{cs}**"
                    elif base < ccm:
                        bs = f"**{bs}**"
                    imps.append((base - ccm) / base * 100.0)
                strs += [bs, cs]
                vals += [base, ccm]
            ranked = sorted((v, i) for i, v in enumerate(vals) if v is not None)
            if ranked:
                i = ranked[0][1]
                strs[i] = f"<u>{strs[i]}</u>"
            imp = f"{statistics.mean(imps):+.2f}" if imps else MISSING
            rows.append([dataset_label(ds), name] + strs + [imp])
    doc.table("Long-horizon forecasting, base vs + CCM",
              md_table(headers, "ll" + "r" * (2 * len(models) + 1), rows),
              note="IMP (%) = mean over the backbones of (base − CCM) / base × 100 "
                   "for that row's metric; positive = CCM ahead.")

    # --- the paper's Table 10 analogue, from the cluster runs' own logs ------- #
    def ccm_log(m: str) -> dict:
        for r in cluster_runs:
            if (model_label(r.model) == m and r.seq_len == sl and r.pred_len == pl
                    and r.k == rep_k and r.dataset == rep_ds):
                with open(r.path) as f:
                    return json.load(f)
        return {}

    logs = {m: ccm_log(m) for m in models}
    fmt_int = lambda v: f"{v:,}" if v is not None else MISSING
    param = lambda m, key: logs[m].get(key)

    def best_k_line(m: str) -> str:
        parts = []
        for ds in datasets:
            scored = [(cluster_cells[(m, ds, sl, pl, conf, k)].mse, k)
                      for k in sorted({key[5] for key in cluster_cells
                                       if key[0] == m and key[1] == ds and at(key)
                                       and key[5] is not None})
                      if cluster_cells[(m, ds, sl, pl, conf, k)].mse is not None]
            parts.append(str(min(scored)[1]) if scored else MISSING)
        return " / ".join(parts)

    cfg_rows = [
        ["Channel clusters k swept"] + [
            ", ".join(str(v) for v in vs) if (vs := sorted(
                {key[5] for key in cluster_cells
                 if key[0] == m and at(key) and key[5] is not None})) else MISSING
            for m in models],
        [f"Best k by test MSE ({' / '.join(dataset_label(d) for d in datasets)})"]
        + [best_k_line(m) for m in models],
        ["β (similarity-loss weight)"] + [
            fmt_num(param(m, "parameters/cluster/beta"), 1) for m in models],
        ["Sinkhorn temperature ε"] + ["0.05" for _ in models],
        ["Cluster-assigner hidden dim (`sl/d_model`)"] + [
            str(param(m, "parameters/sl/d_model") or MISSING) for m in models],
        ["Learning rate"] + [
            f"{param(m, 'parameters/sl/lr'):g}" if param(m, "parameters/sl/lr")
            else MISSING for m in models],
        ["Weight decay"] + [
            f"{param(m, 'parameters/sl/weight_decay'):g}"
            if param(m, "parameters/sl/weight_decay") else MISSING for m in models],
        ["Batch size"] + [
            str(param(m, "parameters/sl/batch_size") or MISSING) for m in models],
        [f"Trainable weights, base → + CCM ({dataset_label(rep_ds)}, k={rep_k})"] + [
            f"{fmt_int(getattr(mv_cells.get((m, rep_ds, sl, pl, conf, None)), 'num_params', None))}"
            f" → {fmt_int(getattr(cluster_cells.get((m, rep_ds, sl, pl, conf, rep_k)), 'num_params', None))}"
            for m in models],
    ]
    doc.table("Experiment configuration",
              md_table(["Configuration"] + models, "l" + "r" * len(models), cfg_rows),
              note=f"Read from the cluster runs' logged parameters ({dataset_label(rep_ds)}, "
                   f"k={rep_k} representative, {setting_label(sl, pl)}; β and the "
                   f"training set-up are identical across the family's cells). `-` = "
                   f"no run of that model at that cell yet. The sinkhorn temperature "
                   f"is upstream's hard-coded default "
                   f"(`TimeSeriesCCM-main/models/layers.py`), not a logged parameter.")

    return doc.render()


DEFAULT_TABLES_FILE = Path(__file__).resolve().parent / "tables.md"
DEFAULT_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

UNIVARIATE_SERIES_TITLE = "Univariate Series"
PER_CHANNEL_TITLE = "Univariate (Per Channel)"
MULTIVARIATE_TITLE = "Multivariate"
CLUSTER_TITLE = "Cluster (k-Grouped Channels)"
COMPARISON_TITLE = "Multivariate vs Univariate"
CCM_TITLE = "CCM vs Base (No Clustering)"
ULTIMATE_TITLE = "Ultimate Comparison"
ROOT_TITLES = {
    "univariate": UNIVARIATE_SERIES_TITLE,
    "univariate (per channel)": PER_CHANNEL_TITLE,
    "multivariate": MULTIVARIATE_TITLE,
    "cluster": CLUSTER_TITLE,
}
SECTION_ORDER = [
    UNIVARIATE_SERIES_TITLE,
    PER_CHANNEL_TITLE,
    MULTIVARIATE_TITLE,
    CLUSTER_TITLE,
    CCM_TITLE,
    COMPARISON_TITLE,
    ULTIMATE_TITLE,
]
# compact forms used as column headers in the side-by-side comparison
SHORT_TITLES = {
    "Univariate Series": "Univariate series",
    "Univariate (Per Channel)": "Per-channel",
    MULTIVARIATE_TITLE: "Multivariate",
}
# headings written by older versions of this script
HEADING_ALIASES = {
    "Univariate": "Univariate Series",
    "Univariate Per Channel": "Univariate (Per Channel)",
}

# only top-level '# ' headings; everything a report emits is '##' or deeper
_SECTION_RE = re.compile(r"^# (.+)$", re.MULTILINE)


def short_label(title: str) -> str:
    return SHORT_TITLES.get(title, title)


def build_pipelines(by_title: dict[str, dict[tuple, Cell]]) -> list[Pipeline]:
    """The ultimate table's rows, ordered least to most channel mixing.

    Empty unless at least two of the three trees were loaded -- one cluster tree
    alone yields several pipelines (one per k), and a k-only table is what the
    '# Cluster (k-Grouped Channels)' section already is.

    The univariate-series tree is deliberately absent: its cells are whole-dataset
    RSS series over a different set of datasets, so it shares no (dataset, model)
    identity with the others and would only ever add empty rows.
    """
    if len([t for t in (PER_CHANNEL_TITLE, CLUSTER_TITLE, MULTIVARIATE_TITLE)
            if t in by_title]) < 2:
        return []
    pipelines = []
    if PER_CHANNEL_TITLE in by_title:
        pipelines.append(Pipeline("Per-channel", canonical_cells(by_title[PER_CHANNEL_TITLE]),
                                  per_channel=True))
    if CLUSTER_TITLE in by_title:
        cluster = canonical_cells(by_title[CLUSTER_TITLE])
        for k in sorted({key[5] for key in cluster if key[5] is not None}):
            pipelines.append(Pipeline(f"Cluster (k={k})", cluster, k=k))
    if MULTIVARIATE_TITLE in by_title:
        pipelines.append(Pipeline("Multivariate", canonical_cells(by_title[MULTIVARIATE_TITLE])))
    return pipelines


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
    parser.add_argument("--exclude", action="append", default=[], metavar="NAME",
                         help="skip log.json paths containing this substring, repeatable")
    args = parser.parse_args()

    if args.logs_root is not None:
        logs_roots = [args.logs_root]
    else:
        if args.output is not None or args.title is not None:
            parser.error("--output/--title require an explicit --logs-root")
        if not DEFAULT_LOGS_DIR.is_dir():
            parser.error(f"no --logs-root given and {DEFAULT_LOGS_DIR} is not a directory")
        # Same archive rule the frames extractor applies, so a bare run here
        # and `make.py extract` cover the same tree.
        logs_roots = sorted(p for p in DEFAULT_LOGS_DIR.iterdir()
                            if p.is_dir() and not is_archive(p))
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
    # The cluster tree is deliberately not a right-hand side here: its cells carry a
    # k the univariate trees have no counterpart for, and this section's framing
    # ("Δ is the univariate value") does not describe a cluster-vs-joint delta. Its
    # own '# Cluster (k-Grouped Channels)' section above carries the numbers.
    uv_roots = [(t, c) for t, c in by_title.items()
                if t not in (MULTIVARIATE_TITLE, CLUSTER_TITLE)]
    if mv_cells and uv_roots:
        comparison = render_comparison(COMPARISON_TITLE, MULTIVARIATE_TITLE, mv_cells, uv_roots)
        if comparison:
            updates.append((COMPARISON_TITLE, comparison))
        else:
            print("WARNING: no (dataset, L, O, model) cells shared between the multivariate and "
                  "univariate trees; skipping the comparison section", file=sys.stderr)

    cl_cells = by_title.get(CLUSTER_TITLE)
    if mv_cells and cl_cells:
        cluster_runs = next((r for t, r, _ in loaded if t == CLUSTER_TITLE), [])
        ccm_section = render_ccm_vs_base(CCM_TITLE, canonical_cells(cl_cells),
                                         canonical_cells(mv_cells), cluster_runs)
        if ccm_section:
            updates.append((CCM_TITLE, ccm_section))

    pipelines = build_pipelines(by_title)
    ultimate = render_ultimate(ULTIMATE_TITLE, pipelines) if pipelines else ""
    if ultimate:
        updates.append((ULTIMATE_TITLE, ultimate))
    elif pipelines:
        print(f"WARNING: no pipeline has {setting_label(*ULTIMATE_SETTING)} cells; "
              f"skipping the '{ULTIMATE_TITLE}' section", file=sys.stderr)

    if len(loaded) < 2:
        print(f"NOTE: '{COMPARISON_TITLE}' and '{ULTIMATE_TITLE}' need every logs/ "
              f"subdirectory; run without --logs-root to rebuild them.", file=sys.stderr)

    for title, replaced in upsert_sections(DEFAULT_TABLES_FILE, updates):
        print(f"{'Updated' if replaced else 'Added'} '# {title}' section in {DEFAULT_TABLES_FILE}.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
