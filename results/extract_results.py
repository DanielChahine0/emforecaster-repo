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
from dataclasses import dataclass, replace
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
from emfres.frames import complete_seeds, family_of, is_archive  # noqa: E402




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
        by_seed: dict = defaultdict(list)
        for r in rs:
            if r.seed is not None:
                by_seed[r.seed].append({"channel": r.channel})
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
            seeds=tuple(sorted(seeds)),
            n_complete_seeds=len(complete_seeds(by_seed)),
            n_foreign=sum(1 for r in rs if r.foreign_overlap),
        )
    return cells


# A cluster cell's metrics that have a fixed-membership twin (runlogs.Run.*_fixed).
FIXED_TWINS = (("mse", "mse_fixed"), ("mae", "mae_fixed"), ("ic", "ic_fixed"),
               ("jc", "jc_fixed"), ("interval_width", "interval_width_fixed"))


def lacks_fixed(run: Run) -> bool:
    """Whether a cluster run has no fixed-membership MSE/MAE (fixed_source "none")."""
    return run.k is not None and (run.mse_fixed is None or run.mae_fixed is None)


def fixed_cells(runs: list[Run], cells: dict[tuple, Cell]) -> dict[tuple, Cell]:
    """A cluster tree's cells under the fixed train-mean membership, the primary protocol.

    ``cells`` is aggregate(runs). Each cluster cell's mse, mae, ic, jc and
    interval_width are swapped for the mean of its runs' fixed-membership
    twins (FIXED_TWINS): every channel keeps the cluster weights computed
    once from the training split, so a forecast depends on its own window
    only. A twin is all or nothing, as in cells.csv (frames.build_cells): a
    cell with a run that lacks it prints `-` (not logged) there, never a mean
    over the other seeds. Time, epochs, size and seed counts are the cell's
    own, over every run: they have no protocol. Cells outside a cluster tree
    (k None) pass unchanged.
    """
    out = dict(cells)
    for key, rs in group_runs(runs).items():
        if key[5] is None:
            continue
        swap = {}
        for attr, twin in FIXED_TWINS:
            vals = [getattr(r, twin) for r in rs]
            swap[attr] = (statistics.mean(vals) if all(v is not None for v in vals)
                          else None)
        out[key] = replace(cells[key], **swap)
    return out


def seed_targets(*cell_maps: dict[tuple, Cell]) -> dict[tuple, int]:
    """{(dataset, L, O, conf): how many seeds a complete cell there has}.

    The seeds any cell of that dataset and window has, over every tree passed
    in: a cluster sweep that has reached one seed of three is judged against
    the three the finished trees have, not against itself. Works on raw and
    canonical_cells keys alike (both put dataset, L, O, conf at 1-4).
    """
    seen: dict[tuple, set] = defaultdict(set)
    for cells in cell_maps:
        for key, cell in cells.items():
            seen[(key[1], key[2], key[3], key[4])].update(cell.seeds)
    return {k: len(v) for k, v in seen.items()}


def shortfall(cell: Optional[Cell], key: tuple, targets: Optional[dict]) -> str:
    """'' for a complete cell, else the ' (n=…)' it is printed with.

    A cell is short when it has fewer complete seeds (every band, one run
    each) than seed_targets gives its dataset and window, or runs of a seed
    that is not complete: it is still training, and its mean is over
    whichever runs finished. n is its complete seeds; '+ partial' says runs
    of an incomplete seed are averaged in too. Such a cell is never ranked,
    bolded or pooled with complete ones.
    """
    if cell is None or not targets:
        return ""
    target = targets.get((key[1], key[2], key[3], key[4]), 0)
    full = (cell.n_complete_seeds if cell.n_complete_seeds is not None
            else cell.n_seeds or 0)
    partial = len(cell.seeds) > full
    if not target or (full >= target and not partial):
        return ""
    return f" (n={full}{' + partial' if partial else ''})"


# The legend for shortfall's marker, added wherever one is printed.
SHORT_NOTE = ("`(n=…)` = a cell with fewer complete seeds than the other cells of its "
              "dataset and window, still training: its mean is over the runs that "
              "finished, and it is never ranked, marked or averaged with complete cells.")

# The timed columns. A cell whose runs overlapped another user's process on
# the GPU (Cell.n_foreign, from specs/contention.yaml) carries FOREIGN_MARK
# there. Sharing the card only adds wall time, so its time is an upper bound
# on what it would have taken alone, and it is ranked as logged: a place it
# wins it would keep uncontended, and one it loses only says it may be
# faster than it shows. Leaving it out of the ranking instead handed its
# mark to a cell slower than it. Its accuracy is not marked -- sharing a GPU
# changes scheduling, not arithmetic.
TIME_ATTRS = ("running_time", "per_epoch")
FOREIGN_MARK = " ‡"
FOREIGN_NOTE = ("‡ = a cell with a run that trained while another user's process was on "
                "the GPU (`specs/contention.yaml`): its time can only be too high, as "
                "sharing the card adds wall time, so it is ranked as logged. A ‡ time "
                "ranked ahead of another would stay ahead uncontended; one ranked behind "
                "may belong ahead. Its accuracy is unaffected.")


def foreign_mark(cell: Optional[Cell]) -> str:
    """FOREIGN_MARK for a cell with a foreign-contended run, else ''."""
    return FOREIGN_MARK if cell is not None and getattr(cell, "n_foreign", 0) else ""


def any_foreign(cells) -> bool:
    """Whether any of these cells has a foreign-contended run."""
    return any(getattr(c, "n_foreign", 0) for c in cells if c is not None)


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


def metric_rows(cells, models, row_keys, conf, attr, fmt, mark_best: bool,
                targets: Optional[dict] = None) -> list[list[str]]:
    show_k = shows_k(row_keys)
    rows = []
    for ds, sl, pl, k in row_keys:
        keys = [(m, ds, sl, pl, conf, k) for m in models]
        vals = [getattr(cells.get(key), attr, None) for key in keys]
        shorts = [shortfall(cells.get(key), key, targets) if v is not None else ""
                  for key, v in zip(keys, vals)]
        strs = [fmt(v) for v in vals]
        if mark_best:
            # Ranked within one k: the models of a cluster row compete at the same
            # cluster count, not against the other k's of the same window. A
            # cell still training is not ranked against finished ones; a time
            # another user's GPU process may have inflated is, as the upper
            # bound it is (FOREIGN_NOTE).
            mark_ranks(strs, [None if short else v for v, short in zip(vals, shorts)])
        strs = [s + short + (foreign_mark(cells.get(key))
                             if attr in TIME_ATTRS and v is not None else "")
                for s, short, key, v in zip(strs, shorts, keys, vals)]
        head = [dataset_label(ds), str(sl), str(pl)]
        if show_k:
            head.append(MISSING if k is None else str(k))
        rows.append(head + strs)
    return rows


def metric_table(cells, models, row_keys, conf, attr, fmt, mark_best=True,
                 targets: Optional[dict] = None) -> list[str]:
    show_k = shows_k(row_keys)
    return md_table(
        ["Dataset", "$L$", "$O$"] + (["$k$"] if show_k else []) + [model_label(m) for m in models],
        "lrr" + ("r" if show_k else "") + "r" * len(models),
        metric_rows(cells, models, row_keys, conf, attr, fmt, mark_best, targets),
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


# The pooled epoch-cost table's marker for a row that pools a cell still
# training, added to the table's note wherever one is printed.
POOLED_SHORT_NOTE = ("`(n=c/C cells)`: c of the C cells the row pools have every seed; "
                     "the others are still training, and the runs of theirs that "
                     "finished are pooled in, so the mean is not final.")


def pooled_shortfall(runs, cells: Optional[dict], targets: Optional[dict]) -> str:
    """'' when every cell these runs belong to is complete, else ' (n=c/C cells)'.

    The pooled-row twin of shortfall: c of the C cells the runs come from are
    complete (shortfall is empty for them).
    """
    if not cells or not targets:
        return ""
    keys = {(r.model, r.dataset, r.seq_len, r.pred_len, r.conf, r.k) for r in runs}
    done = sum(1 for key in keys if not shortfall(cells.get(key), key, targets))
    return "" if done == len(keys) else f" (n={done}/{len(keys)} cells)"


def epoch_summary_table(runs, models, conf, cells: Optional[dict] = None,
                        targets: Optional[dict] = None) -> tuple[list[str], bool]:
    """(the mean-epoch-cost table, whether a row pools a cell still training).

    Split by k as well as model: CCM builds k parallel per-cluster layers, so
    epoch cost grows with k and pooling k=2 with k=9 would report neither. A
    row pools every run of its (model, k) in the section; with ``cells`` and
    ``targets`` one that pools a cell still training carries
    pooled_shortfall's marker.
    """
    rows = []
    for m, k in model_variants({(r.model, None, None, None, None, r.k) for r in runs}, models):
        mine = [r for r in runs
                if r.model == m and r.k == k and r.conf == conf and r.per_epoch is not None]
        if mine:
            foreign = sum(1 for r in mine if r.foreign_overlap)
            rows.append((variant_label(m, k), statistics.mean(r.per_epoch for r in mine),
                         f"{len(mine)}" + (f" ({foreign}{FOREIGN_MARK})" if foreign else ""),
                         pooled_shortfall(mine, cells, targets)))
    table = md_table(
        ["Model", "Mean / epoch", "Estimation for 100 ep", "Runs"],
        "lrrr",
        [[label, format_seconds_fine(mean) + note, f"~{format_seconds(mean * 100)}", n]
         for label, mean, n, note in sorted(rows, key=lambda r: r[1])],
    )
    return table, any(note for *_, note in rows)


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


# How a cluster section names its two protocols (render_report). The fixed
# train-mean membership is the primary one (the owner's decision of
# 2026-09-24); the logged batch protocol is kept, labelled, as a sensitivity.
FIXED_PROTOCOL_TEXT = (
    "the fixed train-mean membership (`sl_test_fixed/*`), the primary cluster protocol: "
    "every channel keeps the cluster weights computed once from the training split, so a "
    "forecast depends on its own window only. The logged batch protocol (`sl_test/*`) "
    "is the last group, a sensitivity; at k = 1 the two are the same forecast. The k "
    "chosen on validation (`sl_val_fixed/loss`) is in `Channel Clustering (CCM) vs. "
    "plain`, `CCM vs Base (No Clustering)` and `Paradigm Comparison`")
BATCH_GROUP = "Batch protocol (sensitivity)"
BATCH_PROTOCOL_TEXT = (
    "The same runs under the logged batch protocol (`sl_test/*`), kept as a sensitivity: "
    "upstream's assigner averages the cluster membership over each evaluation batch, so a "
    "k > 1 forecast depends on the other windows of its batch, and on a batch size that "
    "differs by model. At k = 1 the membership is all ones either way, so the two "
    "protocols give the same forecast.")


# What the single-series tree is and is not (.personal/AUDIT_2026-09-24.md
# S1, S2, S3/E8, S7), printed at the top of its section: it is a different
# dataset family, trained before the comparability contract, and one of its
# three columns is another shifted by one sample.
SERIES_CAVEATS = [
    "a separate dataset -- one broadband series per probe (det, ptv, ptv2), not the "
    "27/29 bands of ED2022/UH2022/EMF2023, on its own sampling grid and calendar. Its "
    "numbers are never pooled or ranked with the multivariate, per-channel or cluster "
    "ones",
    "ptv is det shifted by one 6-minute step (correlation 0.991 at +6 min), so the tree "
    "holds 2 independent series, not 3: count det and ptv as one site. Their columns "
    "are not duplicate numbers -- each is tested on its own calendar window, and ptv's "
    "lies inside det's training window",
    "out of contract -- no run is provenance-stamped (its three training batches ran "
    "at different code versions), PatchTST and the Transformer trained compiled, the "
    "Transformer has no RevIN in this code (the logged revin=True does nothing) and "
    "trained at batch 512 (128 with RevIN in the joint grids), and every run predates "
    "the +1 window fix. L336/O96 has 5 seeds per cell, every other window 1 (69421)",
    "collapsed seeds are averaged in: the series Transformer collapses to about the "
    "training mean at 11 of its 15 L336/O96 seeds, so its 5-seed means mix collapsed "
    "and trained seeds",
]


def concurrency_text(runs: list[Run]) -> str:
    """How many trainings shared the GPU with these runs, from their stamps.

    The per-channel grid trains several bands at once (``concurrency`` in the
    run log, stamped since 2026-09-19), every other grid one run at a time,
    so a per-channel running_time is a contended time.
    """
    vals = sorted({r.concurrency for r in runs if r.concurrency is not None})
    unstamped = sum(1 for r in runs if r.concurrency is None)
    parts = []
    if vals:
        parts.append(" or ".join(str(v) for v in vals) + " trainings at a time")
    if unstamped:
        parts.append(f"{unstamped} runs logged before that was stamped")
    return "; ".join(parts) or "not stamped"


def per_band_note(runs: list[Run]) -> str:
    """The label a per-channel tree's timing tables need, or "" for any other tree."""
    if not runs or any(r.channel is None for r in runs):
        return ""
    return (f" Per-channel times are per band and contended: each is one band's model, "
            f"trained while other bands shared the GPU ({concurrency_text(runs)}), so it "
            f"is neither the cost of forecasting every band nor comparable with a run "
            f"trained alone. `frames/timing.csv` holds the uncontended estimate where the "
            f"solo timing pass measured one (EMF, L336/O96).")


def render_report(title: str, runs: list[Run], cells: dict[tuple, Cell],
                  extra: Optional[list[tuple]] = None,
                  targets: Optional[dict] = None,
                  batch: Optional[dict[tuple, Cell]] = None) -> str:
    """One setup's section. ``extra`` appends (group, table title, lines, note)
    groups after the model-size table -- what only the frames path knows.
    ``targets`` (seed_targets over every tree loaded; default: this tree's
    cells alone) is the seed count a complete cell has.

    ``batch`` is for a cluster tree whose ``cells`` are its fixed_cells, the
    primary protocol: the same runs' logged batch-protocol cells, printed as
    the section's last group, a labelled sensitivity. A cluster tree
    rendered without it is the batch protocol, and says so."""
    if targets is None:
        targets = seed_targets(cells, *([batch] if batch else []))
    short = any(shortfall(c, key, targets) for key, c in cells.items())
    legend = f" {SHORT_NOTE}" if short else ""
    models = sort_models({k[0] for k in cells} | {k[0] for k in batch or {}})
    confs = [c for c in (False, True) if row_keys_for(cells, c)]
    settings = sorted({(r.seq_len, r.pred_len) for r in runs})
    datasets = sorted({r.dataset for r in runs}, key=dataset_rank)

    ks = sorted({r.k for r in runs if r.k is not None})
    # The primary numbers' log keys: sl_test_fixed/* for a cluster tree
    # rendered from its fixed view, sl_test/* for every other tree.
    fixed = bool(ks) and batch is not None
    test_key = "sl_test_fixed" if fixed else "sl_test"

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
        items.append(("Protocol", FIXED_PROTOCOL_TEXT if fixed else
                      "the logged batch protocol (`sl_test/*`): the cluster membership is "
                      "averaged over each evaluation batch, so a k > 1 forecast depends on "
                      "its neighbouring windows. The primary fixed train-mean membership "
                      "numbers (`sl_test_fixed/*`), with k chosen on validation, are in "
                      "`Paradigm Comparison` and `Channel Clustering (CCM) vs. plain`; at "
                      "k = 1 the two are the same forecast"))
    items.append(("Runs", coverage_summary(runs, cells)))
    if title == UNIVARIATE_SERIES_TITLE:
        items += [("Caveat", text) for text in SERIES_CAVEATS]
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
                 f"`{MISSING}` = not logged." + legend)
        doc.table(f"MSE (`{test_key}/loss`)",
                  metric_table(cells, models, row_keys, conf, "mse", num_fmt(4),
                               targets=targets))
        if has_metric(cells, models, row_keys, conf, "mae"):
            doc.table(f"MAE (`{test_key}/mae`)",
                      metric_table(cells, models, row_keys, conf, "mae", num_fmt(4),
                                   targets=targets))

        doc.group("Training cost" + suffix)
        foreign = any_foreign(c for key, c in cells.items() if key[4] == conf)
        doc.para("**Bold** = fastest in the row, <u>underline</u> = runner-up."
                 + (f" `{MISSING}` = not logged (ARIMA is not gradient-trained)."
                    if "ARIMA" in models else "") + per_band_note(runs) + legend
                 + (f" {FOREIGN_NOTE}" if foreign else ""))
        doc.table("Total training time (`parameters/running_time`)",
                  metric_table(cells, models, row_keys, conf, "running_time", format_seconds,
                               targets=targets))
        doc.table("Per-epoch training time (`running_time / len(sl_train/loss)`)",
                  metric_table(cells, models, row_keys, conf, "per_epoch", format_seconds_fine,
                               targets=targets))
        epochs, pooled_short = epoch_summary_table(runs, models, conf, cells, targets)
        doc.table("Mean epoch cost by model", epochs,
                  note="Pooled over every run of that model in this section."
                       + (f" {POOLED_SHORT_NOTE}" if pooled_short else "")
                       + (f" `(m{FOREIGN_MARK})` after the runs: m of them trained while "
                          f"another user's process was on the GPU, so the mean may be "
                          f"inflated." if foreign else ""))

        if conf:
            doc.group("Conformal prediction")
            doc.para('Not marked -- "best" coverage means closest to the target, not highest, and '
                     "a narrower interval only counts if that target is met." + legend)
            conformal_tables(doc, cells, models, row_keys, test_key, targets)

    sizes = model_size_table(cells, models)
    if sizes:
        by = "(model, k, L, O)" if ks else "(model, L, O)"
        doc.group("Model size")
        doc.table("Trainable weights (`parameters/sl/num_params`)", sizes,
                  note=f"For multivariate logs this varies by dataset (different channel counts), "
                       f"not just by {by}; shown here is the first value found per {by}.")

    for group, table_title, lines, note in extra or []:
        doc.group(group)
        doc.table(table_title, lines, note=note)

    if fixed:
        doc.group(BATCH_GROUP)
        short = any(shortfall(c, key, targets) for key, c in batch.items())
        doc.para(BATCH_PROTOCOL_TEXT + " **Bold** = best (lowest) in the row, <u>underline</u> "
                 "= runner-up; coverage and width are not marked."
                 + (f" {SHORT_NOTE}" if short else ""))
        batch_confs = [c for c in (False, True) if row_keys_for(batch, c)]
        for conf in batch_confs:
            suffix = ""
            if len(batch_confs) > 1:
                suffix = (", with conformal prediction" if conf
                          else ", without conformal prediction")
            row_keys = row_keys_for(batch, conf)
            doc.table(f"MSE (`sl_test/loss`){suffix}",
                      metric_table(batch, models, row_keys, conf, "mse", num_fmt(4),
                                   targets=targets))
            if has_metric(batch, models, row_keys, conf, "mae"):
                doc.table(f"MAE (`sl_test/mae`){suffix}",
                          metric_table(batch, models, row_keys, conf, "mae", num_fmt(4),
                                       targets=targets))
            if conf:
                conformal_tables(doc, batch, models, row_keys, "sl_test", targets)

    return doc.render()


def conformal_tables(doc: Doc, cells, models, row_keys, test_key: str,
                     targets: Optional[dict]) -> None:
    """The three conformal tables of one section, unmarked, from ``test_key``'s logs."""
    for name, attr, key, decimals in (
            ("Marginal coverage %", "ic", "ic", 2),
            ("Joint coverage %", "jc", "jc", 2),
            ("Mean interval width", "interval_width", "interval_width_mean", 3)):
        if has_metric(cells, models, row_keys, True, attr):
            doc.table(f"{name} (`{test_key}/{key}`)",
                      metric_table(cells, models, row_keys, True, attr, num_fmt(decimals),
                                   mark_best=False, targets=targets))


def render_setup(title: str, runs: list[Run], cells: dict[tuple, Cell],
                 targets: Optional[dict] = None,
                 extra: Optional[list[tuple]] = None) -> str:
    """render_report for any tree; a cluster tree leads with its fixed-protocol view.

    ``cells`` is aggregate(runs). A cluster tree's section prints fixed_cells'
    numbers as its primary tables and ``cells``, the logged batch protocol, as
    its last group. A cluster tree none of whose runs has fixed metrics (frames
    written before those columns existed) is rendered under the batch protocol,
    labelled as such, with a warning.
    """
    if not any(r.k is not None for r in runs):
        return render_report(title, runs, cells, extra=extra, targets=targets)
    if all(lacks_fixed(r) for r in runs if r.k is not None):
        print(f"WARNING: no run of '{title}' has fixed-membership metrics: rendered under "
              f"the logged batch protocol, labelled as such. Re-extract the frames "
              f"(make.py extract) for the primary protocol.", file=sys.stderr)
        return render_report(title, runs, cells, extra=extra, targets=targets)
    missing = sum(lacks_fixed(r) for r in runs)
    if missing:
        print(f"WARNING: {missing} runs of '{title}' have no fixed-membership metrics "
              f"(fixed_source none): their cells print '-' under the fixed protocol, as "
              f"cells.csv leaves them blank; the batch-protocol group and the cost "
              f"tables keep them. Re-score them (forecasting/ccm_rescore.py).",
              file=sys.stderr)
    return render_report(title, runs, fixed_cells(runs, cells), extra=extra,
                         targets=targets, batch=cells)


def protocol_cells(runs: list[Run], cells: dict[tuple, Cell]) -> tuple:
    """(primary cells, batch cells or None) of a cluster tree, for the cross tables.

    ``cells`` is aggregate(runs). The primary cells are fixed_cells', and the
    logged ones are kept as the batch-protocol sensitivity; when no run has
    fixed metrics (frames written before those columns existed) the logged
    cells are all there is, and no sensitivity is returned.
    """
    if any(r.k is not None and not lacks_fixed(r) for r in runs):
        return fixed_cells(runs, cells), cells
    return cells, None


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


def pair_rows(keys, left, right, attr, fmt, lower_is_better: bool, delta: str,
              targets: Optional[dict] = None):
    """wins is [left, right, tie, pending]: a pair with a side still training
    (shortfall) is pending, never a win."""
    rows, wins = [], [0, 0, 0, 0]
    for key in keys:
        a = getattr(left.get(key), attr, None)
        b = getattr(right.get(key), attr, None)
        na = shortfall(left.get(key), key, targets) if a is not None else ""
        nb = shortfall(right.get(key), key, targets) if b is not None else ""
        sa, sb, sd = fmt(a) + na, fmt(b) + nb, MISSING
        if a is not None and b is not None:
            if delta == "pct" and a:
                sd = fmt_delta((b - a) / a * 100, 1, "%")
            elif delta == "pp":
                sd = fmt_delta(b - a, 2, " pp")
            if lower_is_better and (na or nb):
                wins[3] += 1
            elif lower_is_better:
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
               lower_is_better=True, delta="pct", targets=None):
    rows, wins = pair_rows(keys, left, right, attr, fmt, lower_is_better, delta, targets)
    return md_table(["Dataset", "$L$", "$O$", "Model", left_label, right_label, "Δ"],
                    "lrrlrrr", rows), wins


def render_comparison(title: str, mv_title: str, mv_cells: dict[tuple, Cell],
                      uv_roots: list[tuple[str, dict[tuple, Cell]]],
                      concurrency: Optional[dict[str, str]] = None,
                      targets: Optional[dict] = None) -> str:
    """``concurrency`` maps a univariate tree's title to concurrency_text of its
    runs; ``targets`` is seed_targets over every tree (default: these)."""
    mv = canonical_cells(mv_cells)
    matched = [(label, canonical_cells(c)) for label, c in uv_roots]
    if targets is None:
        targets = seed_targets(mv, *(c for _, c in matched))
    matched = [(label, c, shared_keys(mv, c)) for label, c in matched]
    matched = [(label, c, keys) for label, c, keys in matched if keys]
    if not matched:
        return ""
    short = any(shortfall(side.get(key), key, targets)
                for _, c, keys in matched for key in keys for side in (mv, c))

    left = short_label(mv_title)
    doc = Doc(title)
    doc.preamble([
        ("Compares", " · ".join(f"{left} vs {short_label(label)}" for label, _, _ in matched)),
        ("Matched on", "(dataset, L, O, model, conformal setting); only cells present in both "
                       "trees appear below"),
        ("Delta", f"Δ is the univariate value relative to {left}, so negative Δ means the "
                  "univariate value is lower; **bold** = better of the two where lower is better"
                  + (f". {SHORT_NOTE[:-1]}; a pair with such a side is pending in the "
                     f"scoreboard" if short else "")),
    ])

    all_wins: list[tuple[str, str, list[int]]] = []
    for label, uv, keys in matched:
        right = short_label(label)
        pair = f"{left} vs {right}"

        doc.group(f"Forecast accuracy ({pair})")
        doc.para(f"{len(keys)} matched cells, from `{mv_title}` and `{label}`.")
        for name, log_key, attr, decimals in [("MSE", "sl_test/loss", "mse", 4),
                                              ("MAE", "sl_test/mae", "mae", 4)]:
            table, wins = pair_table(keys, mv, uv, left, right, attr, num_fmt(decimals),
                                     targets=targets)
            doc.table(f"{name} (`{log_key}`)", table)
            all_wins.append((right, name, wins))

        doc.group(f"Training cost ({pair})")
        conc = (concurrency or {}).get(label)
        foreign = any_foreign(side.get(key) for key in keys for side in (mv, uv))
        doc.para(f"`{right}` is the mean cost of one channel's model; `all channels` multiplies it "
                 f"by the channel count. Δ compares that total against {left}."
                 + (f" The {right.lower()} runs trained alongside other bands ({conc}), "
                    f"the {left.lower()} ones one run at a time, so their times are contended "
                    f"and Δ is not a like-for-like cost; `frames/timing.csv` holds the "
                    f"uncontended estimate where the solo timing pass measured one (EMF, "
                    f"L336/O96)." if conc else "")
                 + (f" {FOREIGN_NOTE}" if foreign else ""))
        rows = []
        for key in keys:
            a = getattr(mv.get(key), "running_time", None)
            b = getattr(uv.get(key), "running_time", None)
            chans = getattr(uv.get(key), "n_channels", None)
            total = b * chans if b is not None and chans else None
            sd = fmt_delta((total - a) / a * 100, 1, "%") if a and total is not None else MISSING
            model, ds, sl, pl = key[0], key[1], key[2], key[3]
            ma = foreign_mark(mv.get(key)) if a is not None else ""
            mb = foreign_mark(uv.get(key)) if b is not None else ""
            rows.append([dataset_label(ds), str(sl), str(pl), model, format_seconds(a) + ma,
                         format_seconds(b) + mb, str(chans) if chans else MISSING,
                         format_seconds(total) + mb, sd])
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
                                             num_fmt(decimals), lower_is_better=lower, delta=kind,
                                             targets=targets)
                    doc.table(f"{name} (`{log_key}`)", table)
                    if lower:
                        all_wins.append((right, "Interval width", wins))

    doc.group("Scoreboard")
    pending = any(w[3] for _, _, w in all_wins)
    doc.table("Matched-cell wins by metric",
              md_table(["Comparison", "Metric", left, "Univariate", "Tie"]
                       + (["Pending"] if pending else []), "llrrr" + ("r" if pending else ""),
                       [[f"{left} vs {label}", metric, str(w[0]), str(w[1]), str(w[2])]
                        + ([str(w[3])] if pending else [])
                        for label, metric, w in all_wins]),
              note="How many matched cells each pipeline wins, counted only where lower is better."
                   + (" Pending: a side still training, not counted." if pending else ""))
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


def ultimate_rows(pipelines, models, datasets, sl, pl, conf, attr, fmt,
                  targets: Optional[dict] = None) -> list[list[str]]:
    rows = []
    for ds in datasets:
        for p in pipelines:
            vals = [pipeline_value(p, m, ds, sl, pl, conf, attr) for m in models]
            if all(v is None for v in vals):
                continue
            keys = [(m, ds, sl, pl, conf, p.k) for m in models]
            shorts = [shortfall(p.cells.get(key), key, targets) if v is not None else ""
                      for key, v in zip(keys, vals)]
            strs = [fmt(v) for v in vals]
            # Ranked across the row, i.e. model vs model under one strategy; the
            # strategies are compared by reading down a dataset block instead.
            # A cell still training is not ranked against finished ones; a time
            # another user's GPU process may have inflated is, as the upper
            # bound it is (FOREIGN_NOTE).
            mark_ranks(strs, [None if short else v for v, short in zip(vals, shorts)])
            strs = [s + short + (foreign_mark(p.cells.get(key))
                                 if attr in TIME_ATTRS and v is not None else "")
                    for s, short, key, v in zip(strs, shorts, keys, vals)]
            rows.append([dataset_label(ds), p.label] + strs)
    return rows


def ultimate_table(pipelines, models, datasets, sl, pl, conf, attr, fmt,
                   targets: Optional[dict] = None) -> list[str]:
    rows = ultimate_rows(pipelines, models, datasets, sl, pl, conf, attr, fmt, targets)
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


def render_ultimate(title: str, pipelines: list[Pipeline],
                    pc_concurrency: Optional[str] = None,
                    targets: Optional[dict] = None,
                    batch: Optional[list[Pipeline]] = None) -> str:
    """``pc_concurrency`` is concurrency_text of the per-channel runs;
    ``targets`` is seed_targets over every tree (default: the pipelines').

    ``batch`` is for cluster pipelines built from fixed_cells, the
    primary protocol: the same k's under the logged batch protocol
    (cluster_pipelines of the logged cells), printed after the time table as
    a labelled sensitivity. Without it, the cluster rows are whatever cells
    they were built from, and the section says they are the batch protocol."""
    sl, pl = ULTIMATE_SETTING
    axes = [(conf,) + ultimate_axes(pipelines, sl, pl, conf) for conf in (False, True)]
    axes = [a for a in axes if a[1]]
    if not axes:
        return ""
    if targets is None:
        targets = seed_targets(*(p.cells for p in pipelines))
    short = any(shortfall(cell, key, targets) for p in pipelines
                for key, cell in p.cells.items()
                if key[2] == sl and key[3] == pl and key[5] == p.k)

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
                       f"`Channel Clustering (CCM) vs. plain`)."
                       + (" Runs made before 2026-08-25 by the upstream wrappers are in "
                          "logs/_archive and are not in these tables."
                          if any(family_of(d) == "emf" for d in axes[0][2]) else "")))
    if any(p.k is not None for p in pipelines):
        items.append(
            ("Cluster protocol",
             "the fixed train-mean membership (`sl_test_fixed/*`), the primary one: every "
             "channel keeps the cluster weights computed once from the training split. The "
             "cluster rows under the logged batch protocol (`sl_test/*`), which averages "
             "the membership over each evaluation batch, follow the time table, a "
             "sensitivity. The k chosen on validation is in `CCM vs Base (No "
             "Clustering)`, `Channel Clustering (CCM) vs. plain` and `Paradigm Comparison`"
             if batch else
             "the logged batch protocol (`sl_test/*`); the fixed-membership numbers, the "
             "primary protocol, are in `Paradigm Comparison`"))
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
                 f"there is nothing to cluster)." + (f" {SHORT_NOTE}" if short else ""))

        accuracy = [("MSE", "loss", "mse", num_fmt(4)), ("MAE", "mae", "mae", num_fmt(4))]
        for name, key, attr, fmt in accuracy:
            table = ultimate_table(pipelines, models, datasets, sl, pl, conf, attr, fmt,
                                   targets)
            if table:
                doc.table(f"{name} (`sl_test/{key}`"
                          + (f"; cluster rows `sl_test_fixed/{key}`" if batch else "") + ")",
                          table)

        table = ultimate_table(pipelines, models, datasets, sl, pl, conf,
                               "running_time", format_seconds, targets)
        if table:
            chans = channel_counts(pipelines, datasets, sl, pl, conf)
            foreign = any_foreign(cell for p in pipelines for key, cell in p.cells.items()
                                  if key[2] == sl and key[3] == pl and key[4] == conf
                                  and key[5] == p.k and key[0] in models
                                  and key[1] in datasets)
            doc.table("Total training time (`parameters/running_time`)", table,
                      note=f"Cost of the whole strategy, so the per-channel row is the mean "
                           f"single-channel run times the channel count"
                           + (f" ({chans})" if chans else "") + "; every other row is one "
                           f"model's training run, averaged over seeds. ARIMA is not gradient-trained and "
                           f"logs no time."
                           + (f" The per-channel bands trained alongside each other "
                              f"({pc_concurrency}), every other row one run at a time, so "
                              f"the per-channel row is a contended time."
                              if pc_concurrency else "")
                           + (f" {FOREIGN_NOTE}" if foreign else ""))

        for name, key, attr, fmt in accuracy if batch else []:
            table = ultimate_table(batch, models, datasets, sl, pl, conf, attr, fmt, targets)
            if table:
                doc.table(f"Batch protocol (sensitivity): {name} of the cluster rows "
                          f"(`sl_test/{key}`)", table,
                          note="The cluster rows above under the logged batch protocol: "
                               "upstream's assigner averages the membership over each "
                               "evaluation batch, so a k > 1 forecast depends on the other "
                               "windows of its batch. Marked across the row, as above; at "
                               "k = 1 the two protocols give the same forecast.")

    return doc.render()


# The cluster cell whose logged parameters stand for a family's sweep in the
# CCM paper's configuration table (Table 10): the first dataset at a mid-range
# k. The EMF sweep runs k 1, 2, 3, 5, 9 over 27-29 bands; VoIP k 1, 2, 3, 6
# over 6 signals, where k = 5 was never run.
CCM_REPRESENTATIVE = {"emf": ("ed2022", 5), "voip": ("voip_mobile", 3)}


def validation_choices(runs: list[Run], protocol: str, window: tuple,
                       n_family: Optional[int] = None) -> dict:
    """{(model label, dataset): paradigms.Choice}: each sweep's k chosen on validation.

    paradigms.choose_k, the choice the paradigm comparison and the CCM sweep
    make, from Run objects: per (dataset, model) at ``window``, the k with the
    lowest seed-mean validation loss -- ``val_loss_fixed`` under the fixed
    protocol, the logged ``val_loss`` under the batch one -- over the k values
    that have every seed of the sweep. The test split never enters.
    ``n_family`` is the seed count a finished sweep has (default: every seed
    a cluster run at the window has). For a tree loaded without frames (the
    logs walk); make.py tables passes the paradigm comparison's own choices.
    """
    from emfres import paradigms as PG

    sweeps: dict = defaultdict(lambda: defaultdict(dict))
    seeds: set = set()
    for r in sorted(runs, key=lambda r: model_rank(r.model)):
        if (r.k is None or (r.seq_len, r.pred_len) != tuple(window) or not r.conf
                or r.seed is None):
            continue
        if protocol == "fixed":
            if r.mse_fixed is None or r.mae_fixed is None:
                continue
            val = r.val_loss_fixed
        else:
            if r.mse is None or r.mae is None:
                continue
            val = r.val_loss
        sweeps[(model_label(r.model), r.dataset)][r.k].setdefault(
            r.seed, {"val": val, "source": r.fixed_source})
        seeds.add(r.seed)
    grid = tuple(sorted({k for ks in sweeps.values() for k in ks}))
    n = n_family if n_family is not None else len(seeds)
    return {cell: PG.choose_k(ks, protocol, grid, n) for cell, ks in sweeps.items()}


def choice_lacks(choice, cell: Optional[Cell], key: tuple, targets: Optional[dict]) -> str:
    """What a + CCM cell lacks to count as complete, or "" when it lacks nothing.

    A choice ranked over fewer k values or seeds than the family's (still
    training) says so as the CCM sweep table does; a chosen cell short of its
    dataset's seeds says its n.
    """
    if choice.partial:
        return (f"ranked {','.join(map(str, choice.ranked))} of "
                f"{','.join(map(str, choice.grid))}, "
                f"{len(choice.seeds)}/{choice.n_family} seeds")
    return shortfall(cell, key, targets).strip(" ()")


# The protocols the CCM vs Base section prints, and the log keys it names.
CCM_PROTOCOLS = {
    "fixed": ("the fixed train-mean membership", "sl_test_fixed", "sl_val_fixed/loss"),
    "batch": ("the logged batch protocol", "sl_test", "sl_val/loss"),
}


def render_ccm_vs_base(title: str, cluster_cells: dict[tuple, Cell],
                       mv_cells: dict[tuple, Cell], cluster_runs: list[Run],
                       targets: Optional[dict] = None, *,
                       choices: Optional[dict] = None,
                       batch: Optional[dict[tuple, Cell]] = None,
                       batch_choices: Optional[dict] = None) -> str:
    """The CCM paper's Table 4 shape: base | + CCM per backbone, IMP(%) last.

    Every cells input is canonical_cells-keyed. "Base" is the plain
    multivariate cell from the '# Multivariate' tree - the same job config,
    byte-identical training hyperparameters, only the network swapped for
    upstream's CCM wrapper. The + CCM column takes, per (dataset, model), the
    k chosen on validation (``choices``, {(model label, dataset):
    paradigms.Choice}; default validation_choices over ``cluster_runs``) and
    reuses it for the MAE row, as the paper tunes k on validation data. The
    test split never picks it: the configuration table reports the k it would
    pick as an oracle, labelled, and nothing else uses it.

    ``cluster_cells`` is the primary protocol's: extract_results.fixed_cells
    when ``batch``, the same runs' logged batch-protocol cells, is
    given -- then a second table repeats the comparison under the batch
    protocol, k chosen on its own validation loss (``batch_choices``), as a
    labelled sensitivity. Without ``batch`` the cells are the batch
    protocol's, and the section says so. A second (then third) table mirrors
    the paper's Table 10 configuration summary, read from the cluster runs'
    own logs.

    Partial data (a sweep still training) is shown, never scored as complete:
    a base cell short of seeds prints its n (shortfall), a + CCM cell whose
    k was ranked over fewer k values or seeds than the family's, or whose
    chosen cell is short, prints what it lacks (choice_lacks), and neither is
    marked or averaged into IMP. ``targets`` is seed_targets over every tree
    (default: these).
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
    if targets is None:
        targets = seed_targets(cluster_cells, mv_cells, *([batch] if batch else []))
    # A finished sweep's seed count: the most any tree has on these datasets
    # (not the single-series tree's, a separate family with its own seeds).
    n_family = max((targets.get((ds, sl, pl, conf), 0) for ds in datasets), default=0) or None
    protocol = "fixed" if batch is not None else "batch"
    if choices is None:
        choices = validation_choices(cluster_runs, protocol, (sl, pl), n_family)
    if batch is not None and batch_choices is None:
        batch_choices = validation_choices(cluster_runs, "batch", (sl, pl), n_family)
    # The family's k values at this window, for the configuration table.
    grid = sorted({k[5] for k in cluster_cells if at(k) and k[5] is not None})

    family = family_of(datasets[0])
    rep_ds, rep_k = CCM_REPRESENTATIVE.get(family, (datasets[0], None))
    if rep_ds not in datasets:
        rep_ds = datasets[0]
    windows = sorted({(r.seq_len, r.pred_len) for r in cluster_runs
                      if family_of(r.dataset) == family}, reverse=True)
    others = [setting_label(*w) for w in windows if w != (sl, pl)]
    name, test_key, val_key = CCM_PROTOCOLS[protocol]

    doc = Doc(title)
    doc.preamble([
        ("Format", "the CCM paper's Table 4 - per backbone a base | + CCM column "
                   "pair, **bold** = better of the pair (neither when the two print "
                   "the same), <u>underline</u> = best in the row, last column the "
                   "mean % improvement of CCM over the base models"),
        ("Base", f"the plain multivariate cell from '# {MULTIVARIATE_TITLE}' - same "
                 f"job config and training hyperparameters, no CCM"),
        ("Configuration", f"{setting_label(sl, pl)}"
                          + (f"; the k sweep also ran at {', '.join(others)}, in "
                             f"`Channel Clustering (CCM) vs. plain`" if others
                             else " -- the window the k sweep covers")),
        ("Protocol", f"{name} (`{test_key}/*`)"
                     + (", the primary cluster protocol: every channel keeps the cluster "
                        "weights computed once from the training split. The logged batch "
                        "protocol (`sl_test/*`), which averages the membership over each "
                        "evaluation batch, is table 1.2, a sensitivity"
                        if batch is not None else
                        ": the cluster membership is averaged over each evaluation batch")),
        ("k", f"+ CCM is the cell at the k chosen per (dataset, model) on validation: the "
              f"lowest seed-mean `{val_key}` over the k values with every seed of the "
              f"sweep, reused for MAE, as the CCM paper tunes k on validation data "
              f"(paradigms.choose_k, the choice `Channel Clustering (CCM) vs. plain` and "
              f"`Paradigm Comparison` make). The test split never picks it; the "
              f"configuration table reports the k it would pick as an oracle, for "
              f"reference only. The base column is the plain multivariate cell; "
              f"`Cluster (k=1)` in the Ultimate Comparison is the same network with "
              f"clustering off and is the stricter baseline."),
        ("Conformal prediction", "on for every run"),
    ])

    doc.group("Base vs + CCM")

    headers = ["Dataset", "Metric"]
    for m in models:
        headers += [m, "+ CCM"]
    headers.append("IMP (%)")

    def table(cells: dict, picked: dict) -> tuple[list[str], bool]:
        rows = []
        partial = False
        for ds in datasets:
            for attr, metric in (("mse", "MSE"), ("mae", "MAE")):
                strs: list[str] = []
                vals: list[Optional[float]] = []
                imps = []
                pairs = 0
                for m in models:
                    base_key = (m, ds, sl, pl, conf, None)
                    base_cell = mv_cells.get(base_key)
                    base = getattr(base_cell, attr, None)
                    short = shortfall(base_cell, base_key, targets) if base is not None else ""
                    choice = picked.get((m, ds))
                    k = choice.k if choice is not None else None
                    key = (m, ds, sl, pl, conf, k)
                    ccm_cell = cells.get(key) if k is not None else None
                    ccm = getattr(ccm_cell, attr, None)
                    lacks = choice_lacks(choice, ccm_cell, key, targets) if ccm is not None else ""
                    bs, cs = fmt_num(base, 4) + short, fmt_num(ccm, 4)
                    if ccm is not None:
                        cs = f"{cs} (k={k}" + (f"; partial: {lacks}" if lacks else "") + ")"
                    elif choice is not None and choice.k is None:
                        cs = f"{MISSING} ({choice.tag()})"
                    partial |= bool(short or (ccm is not None and lacks))
                    if base is not None and ccm is not None:
                        pairs += 1
                        if not (short or lacks):
                            # Better of the pair as printed: two values that
                            # round alike at 4 decimals are a tie, unmarked.
                            if round(ccm, 4) < round(base, 4):
                                cs = f"**{cs}**"
                            elif round(base, 4) < round(ccm, 4):
                                bs = f"**{bs}**"
                            imps.append((base - ccm) / base * 100.0)
                    strs += [bs, cs]
                    vals += [None if short else base, None if lacks else ccm]
                ranked = sorted((v, i) for i, v in enumerate(vals) if v is not None)
                if ranked:
                    i = ranked[0][1]
                    strs[i] = f"<u>{strs[i]}</u>"
                imp = f"{statistics.mean(imps):+.2f}" if imps else MISSING
                if imps and len(imps) < pairs:
                    imp += f" ({len(imps)} of {pairs})"
                rows.append([dataset_label(ds), metric] + strs + [imp])
        return md_table(headers, "ll" + "r" * (2 * len(models) + 1), rows), partial

    def note(partial: bool) -> str:
        return ("IMP (%) = mean over the backbones of (base − CCM) / base × 100 "
                "for that row's metric; positive = CCM ahead."
                + (" A base cell short of its dataset's seeds shows `(n=…)`; a + CCM "
                   "cell whose k was ranked over fewer k values or seeds than the family "
                   "has -- a sweep still training -- shows what it lacks (`partial: "
                   "ranked … of …, n/N seeds`, as `Channel Clustering (CCM) vs. plain` "
                   "does), or its `n=…` when its chosen cell is short. Such cells are "
                   "not marked, and IMP averages only the complete pairs, counting them "
                   "when that is not every backbone." if partial else ""))

    lines, partial = table(cluster_cells, choices)
    doc.table(f"Long-horizon forecasting, base vs + CCM (`{test_key}/*`, k chosen on "
              f"`{val_key}`)", lines, note=note(partial))
    if batch is not None:
        lines, partial = table(batch, batch_choices)
        doc.table("Batch protocol (sensitivity): base vs + CCM (`sl_test/*`, k chosen on "
                  "`sl_val/loss`)", lines,
                  note="The same comparison under the logged batch protocol, with its "
                       "own k chosen on the logged validation loss: upstream's assigner "
                       "averages the membership over each evaluation batch, so a k > 1 "
                       "forecast depends on the other windows of its batch. "
                       + note(partial))

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

    def k_line(m: str, picked: dict) -> str:
        parts = []
        for ds in datasets:
            choice = picked.get((m, ds))
            k = choice.k if choice is not None else None
            parts.append(MISSING if k is None else f"{k} (partial)" if choice.partial
                         else str(k))
        return " / ".join(parts)

    def oracle_line(m: str) -> str:
        # The k with the lowest test MSE among the k values validation ranked:
        # what choosing on the test split would pick. Reported, never used.
        parts = []
        for ds in datasets:
            choice = choices.get((m, ds))
            scored = [(cluster_cells[key].mse, key[5]) for k in (choice.ranked if choice else ())
                      if (key := (m, ds, sl, pl, conf, k)) in cluster_cells
                      and cluster_cells[key].mse is not None]
            parts.append(MISSING if not scored or choice.k is None else
                         f"{min(scored)[1]}" + (" (partial)" if choice.partial else ""))
        return " / ".join(parts)

    labels = " / ".join(dataset_label(d) for d in datasets)
    cfg_rows = [
        ["Channel clusters k swept"] + [
            ", ".join(str(v) for v in vs) if (vs := sorted(
                {key[5] for key in cluster_cells
                 if key[0] == m and at(key) and key[5] is not None})) else MISSING
            for m in models],
        [f"k chosen on validation, `{val_key}` ({labels})"]
        + [k_line(m, choices) for m in models],
    ]
    if batch is not None:
        cfg_rows.append([f"k chosen on validation under the batch protocol, `sl_val/loss` "
                         f"({labels})"] + [k_line(m, batch_choices) for m in models])
    cfg_rows += [
        [f"Oracle k, lowest `{test_key}/loss` (reported, never used; {labels})"]
        + [oracle_line(m) for m in models],
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
        pipelines += cluster_pipelines(by_title[CLUSTER_TITLE])
    if MULTIVARIATE_TITLE in by_title:
        pipelines.append(Pipeline("Multivariate", canonical_cells(by_title[MULTIVARIATE_TITLE])))
    return pipelines


def cluster_pipelines(cells: dict[tuple, Cell]) -> list[Pipeline]:
    """One Pipeline per k of a cluster tree's cells (raw keys), in k order."""
    cluster = canonical_cells(cells)
    return [Pipeline(f"Cluster (k={k})", cluster, k=k)
            for k in sorted({key[5] for key in cluster if key[5] is not None})]


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

    # The foreign GPU windows `make.py extract` reads too, so the timing tables
    # mark the same cells on this path (emfres/contention.py).
    from emfres.contention import load as load_contention
    contention = load_contention(Path(__file__).resolve().parent / "specs" / "contention.yaml")

    loaded: list[tuple[str, list[Run], dict[tuple, Cell]]] = []
    for logs_root in logs_roots:
        if not logs_root.is_dir():
            parser.error(f"--logs-root {logs_root} is not a directory")

        runs = load_runs(logs_root, args.exclude, contention=contention)
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
        args.output.write_text(render_setup(title, runs, cells))
        print(f"Wrote {args.output} ({len(runs)} runs, {len(cells)} aggregated cells).", file=sys.stderr)
        return

    # A complete cell has the seeds every tree's cells of its dataset and window
    # have: a sweep still training is judged against the finished trees.
    targets = seed_targets(*(cells for _, _, cells in loaded))
    updates = [(title, render_setup(title, runs, cells, targets=targets))
               for title, runs, cells in loaded]

    by_title = {title: cells for title, _, cells in loaded}
    mv_cells = by_title.get(MULTIVARIATE_TITLE)
    # The cluster tree is deliberately not a right-hand side here: its cells carry a
    # k the univariate trees have no counterpart for, and this section's framing
    # ("Δ is the univariate value") does not describe a cluster-vs-joint delta. Its
    # own '# Cluster (k-Grouped Channels)' section above carries the numbers.
    uv_roots = [(t, c) for t, c in by_title.items()
                if t not in (MULTIVARIATE_TITLE, CLUSTER_TITLE)]
    if mv_cells and uv_roots:
        comparison = render_comparison(COMPARISON_TITLE, MULTIVARIATE_TITLE, mv_cells, uv_roots,
                                       concurrency={t: concurrency_text(r)
                                                    for t, r, _ in loaded if per_band_note(r)},
                                       targets=targets)
        if comparison:
            updates.append((COMPARISON_TITLE, comparison))
        else:
            print("WARNING: no (dataset, L, O, model) cells shared between the multivariate and "
                  "univariate trees; skipping the comparison section", file=sys.stderr)

    # The cross tables lead with the cluster tree's fixed-protocol cells, its
    # logged batch ones their labelled sensitivity (protocol_cells).
    cl_cells = by_title.get(CLUSTER_TITLE)
    pipe_cells, batch = dict(by_title), None
    if cl_cells:
        cluster_runs = next((r for t, r, _ in loaded if t == CLUSTER_TITLE), [])
        pipe_cells[CLUSTER_TITLE], batch = protocol_cells(cluster_runs, cl_cells)
    if mv_cells and cl_cells:
        ccm_section = render_ccm_vs_base(CCM_TITLE, canonical_cells(pipe_cells[CLUSTER_TITLE]),
                                         canonical_cells(mv_cells), cluster_runs,
                                         targets=targets,
                                         batch=canonical_cells(batch) if batch else None)
        if ccm_section:
            updates.append((CCM_TITLE, ccm_section))

    pipelines = build_pipelines(pipe_cells)
    pc_runs = next((r for t, r, _ in loaded if t == PER_CHANNEL_TITLE), [])
    ultimate = (render_ultimate(ULTIMATE_TITLE, pipelines,
                                concurrency_text(pc_runs) if pc_runs else None,
                                targets=targets,
                                batch=cluster_pipelines(batch) if batch else None)
                if pipelines else "")
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
