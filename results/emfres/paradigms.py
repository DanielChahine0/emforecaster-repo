"""Paradigm comparison: multivariate vs per-channel vs channel clustering.

The question the whole benchmark exists to answer is which way of treating
the channels forecasts best. The per-setup tables cannot answer it: they print
one mean per cell, so a 0.4% gap reads the same whether it is ten times the
seed noise or a tenth of it. This section answers it the way the numbers
allow:

* **Seed means with their spread.** Every value is a mean over seeds, with the
  sample standard deviation beside it. A per-channel seed is first averaged
  over its channels, and a seed with any channel missing is dropped rather
  than averaged over whichever channels finished (frames.complete_seeds).
* **Matched seeds.** Two pipelines are compared on the seeds both have, so a
  third seed on one side cannot shift the comparison.
* **A tie band.** A gap no larger than the pooled seed standard deviation is a
  tie, not a win (floor TIE_FLOOR).
* **The fixed cluster protocol.** A cluster run's primary numbers are the fixed
  train-mean membership ones (runs.csv ``mse_fixed``/``mae_fixed``/
  ``val_loss_fixed``): every channel keeps the cluster weights computed once
  from the training split. The logged numbers average the membership over each
  evaluation batch, so a k > 1 test forecast depends on its neighbouring
  windows; they are kept as the "batch protocol" sensitivity rows.
* **Best k chosen on validation, per cell.** "Cluster (best k)" is the k with
  the lowest seed-mean validation loss of the cell's sweep under the protocol
  in use -- never the test metric. Only k values that have every seed of the
  sweep are ranked, and under the fixed protocol only when the ranked runs'
  fixed metrics share one origin (runlogs.comparable_fixed): a CPU re-score
  and a training run's own numbers differ by device noise as large as the gap
  between two k's.
* **Identical by construction.** ARIMA and Persistence have no gradient path
  and fit one model per band, so the multivariate and per-channel grids run
  the same computation (and there is nothing to cluster). Their cells are
  shown once, n = 1, and left out of every count and test.
* **Complete cells only.** A comparison is counted when both sides have every
  seed of the family (and the best k comes from a finished sweep). A cell
  still waiting for runs is shown with its n and counted as pending, never
  averaged as if it were complete.
* **A test across cells.** Each comparison's wins, ties and losses are counted
  over (dataset, model) cells, with a two-sided Wilcoxon signed-rank p-value on
  the relative gaps. Gaps under the floor are zero differences and are
  dropped from the test, as its default zero_method does.

Training cost is deliberately absent. Timings are only comparable between runs
that trained under the same contention (``concurrency`` in runs.csv), and the
per-channel grid trains 8 to 12 bands at a time; timing.csv holds the solo
estimate.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .frames import complete_seeds, family_of
from .runlogs import comparable_fixed

METRICS = (("mae", "MAE"), ("mse", "MSE"))

# Below this, two values are the same computation, not a close race.
TIE_FLOOR = 5e-5

MV, PC, K1, BEST = "multivariate", "perchannel", "cluster k=1", "cluster best"
LABEL = {MV: "Multivariate", PC: "Per-channel", K1: "Cluster (k=1)",
         BEST: "Cluster (best k)"}
PIPES = (MV, PC, K1, BEST)

# (baseline, challenger, question). Deltas read challenger relative to baseline.
COMPARISONS = [
    (MV, PC, "Is forecasting each channel alone better than jointly?"),
    (MV, K1, "Does the cluster module alone (clustering off) change anything?"),
    (K1, BEST, "Does clustering itself help, against the same network without it?"),
    (MV, BEST, "Is the best clustered network better than the plain joint one?"),
]

# The cluster protocols (runlogs.FIXED_SOURCES): "fixed" is the primary one,
# "batch" the logged membership-averaged-per-batch one, a sensitivity row.
PROTOCOLS = ("fixed", "batch")

# No gradient path, one model per band: MV == PC == k = C by construction.
DETERMINISTIC = ("ARIMA", "Persistence")

def _f(v) -> Optional[float]:
    return float(v) if v not in ("", None) else None


def seed_values(runs_csv: Path, family: str, seq_len: int, pred_len: int) -> dict:
    """{(pipeline, dataset, model): {seed: metrics}} at one window.

    pipeline is "multivariate", "perchannel" or ("cluster", k). A per-channel
    seed is the mean over its channels, and is kept only if it has every
    channel any seed of the cell has (frames.complete_seeds, the rule
    cells.csv's per-channel seed sd uses too).

    metrics holds the logged ``mse``, ``mae`` and ``val`` (best validation
    loss), and for a cluster run the fixed-protocol
    ``mse_fixed``, ``mae_fixed``, ``val_fixed`` and their ``source``.
    """
    raw: dict = defaultdict(lambda: defaultdict(list))
    with open(runs_csv) as f:
        for r in csv.DictReader(f):
            if (r.get("family") or family_of(r["dataset"])) != family:
                continue
            if int(r["seq_len"]) != seq_len or int(r["pred_len"]) != pred_len:
                continue
            if not r["seed"] or r["conf"] != "1":
                continue
            setup = r["setup"]
            if setup == "cluster":
                pipeline = ("cluster", int(r["k"]))
            elif setup in (MV, PC):
                pipeline = setup
            else:
                continue
            raw[(pipeline, r["dataset"], r["model"])][r["seed"]].append(r)

    out = {}
    for key, by_seed in raw.items():
        complete = complete_seeds(by_seed) if key[0] == PC else set(by_seed)
        cell = {}
        for seed, rs in by_seed.items():
            if seed not in complete:
                continue
            # .get: a frames file written before a column existed still
            # renders, just without what that column feeds.
            vals = {m: [_f(r.get(m)) for r in rs] for m in ("mse", "mae", "val_loss")}
            if any(v is None for v in vals["mse"] + vals["mae"]):
                continue
            val = [v for v in vals["val_loss"] if v is not None]
            s = {"mse": statistics.fmean(vals["mse"]),
                 "mae": statistics.fmean(vals["mae"]),
                 "val": statistics.fmean(val) if val else None}
            if isinstance(key[0], tuple):
                r = rs[0]
                s.update(mse_fixed=_f(r.get("mse_fixed")), mae_fixed=_f(r.get("mae_fixed")),
                         val_fixed=_f(r.get("val_loss_fixed")),
                         source=r.get("fixed_source") or "")
            cell[seed] = s
        if cell:
            out[key] = cell
    return out


def view(seeds: dict, protocol: str) -> dict:
    """A cluster pipeline's seeds with mse/mae/val under ``protocol``.

    "batch" is the logged numbers as they are. "fixed" swaps in the fixed
    membership ones, and drops a seed that has none (fixed_source "none"): it
    is not scored under the primary protocol, so it cannot be averaged in.
    """
    if protocol == "batch":
        return dict(seeds)
    if protocol != "fixed":
        raise ValueError(f"unknown cluster protocol {protocol!r}")
    return {seed: {**s, "mse": s["mse_fixed"], "mae": s["mae_fixed"], "val": s["val_fixed"]}
            for seed, s in seeds.items()
            if s.get("mse_fixed") is not None and s.get("mae_fixed") is not None}


@dataclass
class Choice:
    """How a cell's best k was chosen, or why none was."""
    k: Optional[int]
    swept: tuple = ()      # the k values with a scored seed
    ranked: tuple = ()     # those ranked: every seed of the sweep, a val loss each
    grid: tuple = ()       # the family's k values at this window
    seeds: tuple = ()      # the sweep's seeds (the union over its k)
    n_family: int = 0      # the family's seed count at this window
    sources: tuple = ()    # fixed_source of the ranked runs (fixed protocol)
    reason: str = ""       # why no k was chosen

    @property
    def partial(self) -> bool:
        """Ranked over fewer k values, or fewer seeds, than the family has."""
        return set(self.grid) != set(self.ranked) or len(self.seeds) < self.n_family

    def tag(self) -> str:
        """What the per-cell table prints after the best-k value."""
        if self.k is None:
            return f"no k: {self.reason}"
        parts = [f"k={self.k}"]
        if self.sources:
            parts.append("/".join(self.sources))
        txt = ", ".join(parts)
        if self.partial:
            txt += (f"; ranked {','.join(map(str, self.ranked))} of "
                    f"{','.join(map(str, self.grid))}, {len(self.seeds)}/{self.n_family} seeds")
        return txt


def choose_k(ks: dict, protocol: str, grid: tuple, n_family: int) -> Choice:
    """The validation-chosen k of one sweep ({k: {seed: metrics}} under ``protocol``).

    Per cell, as before the fixed protocol existed: the k with the lowest mean
    validation loss over the sweep's seeds, the smaller k on a tie. A k that
    lacks one of the sweep's seeds (still training) is not ranked -- its mean
    is over other seeds. Under the fixed protocol the ranked runs' fixed
    metrics must share one origin (runlogs.comparable_fixed), or no k is
    chosen.
    """
    swept = tuple(sorted(k for k, s in ks.items() if s))
    seeds = tuple(sorted(set().union(*(set(ks[k]) for k in swept)))) if swept else ()
    ranked = tuple(k for k in swept if set(ks[k]) == set(seeds)
                   and all(v.get("val") is not None for v in ks[k].values()))
    base = dict(swept=swept, ranked=ranked, grid=grid, seeds=seeds, n_family=n_family)
    if not ranked:
        return Choice(None, reason="no k has a validation loss at every seed", **base)
    sources = ()
    if protocol == "fixed":
        runs = [v.get("source", "") for k in ranked for v in ks[k].values()]
        sources = tuple(sorted(set(runs)))
        if not comparable_fixed(runs):
            return Choice(None, sources=sources,
                          reason=f"fixed metrics of two origins ({'+'.join(sources)})",
                          **base)
    scored = {k: statistics.fmean(v["val"] for v in ks[k].values()) for k in ranked}
    k = min(scored, key=lambda kk: (scored[kk], kk))
    return Choice(k, sources=sources, **base)


@dataclass
class Board:
    """Every cell's pipelines under one cluster protocol."""
    cells: dict                       # {(dataset, model): {pipe: {seed: metrics}}}
    choices: dict                     # {(dataset, model): Choice}
    seeds: tuple                      # the family's seeds at this window
    grid: tuple                       # the family's k values at this window
    protocol: str
    clusterable: frozenset = field(default_factory=frozenset)  # models with cluster runs

    def complete(self, cell, pipe) -> bool:
        """Every seed of the family, and for the best k a finished sweep."""
        p = self.cells.get(cell, {})
        if pipe not in p or set(p[pipe]) != set(self.seeds):
            return False
        return pipe != BEST or not self.choices[cell].partial

    def applies(self, cell, pipe) -> bool:
        """Whether the pipeline exists for this model (ran, or should run)."""
        if cell[1] in DETERMINISTIC:
            return pipe in (MV, PC)
        return pipe in (MV, PC) or cell[1] in self.clusterable


def build(values: dict, protocol: str = "fixed") -> Board:
    """The Board of seed_values' output under ``protocol``."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown cluster protocol {protocol!r}")
    family_seeds = set().union(*(set(s) for s in values.values())) if values else set()
    grid = tuple(sorted({key[0][1] for key in values if isinstance(key[0], tuple)}))

    cells: dict = defaultdict(dict)
    by_cell_k: dict = defaultdict(dict)
    for (pipeline, dataset, model), by_seed in values.items():
        kept = {s: m for s, m in by_seed.items() if s in family_seeds}
        if isinstance(pipeline, tuple):
            kept = view(kept, protocol)
            if kept:
                by_cell_k[(dataset, model)][pipeline[1]] = kept
        elif kept:
            cells[(dataset, model)][pipeline] = kept
    choices = {}
    for cell, ks in by_cell_k.items():
        if 1 in ks:
            cells[cell][K1] = ks[1]
        choice = choose_k(ks, protocol, grid, len(family_seeds))
        choices[cell] = choice
        if choice.k is not None:
            cells[cell][BEST] = ks[choice.k]
    clusterable = frozenset(key[2] for key in values if isinstance(key[0], tuple))
    return Board(dict(cells), choices, tuple(sorted(family_seeds)), grid, protocol,
                 clusterable)


def pipelines(values: dict, protocol: str = "fixed") -> tuple:
    """({(dataset, model): {MV|PC|K1|BEST: {seed: metrics}}}, {cell: chosen k}).

    The cluster pipelines are under ``protocol`` (the fixed one unless asked),
    and the chosen k is the validation-chosen one of choose_k.
    """
    board = build(values, protocol)
    return board.cells, {c: ch.k for c, ch in board.choices.items() if ch.k is not None}


def compare(a: dict, b: dict, metric: str) -> Optional[dict]:
    """Challenger b against baseline a on the seeds both have."""
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    xa = [a[s][metric] for s in seeds]
    xb = [b[s][metric] for s in seeds]
    ma, mb = statistics.fmean(xa), statistics.fmean(xb)
    sa = statistics.stdev(xa) if len(xa) > 1 else 0.0
    sb = statistics.stdev(xb) if len(xb) > 1 else 0.0
    band = max(TIE_FLOOR, math.sqrt((sa * sa + sb * sb) / 2))
    delta = mb - ma
    outcome = "tie" if abs(delta) <= band else ("challenger" if delta < 0 else "baseline")
    return {"n": len(seeds), "a": ma, "b": mb, "sa": sa, "sb": sb, "delta": delta,
            "rel": delta / ma if ma else float("nan"), "band": band,
            "outcome": outcome, "duplicate": abs(delta) < TIE_FLOOR}


def wilcoxon_p(rels: list[float]) -> Optional[float]:
    rels = [r for r in rels if not math.isnan(r)]
    if len(rels) < 2 or all(r == 0 for r in rels):
        return None
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(rels).pvalue)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #

def tally(board: Board, base: str, chal: str, metric: str, order: list, *,
          skip: frozenset = frozenset()) -> dict:
    """One scoreboard row: outcomes over the cells both sides complete.

    A deterministic model's cell counts as identical, one whose sides are
    not both complete (missing seeds, a partial sweep, no k chosen) as
    pending; neither enters the outcomes or the test.
    """
    res, pending, identical = [], 0, 0
    for c in order:
        if c in skip or not (board.applies(c, base) and board.applies(c, chal)):
            continue
        p = board.cells.get(c, {})
        if c[1] in DETERMINISTIC:
            identical += base in p and chal in p
            continue
        if not (board.complete(c, base) and board.complete(c, chal)):
            pending += 1
            continue
        r = compare(p[base], p[chal], metric)
        if r:
            res.append(r)
    count = {o: sum(r["outcome"] == o for r in res) for o in ("challenger", "tie", "baseline")}
    gaps = [r["rel"] for r in res if not r["duplicate"]]
    return {"count": count, "cells": len(res), "pending": pending, "identical": identical,
            "n_test": len(gaps), "median": statistics.median(gaps) if gaps else float("nan"),
            "p": wilcoxon_p(gaps)}


def _fmt(mean: float, sd: float, n: int) -> str:
    return f"{mean:.4f}" if n < 2 else f"{mean:.4f} ±{sd:.4f}"


def _p(p: Optional[float]) -> str:
    return "-" if p is None else f"{p:.3g}"


def render(runs_csv: Path, family: str, setting: tuple,
           dataset_label: Callable[[str], str] = str,
           model_label: Callable[[str], str] = str,
           dataset_rank: Callable = lambda d: d,
           model_rank: Callable = lambda m: m,
           title: str = "Paradigm Comparison") -> str:
    """The section's markdown, or "" when no two paradigms share a cell."""
    seq_len, pred_len = setting
    values = seed_values(runs_csv, family, seq_len, pred_len)
    board = build(values, "fixed")
    batch = build(values, "batch")
    cells = board.cells
    if not any(len(p) > 1 for p in cells.values()):
        return ""
    order = sorted(cells, key=lambda c: (dataset_rank(c[0]), model_rank(c[1])))
    n_family = len(board.seeds)

    out = [f"# {title}", "",
           f"- Configuration: L{seq_len}/O{pred_len}, family `{family}`; "
           f"seeds {', '.join(board.seeds)} (N = {n_family})",
           "- Values: mean over seeds, ± the sample standard deviation; a per-channel "
           "seed is first averaged over its channels, and kept only if it has all of them",
           "- Pairing: every comparison uses only the seeds both sides have (n)",
           f"- Tie: a gap no larger than the pooled seed standard deviation "
           f"(floor {TIE_FLOOR:g}) is a tie, not a win",
           "- Cluster protocol: the fixed train-mean membership (`sl_test_fixed/*`, "
           "primary): every channel keeps the cluster weights computed once from the "
           "training split. The logged batch protocol, which averages the membership "
           "over each evaluation batch so that a k > 1 forecast depends on its "
           "neighbouring windows, is a sensitivity row. At k = 1 the two are the same "
           "forecast",
           "- Cluster (best k): per cell, the k with the lowest seed-mean validation "
           "loss under the fixed protocol (`sl_val_fixed/loss`) -- chosen without the "
           "test split. Only k values with every seed of the sweep are ranked, and only "
           "runs whose fixed metrics share one origin: the training run's own (`log`) "
           "or a CPU re-score (`sidecar`), named after the k",
           f"- Identical by construction: {' and '.join(DETERMINISTIC)} have no gradient "
           "path and fit one model per band, so the multivariate and per-channel grids "
           "run the same computation and there is nothing to cluster (n/a). Their cells "
           "are shown once (n = 1) and left out of every count and test",
           f"- Counted: a cell enters a comparison when both sides have all {n_family} "
           "seeds (and the best k comes from a finished sweep); a cell still missing "
           "runs is shown with its n and counted as pending",
           f"- Wilcoxon p: two-sided signed-rank test on the relative gaps of the counted "
           f"cells; a gap under {TIE_FLOOR:g} is a zero difference and is dropped from "
           f"the test (its default), so n (test) can be smaller than the cells counted",
           ]
    out.append("")

    # ------------------------------------------------------------ scoreboard
    out += ["## 1 · Scoreboard", "",
            "Challenger against baseline; lower is better. Cells: counted, with every "
            "seed on both sides; Pending: not yet complete; Identical: deterministic "
            "models, identical by construction.", "",
            "| Question | Metric | Challenger better | Tie | Baseline better | Cells | "
            "Pending | Identical | n (test) | Median gap | Wilcoxon p |",
            "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for base, chal, question in COMPARISONS:
        for metric, name in METRICS:
            t = tally(board, base, chal, metric, order)
            if not (t["cells"] or t["pending"] or t["identical"]):
                continue
            c = t["count"]
            med = "-" if math.isnan(t["median"]) else f"{t['median'] * 100:+.2f}%"
            out.append(
                f"| {LABEL[chal]} vs {LABEL[base]} — {question} | {name} "
                f"| {c['challenger']} | {c['tie']} | {c['baseline']} | {t['cells']} "
                f"| {t['pending']} | {t['identical']} | {t['n_test']} | {med} "
                f"| {_p(t['p'])} |")

    # ----------------------------------------------------------- sensitivity
    rows = []

    def add(base, chal, label, b, metric):
        t = tally(b, base, chal, metric, order)
        if not t["cells"] and not t["pending"]:
            return
        c = t["count"]
        med = "-" if math.isnan(t["median"]) else f"{t['median'] * 100:+.2f}%"
        rows.append(f"| {LABEL[chal]} vs {LABEL[base]} | {label} | "
                    f"{dict(METRICS)[metric]} | {c['challenger']} | {c['tie']} | "
                    f"{c['baseline']} | {t['cells']} | {t['pending']} | {t['n_test']} "
                    f"| {med} | {_p(t['p'])} |")

    for base, chal, _ in COMPARISONS:
        if K1 in (base, chal) or BEST in (base, chal):
            for metric, _name in METRICS:
                add(base, chal, "batch protocol (k chosen on the logged `sl_val/loss`)",
                    batch, metric)
    if rows:
        out += ["", "## 2 · Sensitivity", "",
                "The scoreboard's comparisons under other choices, counted the same way.",
                "",
                "| Comparison | Variant | Metric | Challenger better | Tie | "
                "Baseline better | Cells | Pending | n (test) | Median gap | Wilcoxon p |",
                "|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        out += rows

    # -------------------------------------------------------------- per cell
    section = 3 if rows else 2
    for metric, name in METRICS:
        out += ["", f"## {section} · Per cell — {name}", "",
                "**Bold** = better side of a comparison outside the tie band; ≈ = inside "
                "it; no mark = not both sides complete, n in brackets. Δ is the "
                "challenger relative to its baseline.", "",
                f"| Dataset | Model | Multivariate | Per-channel | Δ | Cluster (k=1) "
                f"| Cluster (best k) | Δ vs k=1 | n (of {n_family}) |",
                "|:---|:---|---:|---:|---:|---:|---:|---:|:---|"]
        section += 1
        for c in order:
            out.append(_cell_row(board, c, metric, dataset_label, model_label))

    # ----------------------------------------------- cluster protocol per cell
    clustered = [c for c in order if c in board.choices or c in batch.choices]
    if clustered:
        out += ["", f"## {section} · Cluster protocol per cell", "",
                "The best k and its gap to k = 1 under each protocol, each choosing k on "
                "its own validation loss. Marks as above.", "",
                "| Dataset | Model | Fixed: best k | Δ MAE | Δ MSE | Batch: best k | "
                "Δ MAE | Δ MSE |",
                "|:---|:---|:---|---:|---:|:---|---:|---:|"]
        section += 1
        for c in clustered:
            row = [dataset_label(c[0]), model_label(c[1])]
            for b in (board, batch):
                choice = b.choices.get(c)
                row.append(choice.tag() if choice else "-")
                for metric, _ in METRICS:
                    row.append(_delta(b, c, K1, BEST, metric))
            out.append("| " + " | ".join(row) + " |")

    out += ["", f"*n = seeds for multivariate / per-channel / cluster k=1 / cluster best "
                f"k, of the family's {n_family}. A bold Δ means the challenger "
                "(per-channel, or the best k) is better beyond seed noise.*"]
    return "\n".join(out) + "\n"


def _delta(board: Board, cell: tuple, base: str, chal: str, metric: str) -> str:
    """Δ of chal against base, marked only when both sides are complete."""
    p = board.cells.get(cell, {})
    if base not in p or chal not in p:
        return "-"
    r = compare(p[base], p[chal], metric)
    if r is None:
        return "-"
    txt = f"{r['rel'] * 100:+.1f}%"
    if not (board.complete(cell, base) and board.complete(cell, chal)):
        return f"{txt} (n={r['n']})"
    if r["outcome"] == "tie":
        return f"≈ {txt}"
    return f"**{txt}**" if r["outcome"] == "challenger" else txt


def _cell_row(board: Board, c: tuple, metric: str, dataset_label, model_label) -> str:
    p = board.cells.get(c, {})
    det = c[1] in DETERMINISTIC

    def value(pipe):
        if det and pipe in (K1, BEST):
            return "n/a (per-band)"
        if pipe == BEST and c in board.choices and board.choices[c].k is None:
            return f"- ({board.choices[c].tag()})"
        if pipe not in p:
            return "-"
        s = list(p[pipe].values())
        n = 1 if det else len(s)
        txt = _fmt(statistics.fmean(v[metric] for v in s),
                   statistics.stdev(v[metric] for v in s) if len(s) > 1 else 0.0, n)
        if pipe == BEST and c in board.choices:
            txt += f" ({board.choices[c].tag()})"
        return txt

    if det:
        pc_delta = "identical" if MV in p and PC in p else "-"
        ns = "/".join("1" if pipe in p else "-" for pipe in (MV, PC)) + "/n/a/n/a"
    else:
        pc_delta = _delta(board, c, MV, PC, metric)
        ns = "/".join(str(len(p[x])) if x in p else "-" for x in PIPES)
    k1_delta = "n/a" if det else _delta(board, c, K1, BEST, metric)
    return (f"| {dataset_label(c[0])} | {model_label(c[1])} | {value(MV)} | {value(PC)} "
            f"| {pc_delta} | {value(K1)} | {value(BEST)} | {k1_delta} | {ns} |")
