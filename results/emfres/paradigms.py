"""Paradigm comparison: multivariate vs per-channel vs channel clustering.

The question the whole benchmark exists to answer is which way of treating
the channels forecasts best. The per-setup tables cannot answer it: they print
one mean per cell, so a 0.4% gap reads the same whether it is ten times the
seed noise or a tenth of it, and until the Phase 2 reruns the joint and
per-channel cells were single-seed besides. This section answers it the way
the numbers allow:

* **Seed means with their spread.** Every value is a mean over seeds, with the
  sample standard deviation beside it. A per-channel seed is first averaged
  over its channels, and a seed with any channel missing is dropped rather
  than averaged over whichever channels finished.
* **Matched seeds.** Two pipelines are compared on the seeds both have, so a
  third seed on one side cannot shift the comparison.
* **A tie band.** A gap no larger than the pooled seed standard deviation is a
  tie, not a win. Deterministic models (ARIMA, Persistence) have no seed
  spread, so an absolute floor catches their multivariate and per-channel
  cells, which are the same computation and agree to ~1e-7.
* **Best k chosen on validation.** "Cluster (best k)" is the cluster count
  with the lowest mean best-validation loss -- what early stopping selected on
  -- never the test metric. The older *CCM vs Base* section picks k on test and
  says so; this one is the version that could be reported as a result.
* **A test across cells.** Each comparison's wins, ties and losses are counted
  over (dataset, model) cells, with a two-sided Wilcoxon signed-rank p-value on
  the relative gaps. Numerical duplicates (|gap| under the floor) are dropped
  from the test: they are the same arithmetic run twice, not evidence.

Training cost is deliberately absent. Timings are only comparable between runs
that trained under the same contention (``concurrency`` in runs.csv), and the
per-channel grid trains eight bands at a time; a cost table belongs to a
dedicated sequential pass.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

from .frames import complete_seeds, family_of

METRICS = (("mae", "MAE"), ("mse", "MSE"))

# Below this, two values are the same computation, not a close race.
TIE_FLOOR = 5e-5

MV, PC, K1, BEST = "multivariate", "perchannel", "cluster k=1", "cluster best"
LABEL = {MV: "Multivariate", PC: "Per-channel", K1: "Cluster (k=1)",
         BEST: "Cluster (best k)"}

# (baseline, challenger, question). Deltas read challenger relative to baseline.
COMPARISONS = [
    (MV, PC, "Is forecasting each channel alone better than jointly?"),
    (MV, K1, "Does the cluster module alone (clustering off) change anything?"),
    (K1, BEST, "Does clustering itself help, against the same network without it?"),
    (MV, BEST, "Is the best clustered network better than the plain joint one?"),
]


def _f(v: str) -> Optional[float]:
    return float(v) if v not in ("", None) else None


def seed_values(runs_csv: Path, family: str, seq_len: int, pred_len: int) -> dict:
    """{(pipeline, dataset, model): {seed: {"mse", "mae", "val"}}} at one window.

    pipeline is "multivariate", "perchannel" or ("cluster", k). A per-channel
    seed is the mean over its channels, and is kept only if it has every
    channel any seed of the cell has (frames.complete_seeds, the rule
    cells.csv's per-channel seed sd uses too).
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
            # .get: a frames file written before val_loss existed still renders,
            # just without the validation-chosen best k.
            vals = {m: [_f(r.get(m)) for r in rs] for m in ("mse", "mae", "val_loss")}
            if any(v is None for v in vals["mse"] + vals["mae"]):
                continue
            val = [v for v in vals["val_loss"] if v is not None]
            cell[seed] = {"mse": statistics.fmean(vals["mse"]),
                          "mae": statistics.fmean(vals["mae"]),
                          "val": statistics.fmean(val) if val else None}
        if cell:
            out[key] = cell
    return out


def pipelines(values: dict) -> dict:
    """{(dataset, model): {MV|PC|K1|BEST: {seed: metrics}}} plus the chosen k.

    BEST is the cluster count with the lowest mean validation loss over its own
    seeds; the chosen k is kept alongside so the table can print it.
    """
    cells: dict = defaultdict(dict)
    chosen: dict = {}
    by_cell_k: dict = defaultdict(dict)
    for (pipeline, dataset, model), seeds in values.items():
        if isinstance(pipeline, tuple):
            by_cell_k[(dataset, model)][pipeline[1]] = seeds
        else:
            cells[(dataset, model)][pipeline] = seeds
    for cell, ks in by_cell_k.items():
        if 1 in ks:
            cells[cell][K1] = ks[1]
        scored = {k: statistics.fmean(s["val"] for s in seeds.values())
                  for k, seeds in ks.items()
                  if all(s["val"] is not None for s in seeds.values())}
        if scored:
            k = min(scored, key=lambda kk: (scored[kk], kk))
            cells[cell][BEST] = ks[k]
            chosen[cell] = k
    return cells, chosen


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


def _fmt(mean: float, sd: float, n: int) -> str:
    return f"{mean:.4f}" if n < 2 else f"{mean:.4f} ±{sd:.4f}"


def render(runs_csv: Path, family: str, setting: tuple,
           dataset_label: Callable[[str], str] = str,
           model_label: Callable[[str], str] = str,
           dataset_rank: Callable = lambda d: d,
           model_rank: Callable = lambda m: m) -> str:
    """The section's markdown, or "" when no two paradigms share a cell."""
    seq_len, pred_len = setting
    cells, chosen = pipelines(seed_values(runs_csv, family, seq_len, pred_len))
    if not any(len(p) > 1 for p in cells.values()):
        return ""
    order = sorted(cells, key=lambda c: (dataset_rank(c[0]), model_rank(c[1])))

    out = [
        "# Paradigm Comparison", "",
        f"- Configuration: L{seq_len}/O{pred_len}, family `{family}`",
        "- Values: mean over seeds, ± the sample standard deviation; a per-channel "
        "seed is first averaged over its channels",
        "- Pairing: every comparison uses only the seeds both sides have (n)",
        f"- Tie: a gap no larger than the pooled seed standard deviation "
        f"(floor {TIE_FLOOR:g}) is a tie, not a win",
        "- Cluster (best k): the k with the lowest mean validation loss -- chosen "
        "without the test split",
        "- Wilcoxon p: two-sided signed-rank test on the relative gaps across "
        "cells; numerical duplicates (ARIMA, Persistence) are left out of it",
        "",
        "## 1 · Scoreboard", "",
        "Challenger against baseline; lower is better.", "",
        "| Question | Metric | Challenger better | Tie | Baseline better | Cells | "
        "Median gap | Wilcoxon p |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for base, chal, question in COMPARISONS:
        for metric, name in METRICS:
            res = [compare(cells[c][base], cells[c][chal], metric)
                   for c in order if base in cells[c] and chal in cells[c]]
            res = [r for r in res if r]
            if not res:
                continue
            count = {o: sum(r["outcome"] == o for r in res)
                     for o in ("challenger", "tie", "baseline")}
            rels = [r["rel"] for r in res if not r["duplicate"]]
            p = wilcoxon_p(rels)
            med = statistics.median(rels) * 100 if rels else float("nan")
            out.append(
                f"| {LABEL[chal]} vs {LABEL[base]} — {question} | {name} "
                f"| {count['challenger']} | {count['tie']} | {count['baseline']} "
                f"| {len(res)} | {med:+.2f}% "
                f"| {'-' if p is None else f'{p:.3g}'} |")

    for metric, name in METRICS:
        out += ["", f"## {2 if metric == 'mae' else 3} · Per cell — {name}", "",
                "**Bold** = better side of a comparison outside the tie band; "
                "≈ = inside it. Δ is the challenger relative to its baseline.", "",
                "| Dataset | Model | Multivariate | Per-channel | Δ | Cluster (k=1) "
                "| Cluster (best k) | Δ vs k=1 | n |",
                "|:---|:---|---:|---:|---:|---:|---:|---:|:---|"]
        for c in order:
            p = cells[c]

            def value(pipe, other=None, base=None):
                if pipe not in p:
                    return "-"
                s = list(p[pipe].values())
                txt = _fmt(statistics.fmean(v[metric] for v in s),
                           statistics.stdev(v[metric] for v in s) if len(s) > 1 else 0.0,
                           len(s))
                if pipe == BEST and c in chosen:
                    txt += f" (k={chosen[c]})"
                return txt

            def delta(base, chal):
                if base not in p or chal not in p:
                    return "-"
                r = compare(p[base], p[chal], metric)
                if r is None:
                    return "-"
                sym = "≈ " if r["outcome"] == "tie" else ""
                txt = f"{sym}{r['rel'] * 100:+.1f}%"
                return f"**{txt}**" if r["outcome"] == "challenger" else txt

            ns = "/".join(str(len(p[x])) if x in p else "-" for x in (MV, PC, K1, BEST))
            out.append(
                f"| {dataset_label(c[0])} | {model_label(c[1])} | {value(MV)} "
                f"| {value(PC)} | {delta(MV, PC)} | {value(K1)} | {value(BEST)} "
                f"| {delta(K1, BEST)} | {ns} |")
    out += ["", "*n = seeds for multivariate / per-channel / cluster k=1 / cluster "
            "best k. A bold Δ means the challenger (per-channel, or the best k) "
            "is better beyond seed noise.*"]
    return "\n".join(out) + "\n"
