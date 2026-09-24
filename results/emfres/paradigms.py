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
  over (dataset, model) cells, with the median relative gap of every counted
  cell and a two-sided Wilcoxon signed-rank p-value on those gaps. A cell
  whose two sides differ by less than TIE_FLOOR -- an absolute difference in
  the metric itself, not a relative gap -- is a zero difference: it stays in
  the count and the median but is dropped from the test, as the test's
  default zero_method drops zeros.
* **Whether the Transformer trains.** Seeds 69422 and 69423 of the EMF
  Transformer are dead at initialisation and forecast the lookback mean, and
  the VoIP Transformer never beats it; TRAINS_FRACTION holds the rule. The
  table says how many seeds train, adds a descriptive row for the seeds the
  multivariate run trains at, and flags a cell where no seed of any paradigm
  trains -- the scoreboard is then also shown without it.
* **Sensitivity rows** for what a pooled MSE hides (audit PE-1, PE-2): the
  ratio-of-sums skill, the two named near-unforecastable bands dropped, each
  cell's top band dropped (the more telling of the two: it is where the MSE
  test's p moves), and the failed per-channel bands dropped, every band
  variant from both sides.

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

# Below this absolute difference in the metric, two values are the same
# computation, not a close race: the floor of the tie band, and the zero
# difference the Wilcoxon test leaves out (tally).
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

# The models each family's cluster grid trains, by model id: forecasting/
# grids.py CLUSTER_MODELS, the list both GRIDS["cluster"] and
# GRIDS["voip_cluster"] use (the results package does not import the training
# one; test_paradigms checks the two agree). A planned model whose sweep has
# not started is pending in every cluster comparison, not absent from it --
# on VoIP, EMForecaster, LSTM and the Transformer until the sweep reaches them.
CLUSTERED = {
    family: ("DLinear", "PatchTST", "TSMixer", "RF_EMF_Transformer", "EMForecaster", "LSTM")
    for family in ("emf", "voip")
}

# The models whose seeds are checked for training. A seed trains when its
# skill over the lookback-mean forecast (1 - MSE / lookback-mean MSE, ratio
# of sums over the bands) is at least TRAINS_FRACTION of the reference: the
# best multivariate seed-mean skill any other gradient-trained model reaches
# on the same dataset and window. Relative, because the datasets differ in how
# forecastable they are (the best skill is 0.16 on EMF2023, 0.52 on ED2022).
# On the 2026-09-24 logs the Transformer's trained seeds reach 0.65 to 0.97 of
# the reference (the top is cluster k=1 on ED2022) and every other seed 0.31 at
# most (per-channel VoIP Mobile L48/O12 seed 69424; no other above 0.17, and a
# collapsed multivariate seed within 0.002 of zero), under either cluster
# protocol. Only the Transformer is judged: skill is not a collapse
# detector for models without RevIN (VoIP DLinear is below the lookback mean
# while learning, baselines.py), so the rule is not applied to them.
TRAINS_MODELS = ("RF_EMF_Transformer", "Transformer")
TRAINS_FRACTION = 0.5

# The bands no model forecasts better than their own lookback mean and that
# carry a third of their cell's MSE (audit PE-1; frames/bands.csv, 2026-09-24):
# UH2022 ch21 (3568.0 MHz, 33% of the multivariate MSE at skill -0.01) and
# EMF2023 ch24 (3547.5 MHz, 36% at skill 0.00). Keyed by dataset, channel
# indices in the loader's order.
NEAR_UNFORECASTABLE = {"uh2022": (21,), "emf_2023_university_5_with_ptv": (24,)}

# A per-channel band run whose best validation epoch is at most this failed to
# train: its validation loss rose from the start and patience ran out inside
# the OneCycle warm-up (audit PE-2: nine PatchTST bands on ED2022 and UH2022;
# no multivariate or cluster PatchTST run does this).
FAILED_BEST_EPOCH = 2


def _f(v) -> Optional[float]:
    return float(v) if v not in ("", None) else None


def _i(v) -> Optional[int]:
    return int(float(v)) if v not in ("", None) else None


def _mean_all(vals: list) -> Optional[float]:
    return statistics.fmean(vals) if vals and all(v is not None for v in vals) else None


def seed_values(runs_csv: Path, family: str, seq_len: int, pred_len: int) -> dict:
    """{(pipeline, dataset, model): {seed: metrics}} at one window.

    pipeline is "multivariate", "perchannel" or ("cluster", k). A per-channel
    seed is the mean over its channels, and is kept only if it has every
    channel any seed of the cell has (frames.complete_seeds, the rule
    cells.csv's per-channel seed sd uses too).

    metrics holds the logged ``mse``, ``mae`` and ``val`` (best validation
    loss), ``lookback`` (the lookback-mean MSE on the same windows and bands),
    for a per-channel seed the ``failed`` channels (best epoch <=
    FAILED_BEST_EPOCH), and for a cluster run the fixed-protocol
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
                 "val": statistics.fmean(val) if val else None,
                 "lookback": _mean_all([_f(r.get("lookback_mse")) for r in rs]),
                 "failed": tuple(sorted(
                     int(r["channel"]) for r in rs
                     if r.get("channel", "") != "" and _i(r.get("best_epoch")) is not None
                     and _i(r.get("best_epoch")) <= FAILED_BEST_EPOCH))}
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
        return Choice(None, reason="no k has every seed of the sweep yet", **base)
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
    # Models with cluster runs, or planned ones (CLUSTERED) still to come.
    clusterable: frozenset = field(default_factory=frozenset)
    sweeps: dict = field(default_factory=dict)  # {(dataset, model): {k: {seed: metrics}}}

    def complete(self, cell, pipe) -> bool:
        """Every seed of the family, and for the best k a finished sweep."""
        p = self.cells.get(cell, {})
        if pipe not in p or set(p[pipe]) != set(self.seeds):
            return False
        return pipe != BEST or not self.choices[cell].partial

    def applies(self, cell, pipe) -> bool:
        """Whether the pipeline exists for this model (ran, or is planned to)."""
        if cell[1] in DETERMINISTIC:
            return pipe in (MV, PC)
        return pipe in (MV, PC) or cell[1] in self.clusterable


def build(values: dict, protocol: str = "fixed", seeds: Optional[set] = None,
          planned: tuple = (), choices: Optional[dict] = None) -> Board:
    """The Board of seed_values' output under ``protocol``.

    ``seeds`` keeps only those seeds (the descriptive trained-seed row); the
    family's seed set is then that set. ``planned`` names the models the
    family's cluster grid trains (CLUSTERED[family]): one with no cluster run
    yet still counts as pending there. ``choices`` reuses another board's
    chosen k per cell instead of choosing again -- the descriptive row takes
    the pooled cell's k, so no table names two best k's for one cell.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown cluster protocol {protocol!r}")
    family_seeds = set().union(*(set(s) for s in values.values())) if values else set()
    if seeds is not None:
        family_seeds &= set(seeds)
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
    chosen = {}
    for cell, ks in by_cell_k.items():
        if 1 in ks:
            cells[cell][K1] = ks[1]
        if choices is None:
            choice = choose_k(ks, protocol, grid, len(family_seeds))
        else:
            choice = choices.get(cell) or Choice(None, reason="no k for the pooled cell")
        chosen[cell] = choice
        if choice.k is not None and choice.k in ks:
            cells[cell][BEST] = ks[choice.k]
    clusterable = (frozenset(key[2] for key in values if isinstance(key[0], tuple))
                   | frozenset(planned))
    return Board(dict(cells), chosen, tuple(sorted(family_seeds)), grid, protocol,
                 clusterable, dict(by_cell_k))


def pipelines(values: dict, protocol: str = "fixed") -> tuple:
    """({(dataset, model): {MV|PC|K1|BEST: {seed: metrics}}}, {cell: chosen k}).

    The cluster pipelines are under ``protocol`` (the fixed one unless asked),
    and the chosen k is the validation-chosen one of choose_k.
    """
    board = build(values, protocol)
    return board.cells, {c: ch.k for c, ch in board.choices.items() if ch.k is not None}


def compare(a: dict, b: dict, metric: str, gap: str = "rel") -> Optional[dict]:
    """Challenger b against baseline a on the seeds both have.

    ``gap`` is what the across-cell test ranks: "rel", the challenger's gap
    relative to the baseline, or "abs", the plain difference (for a metric
    already on a common scale, like the skill ratio).
    """
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
    rel = delta / ma if ma else float("nan")
    return {"n": len(seeds), "a": ma, "b": mb, "sa": sa, "sb": sb, "delta": delta,
            "rel": rel, "gap": rel if gap == "rel" else delta, "band": band,
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
# Whether a seed trains (TRAINS_MODELS only)
# --------------------------------------------------------------------------- #

def skill_of(s: dict) -> Optional[float]:
    """1 - MSE / lookback-mean MSE of one seed (a ratio of sums over bands)."""
    lb = s.get("lookback")
    return 1.0 - s["mse"] / lb if lb and s.get("mse") is not None else None


def trains_reference(board: Board, dataset: str) -> Optional[float]:
    """The best multivariate seed-mean skill of the other gradient-trained models."""
    best = None
    for (ds, model), p in board.cells.items():
        if ds != dataset or model in TRAINS_MODELS or model in DETERMINISTIC or MV not in p:
            continue
        sk = [skill_of(s) for s in p[MV].values()]
        if sk and all(v is not None for v in sk):
            v = statistics.fmean(sk)
            best = v if best is None else max(best, v)
    return best


def trained_seeds(board: Board, cell: tuple, pipe: str) -> Optional[set]:
    """The seeds of a pipeline that train, or None where the rule does not apply."""
    if cell[1] not in TRAINS_MODELS or pipe not in board.cells.get(cell, {}):
        return None
    ref = trains_reference(board, cell[0])
    if ref is None or ref <= 0:
        return None
    return {seed for seed, s in board.cells[cell][pipe].items()
            if (sk := skill_of(s)) is not None and sk >= TRAINS_FRACTION * ref}


def does_not_train(board: Board, cell: tuple) -> bool:
    """A TRAINS_MODELS cell where no seed of any of its pipelines trains."""
    judged = [trained_seeds(board, cell, pipe) for pipe in PIPES
              if pipe in board.cells.get(cell, {})]
    return bool(judged) and all(t is not None and not t for t in judged)


# --------------------------------------------------------------------------- #
# Band-level variants (frames/bands.csv)
# --------------------------------------------------------------------------- #

def band_values(bands_csv: Path, family: str, seq_len: int, pred_len: int) -> dict:
    """{(pipeline, dataset, model): {seed: {channel: (mse, mse_fixed)}}} and band names.

    pipeline as in seed_values. Returns (values, {(dataset, channel): band}).
    """
    out: dict = defaultdict(lambda: defaultdict(dict))
    names: dict = {}
    with open(bands_csv) as f:
        for r in csv.DictReader(f):
            if (r.get("family") or family_of(r["dataset"])) != family:
                continue
            if int(r["seq_len"]) != seq_len or int(r["pred_len"]) != pred_len:
                continue
            if not r["seed"] or r["conf"] != "1":
                continue
            setup = r["setup"]
            pipeline = ("cluster", int(r["k"])) if setup == "cluster" else setup
            if setup not in (MV, PC, "cluster"):
                continue
            ch = int(r["channel"])
            out[(pipeline, r["dataset"], r["model"])][r["seed"]][ch] = (
                _f(r["mse"]), _f(r.get("mse_fixed")))
            names[(r["dataset"], ch)] = r.get("band", "")
    return out, names


def _source_key(board: Board, cell: tuple, pipe: str):
    if pipe in (MV, PC):
        return pipe
    if pipe == K1:
        return ("cluster", 1)
    choice = board.choices.get(cell)
    return ("cluster", choice.k) if choice and choice.k is not None else None


def dropped(board: Board, cell: tuple, pipe: str, bands: dict,
            drop: tuple) -> Optional[dict]:
    """A pipeline's seeds with ``mse`` re-averaged over the bands not in ``drop``.

    From bands.csv, the per-band vectors (the fixed ones for a cluster pipeline
    under the fixed protocol), so a cell's "all bands" and "without" values
    come from the same numbers. A seed without a full band vector is left out.
    """
    seeds = board.cells.get(cell, {}).get(pipe)
    key = _source_key(board, cell, pipe)
    if seeds is None or key is None:
        return None
    per_seed = bands.get((key, cell[0], cell[1]), {})
    fixed = pipe in (K1, BEST) and board.protocol == "fixed"
    out = {}
    for seed, s in seeds.items():
        per = per_seed.get(seed)
        if not per:
            continue
        vals = [v[1] if fixed else v[0] for ch, v in per.items() if ch not in drop]
        if vals and all(v is not None for v in vals):
            out[seed] = {**s, "mse": statistics.fmean(vals)}
    return out


def top_band(board: Board, cell: tuple, bands: dict) -> tuple:
    """(channel,) of the cell's band with the largest seed-mean multivariate MSE.

    Over the multivariate seeds the board keeps, from their band vectors; ()
    when there are none. The drop-top-band variant takes it out of every
    pipeline of the cell (audit PE-1: on the 2026-09-24 logs this moves the
    EMF per-channel-vs-multivariate MSE Wilcoxon p from 0.018 to 0.090, where
    dropping the two named near-unforecastable bands moves it to 0.024).
    """
    seeds = board.cells.get(cell, {}).get(MV, {})
    per_seed = bands.get((MV, cell[0], cell[1]), {})
    by_band: dict = defaultdict(list)
    for seed in seeds:
        for ch, (mse, _) in per_seed.get(seed, {}).items():
            by_band[ch].append(mse)
    means = {ch: statistics.fmean(v) for ch, v in by_band.items()
             if v and all(x is not None for x in v)}
    return (max(means, key=lambda ch: (means[ch], -ch)),) if means else ()


def failed_bands(board: Board, cell: tuple) -> tuple:
    """The per-channel bands of a cell that failed at any seed (not TRAINS_MODELS)."""
    if cell[1] in TRAINS_MODELS:
        return ()
    seeds = board.cells.get(cell, {}).get(PC, {})
    return tuple(sorted({ch for s in seeds.values() for ch in s.get("failed", ())}))


def variant(board: Board, fn: Callable) -> Board:
    """A copy of ``board`` whose pipelines are fn(cell, pipe, seeds) (None drops)."""
    cells = {}
    for cell, p in board.cells.items():
        new = {}
        for pipe, seeds in p.items():
            v = fn(cell, pipe, seeds)
            if v:
                new[pipe] = v
        cells[cell] = new
    return Board(cells, board.choices, board.seeds, board.grid, board.protocol,
                 board.clusterable, board.sweeps)


def skill_variant(board: Board) -> Board:
    """``mse`` replaced by MSE / lookback-mean MSE, i.e. 1 - skill (ratio of sums)."""
    return variant(board, lambda cell, pipe, seeds: {
        seed: {**s, "mse": s["mse"] / s["lookback"]}
        for seed, s in seeds.items() if s.get("lookback")})


def drop_variant(board: Board, bands: dict, drops: Callable,
                 recompute: bool = False) -> Board:
    """Every cell's ``mse`` without the bands drops(cell) names.

    A cell with nothing to drop keeps its run-level numbers, unless
    ``recompute``: then it too is re-averaged from its band vectors, so it
    can stand beside a dropped variant built from the same numbers.
    """
    def fn(cell, pipe, seeds):
        d = drops(cell)
        return dropped(board, cell, pipe, bands, d) if d or recompute else seeds
    return variant(board, fn)


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #

def tally(board: Board, base: str, chal: str, metric: str, order: list, *,
          skip: frozenset = frozenset(), gap: str = "rel") -> dict:
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
        r = compare(p[base], p[chal], metric, gap)
        if r:
            res.append(r)
    count = {o: sum(r["outcome"] == o for r in res) for o in ("challenger", "tie", "baseline")}
    # The median describes every counted cell. Only the test leaves out a
    # zero difference (compare's "duplicate"), as its default zero_method does.
    gaps = [r["gap"] for r in res]
    tested = [r["gap"] for r in res if not r["duplicate"]]
    return {"count": count, "cells": len(res), "pending": pending, "identical": identical,
            "n_test": len(tested), "median": statistics.median(gaps) if gaps else float("nan"),
            "p": wilcoxon_p(tested)}


def _fmt(mean: float, sd: float, n: int) -> str:
    return f"{mean:.4f}" if n < 2 else f"{mean:.4f} ±{sd:.4f}"


def _p(p: Optional[float]) -> str:
    return "-" if p is None else f"{p:.3g}"


def _band_label(names: dict, dataset: str, ch: int) -> str:
    band = names.get((dataset, ch), "")
    try:
        float(band)
        return f"ch{ch} ({band} MHz)"
    except ValueError:
        return f"ch{ch} ({band})" if band else f"ch{ch}"


def render(runs_csv: Path, family: str, setting: tuple,
           dataset_label: Callable[[str], str] = str,
           model_label: Callable[[str], str] = str,
           dataset_rank: Callable = lambda d: d,
           model_rank: Callable = lambda m: m,
           bands_csv: Optional[Path] = None,
           title: str = "Paradigm Comparison") -> str:
    """The section's markdown, or "" when no two paradigms share a cell.

    ``bands_csv`` (default: bands.csv beside runs_csv) feeds the band-level
    sensitivity rows; without it they are left out and the note says so.
    """
    seq_len, pred_len = setting
    values = seed_values(runs_csv, family, seq_len, pred_len)
    planned = CLUSTERED.get(family, ())
    board = build(values, "fixed", planned=planned)
    batch = build(values, "batch", planned=planned)
    cells = board.cells
    if not any(len(p) > 1 for p in cells.values()):
        return ""
    order = sorted(cells, key=lambda c: (dataset_rank(c[0]), model_rank(c[1])))
    n_family = len(board.seeds)
    flagged = frozenset(c for c in order if does_not_train(board, c))

    if bands_csv is None:
        bands_csv = Path(runs_csv).parent / "bands.csv"
    bands, names = (band_values(bands_csv, family, seq_len, pred_len)
                    if Path(bands_csv).exists() else ({}, {}))

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
           f"- Median gap and Wilcoxon p: the median relative gap of the counted cells, "
           f"and a two-sided signed-rank test on those gaps. A cell whose two sides "
           f"differ by less than {TIE_FLOOR:g} in the metric itself (an absolute "
           f"difference, not a relative gap) is a zero difference: it stays in the "
           f"count and the median but not in the test (the test's default drops zeros), "
           f"so n (test) can be smaller than the cells counted",
           ]
    if any(c[1] in TRAINS_MODELS for c in order):
        out.append(
            "- Trains: for the Transformer only, the seeds whose skill over the "
            "lookback-mean forecast (1 - MSE / lookback-mean MSE over all bands) is at "
            f"least {TRAINS_FRACTION:g} of the best multivariate skill of the other "
            "gradient-trained models on the dataset; a collapsed seed forecasts the "
            "lookback mean (skill 0). Its cell is pooled over every seed as usual, with "
            "a descriptive row over the seeds whose multivariate run trains, at the "
            "pooled cell's best k"
            + ("; a cell where no seed of any paradigm trains is flagged, and the "
               "scoreboard is also given without it" if flagged else ""))
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

    def add(base, chal, label, b, metric, *, skip=frozenset(), gap="rel"):
        t = tally(b, base, chal, metric, order, skip=skip, gap=gap)
        if not t["cells"] and not t["pending"]:
            return
        c = t["count"]
        if math.isnan(t["median"]):
            med = "-"
        elif gap == "rel":
            med = f"{t['median'] * 100:+.2f}%"
        else:
            med = f"{t['median'] * 100:+.2f} pts"
        rows.append(f"| {LABEL[chal]} vs {LABEL[base]} | {label} | "
                    f"{dict(METRICS)[metric]} | {c['challenger']} | {c['tie']} | "
                    f"{c['baseline']} | {t['cells']} | {t['pending']} | {t['n_test']} "
                    f"| {med} | {_p(t['p'])} |")

    for base, chal, _ in COMPARISONS:
        if K1 in (base, chal) or BEST in (base, chal):
            for metric, _name in METRICS:
                add(base, chal, "batch protocol (k chosen on the logged `sl_val/loss`)",
                    batch, metric)
    skill = skill_variant(board)
    for base, chal, _ in COMPARISONS:
        add(base, chal, "ratio-of-sums skill (gap in points of MSE / lookback-mean MSE)",
            skill, "mse", gap="abs")
    near = {ds: chans for ds, chans in NEAR_UNFORECASTABLE.items()
            if any(c[0] == ds for c in order)}
    if bands and near:
        drop_near = drop_variant(board, bands, lambda c: near.get(c[0], ()))
        label = "without " + ", ".join(
            f"{dataset_label(ds)} {'/'.join(_band_label(names, ds, ch) for ch in chans)}"
            for ds, chans in near.items())
        for base, chal, _ in COMPARISONS:
            add(base, chal, label, drop_near, "mse")
    top = {c: top_band(board, c, bands) for c in order if c[1] not in DETERMINISTIC}
    top = {c: t for c, t in top.items() if t}
    if bands and top:
        # A cell with no multivariate band vector has no top band to drop; it
        # is left out of this row rather than counted with every band.
        drop_top = drop_variant(board, bands, lambda c: top.get(c, ()))
        untopped = frozenset(c for c in order if c not in top)
        for base, chal, _ in COMPARISONS:
            add(base, chal, "without each cell's top band (its largest multivariate band "
                "MSE), dropped on both sides", drop_top, "mse", skip=untopped)
    failed = {c: failed_bands(board, c) for c in order}
    failed = {c: f for c, f in failed.items() if f}
    if bands and failed:
        drop_failed = drop_variant(board, bands, lambda c: failed.get(c, ()))
        label = ("without the failed per-channel bands, dropped on both sides ("
                 + "; ".join(f"{dataset_label(c[0])} {model_label(c[1])} "
                             + ", ".join(f"ch{ch}" for ch in f)
                             for c, f in failed.items()) + ")")
        add(MV, PC, label, drop_failed, "mse")
    if flagged:
        label = "without the cells that do not train (" + ", ".join(
            f"{dataset_label(c[0])} {model_label(c[1])}" for c in order if c in flagged) + ")"
        for base, chal, _ in COMPARISONS:
            for metric, _name in METRICS:
                add(base, chal, label, board, metric, skip=flagged)
    if rows:
        out += ["", "## 2 · Sensitivity", "",
                "The scoreboard's comparisons under other choices, counted the same way. "
                "The band variants are MSE only: a joint run logs no per-band MAE, and "
                "they are recomputed from `bands.csv`'s per-band vectors on both sides."
                + ("" if bands else " `bands.csv` was not found, so they are left out."),
                "",
                "| Comparison | Variant | Metric | Challenger better | Tie | "
                "Baseline better | Cells | Pending | n (test) | Median gap | Wilcoxon p |",
                "|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        out += rows

    # -------------------------------------------------------------- per cell
    trained = {c: {pipe: trained_seeds(board, c, pipe) for pipe in PIPES} for c in order}
    alive = {}
    for c in order:
        mv_trained = trained[c][MV]
        if mv_trained and len(mv_trained) < len(cells[c].get(MV, {})):
            alive[c] = mv_trained
    described = {}
    for seeds in {frozenset(s) for s in alive.values()}:
        sub = build(values, "fixed", seeds=set(seeds), choices=board.choices)
        for c, s in alive.items():
            if frozenset(s) == seeds:
                described[c] = sub

    section = 3 if rows else 2
    for metric, name in METRICS:
        out += ["", f"## {section} · Per cell — {name}", "",
                "**Bold** = better side of a comparison outside the tie band; ≈ = inside "
                "it; no mark = not both sides complete, n in brackets. Δ is the "
                "challenger relative to its baseline.", "",
                f"| Dataset | Model | Multivariate | Per-channel | Δ | Cluster (k=1) "
                f"| Cluster (best k) | Δ vs k=1 | n (of {n_family}) | Trains |",
                "|:---|:---|---:|---:|---:|---:|---:|---:|:---|:---|"]
        section += 1
        for c in order:
            out.append(_cell_row(board, c, metric, dataset_label, model_label,
                                 flagged, trained[c]))
            if c in described:
                out.append(_cell_row(described[c], c, metric, dataset_label, model_label,
                                     frozenset(), None, descriptive=True))

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
                    row.append(marked_delta(b, c, K1, BEST, metric))
            out.append("| " + " | ".join(row) + " |")

    # --------------------------------------------------- band sensitivity per cell
    def band_drops(c):
        """(kind, bands) of every band variant that changes cell c. The top
        band is listed where it is not already a named near-unforecastable one."""
        named = near.get(c[0], ())
        extra = tuple(ch for ch in top.get(c, ()) if ch not in named)
        return [(kind, d) for kind, d in (("near-unforecastable", named),
                                          ("top band", extra),
                                          ("failed per-channel", failed.get(c, ()))) if d]

    affected = [c for c in order if c[1] not in DETERMINISTIC and band_drops(c)]
    if bands and affected:
        out += ["", f"## {section} · Band sensitivity per cell (MSE)", "",
                "The cells a band variant changes, with the dropped bands' share of the "
                "multivariate cell's MSE. Both columns of a pair are recomputed from "
                "`bands.csv`. Marks as above.", "",
                "| Dataset | Model | Dropped | Share of MV MSE | Per-channel vs MV: all "
                "| without | Best k vs k=1: all | without |",
                "|:---|:---|:---|---:|---:|---:|---:|---:|"]
        everything = drop_variant(board, bands, lambda c: (), recompute=True)
        for c in affected:
            for kind, drop in band_drops(c):
                cut = drop_variant(board, bands, lambda cc, d=drop: d if cc == c else (),
                                   recompute=True)
                share = _share(bands, c, drop)
                # The failed-band variant is a per-channel repair: the cluster
                # pipelines keep every band in it.
                cluster = kind != "failed per-channel"
                out.append(
                    f"| {dataset_label(c[0])} | {model_label(c[1])} | "
                    f"{', '.join(_band_label(names, c[0], ch) for ch in drop)} ({kind}) | "
                    f"{'-' if share is None else f'{share * 100:.1f}%'} | "
                    f"{marked_delta(everything, c, MV, PC, 'mse')} | "
                    f"{marked_delta(cut, c, MV, PC, 'mse')} | "
                    f"{marked_delta(everything, c, K1, BEST, 'mse') if cluster else '-'} | "
                    f"{marked_delta(cut, c, K1, BEST, 'mse') if cluster else '-'} |")

    out += ["", f"*n = seeds for multivariate / per-channel / cluster k=1 / cluster best "
                f"k, of the family's {n_family}. Trains = the seeds that train of the "
                "pipeline's own seeds, n/N in the same order (Transformer only). A bold Δ "
                "means the challenger (per-channel, or the best k) is better beyond seed "
                "noise.*"]
    return "\n".join(out) + "\n"


def _share(bands: dict, cell: tuple, drop: tuple) -> Optional[float]:
    """The dropped bands' share of the multivariate cell's summed band MSE."""
    per_seed = bands.get((MV, cell[0], cell[1]), {})
    total = kept = 0.0
    for per in per_seed.values():
        for ch, (mse, _) in per.items():
            if mse is None:
                return None
            total += mse
            kept += mse if ch in drop else 0.0
    return kept / total if total else None


def marked_delta(board: Board, cell: tuple, base: str, chal: str, metric: str,
                 marks: bool = True) -> str:
    """Δ of chal against base: **bold** when chal is better beyond the tie band,
    ≈ inside it, unmarked (with the matched n) when a side is incomplete."""
    p = board.cells.get(cell, {})
    if base not in p or chal not in p:
        return "-"
    r = compare(p[base], p[chal], metric)
    if r is None:
        return "-"
    txt = f"{r['rel'] * 100:+.1f}%"
    if not marks:
        return txt
    if not (board.complete(cell, base) and board.complete(cell, chal)):
        return f"{txt} (n={r['n']})"
    if r["outcome"] == "tie":
        return f"≈ {txt}"
    return f"**{txt}**" if r["outcome"] == "challenger" else txt


def _cell_row(board: Board, c: tuple, metric: str, dataset_label, model_label,
              flagged: frozenset, trained: Optional[dict],
              descriptive: bool = False) -> str:
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
        pc_delta = marked_delta(board, c, MV, PC, metric, marks=not descriptive)
        ns = "/".join(str(len(p[x])) if x in p else "-" for x in PIPES)
    k1_delta = "n/a" if det else marked_delta(board, c, K1, BEST, metric, marks=not descriptive)
    label = model_label(c[1])
    if descriptive:
        label += f", seed {', '.join(board.seeds)} only (descriptive)"
    elif c in flagged:
        label += " (does not train)"
    trains = ""
    if trained is not None and c[1] in TRAINS_MODELS:
        # n/N per pipeline: a bare count beside the n column read as "1 of 1".
        trains = " · ".join("-" if trained.get(x) is None
                            else f"{len(trained[x])}/{len(p[x])}" for x in PIPES)
    return (f"| {dataset_label(c[0])} | {label} | {value(MV)} | {value(PC)} | {pc_delta} "
            f"| {value(K1)} | {value(BEST)} | {k1_delta} | {ns} | {trains} |")
