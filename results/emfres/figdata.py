"""What the paradigm-comparison figures plot, computed from frames/ alone.

The figures answer the questions ``tables/paradigms.md`` answers, so they are
built from the same functions: emfres.paradigms.seed_values reads runs.csv,
build() pairs the pipelines and chooses k on validation under the fixed
cluster protocol, compare() and tally() apply the seed-noise tie band and the
Wilcoxon test. Nothing here re-derives a mean, a tie or a best k, so a figure
cannot name a different k, or a different winner, than the table beside it.

This module holds numbers only -- no matplotlib -- so the tests can pin what a
figure shows without drawing it. Every model, dataset and k comes from the
frames: a model whose cluster sweep is still training appears as soon as its
first run is extracted, marked incomplete, and joins the counts once every
seed of every k has landed.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import paradigms as PG
from .frames import family_of

# Datasets no generated figure shows. The 2024 measurement campaigns were
# never part of the benchmark (.personal/RESULTS.md is L336/O96 on ED2022,
# UH2022 and EMF2023 only); a run of them that reaches the frames is dropped
# here rather than drawn as a fourth facet.
HIDDEN_DATASETS = frozenset({"ed2024", "uh2024"})

# Each family's windows, the benchmark first (emfres.cli.WINDOWS mirrors it).
WINDOWS = {"emf": [(336, 96)], "voip": [(48, 12), (24, 6)]}

# The single-series grid's benchmark window: the only one of its four windows
# with more than one seed (audit section 2.4).
SERIES_WINDOW = (336, 96)

# Conformal target: every run calibrates at alpha = 0.1 (args.yaml
# conformal.alpha), and ``ic`` is logged in percent.
TARGET_COVERAGE = 90.0


def visible(dataset: str) -> bool:
    return str(dataset).lower() not in HIDDEN_DATASETS


# --------------------------------------------------------------------------- #
# One family at one window
# --------------------------------------------------------------------------- #

@dataclass
class Corpus:
    """Every pipeline of one family at one window, under both cluster protocols."""
    family: str
    window: tuple
    values: dict                 # paradigms.seed_values, hidden datasets dropped
    board: PG.Board              # the fixed protocol (primary)
    batch: PG.Board              # the logged batch protocol (sensitivity)
    order: list                  # (dataset, model) cells, dataset then model order
    flagged: frozenset = field(default_factory=frozenset)  # cells that do not train

    @property
    def datasets(self) -> list:
        return list(dict.fromkeys(ds for ds, _ in self.order))

    def models(self, dataset: Optional[str] = None) -> list:
        return list(dict.fromkeys(m for ds, m in self.order
                                  if dataset is None or ds == dataset))

    @property
    def n_family(self) -> int:
        return len(self.board.seeds)


def load(runs_csv: Path, family: str, window: tuple,
         dataset_rank: Callable = lambda d: d,
         model_rank: Callable = lambda m: m) -> Optional[Corpus]:
    """The Corpus of one family and window, or None when it has no cell."""
    values = PG.seed_values(runs_csv, family, *window)
    values = {key: v for key, v in values.items() if visible(key[1])}
    if not values:
        return None
    planned = PG.CLUSTERED.get(family, ())
    board = PG.build(values, "fixed", planned=planned)
    batch = PG.build(values, "batch", planned=planned)
    order = sorted(board.cells, key=lambda c: (dataset_rank(c[0]), model_rank(c[1])))
    flagged = frozenset(c for c in order if PG.does_not_train(board, c))
    return Corpus(family, tuple(window), values, board, batch, order, flagged)


def load_family(runs_csv: Path, family: str, **ranks) -> list:
    """The family's corpora, one per window it ran, benchmark first."""
    out = []
    for window in WINDOWS.get(family, []):
        corpus = load(runs_csv, family, window, **ranks)
        if corpus is not None:
            out.append(corpus)
    return out


# --------------------------------------------------------------------------- #
# Per-pipeline summaries
# --------------------------------------------------------------------------- #

@dataclass
class Marks:
    """One pipeline of one cell: its seed values and their mean."""
    seeds: dict            # {seed: value}
    complete: bool         # every seed of the family (and, for best k, a finished sweep)
    k: Optional[int] = None

    @property
    def mean(self) -> float:
        return statistics.fmean(self.seeds.values())

    @property
    def sd(self) -> float:
        return statistics.stdev(self.seeds.values()) if len(self.seeds) > 1 else 0.0

    @property
    def n(self) -> int:
        return len(self.seeds)


def marks(corpus: Corpus, cell: tuple, pipe: str, metric: str,
          board: Optional[PG.Board] = None) -> Optional[Marks]:
    """``pipe``'s seed values of ``metric`` in ``cell``, or None when it has none.

    ``metric`` is a key of paradigms' seed metrics ("mae", "mse") or "skill"
    (1 - MSE / lookback-mean MSE, paradigms.skill_of).
    """
    board = corpus.board if board is None else board
    seeds = board.cells.get(cell, {}).get(pipe)
    if not seeds:
        return None
    if metric == "skill":
        vals = {s: PG.skill_of(v) for s, v in seeds.items()}
        vals = {s: v for s, v in vals.items() if v is not None}
    else:
        vals = {s: v[metric] for s, v in seeds.items() if v.get(metric) is not None}
    if not vals:
        return None
    choice = board.choices.get(cell)
    k = choice.k if pipe == PG.BEST and choice is not None else (1 if pipe == PG.K1 else None)
    return Marks(vals, board.complete(cell, pipe), k)


def trains_threshold(corpus: Corpus, dataset: str) -> Optional[float]:
    """The skill a Transformer seed needs to count as training (paradigms rule)."""
    ref = PG.trains_reference(corpus.board, dataset)
    return None if ref is None or ref <= 0 else PG.TRAINS_FRACTION * ref


# --------------------------------------------------------------------------- #
# Gaps and the scoreboard
# --------------------------------------------------------------------------- #

# (baseline, challenger, short label) of the figure comparisons: the table's
# four minus k = 1 vs multivariate, which is a control, not a question. The
# scoreboard figure keeps all four.
GAP_COMPARISONS = [
    (PG.MV, PG.PC, "Per-channel vs MV"),
    (PG.K1, PG.BEST, "Best k vs k=1"),
    (PG.MV, PG.BEST, "Best k vs MV"),
]
SCORE_COMPARISONS = [
    (PG.MV, PG.PC, "Per-channel vs MV"),
    (PG.MV, PG.K1, "k=1 vs MV"),
    (PG.K1, PG.BEST, "Best k vs k=1"),
    (PG.MV, PG.BEST, "Best k vs MV"),
]


@dataclass
class Gap:
    cell: tuple
    rel: float           # challenger relative to baseline, e.g. -0.02 = 2% lower
    band: float          # the tie band relative to the baseline mean
    outcome: str         # "challenger", "tie", "baseline", or "pending"
    n: int


def gaps(corpus: Corpus, base: str, chal: str, metric: str) -> list:
    """One Gap per non-deterministic cell where both sides have a value.

    A cell whose sides are not both complete is "pending": it is plotted at
    its matched-seed gap but carries no outcome, exactly as the tables leave
    it out of every count.
    """
    board = corpus.board
    out = []
    for c in corpus.order:
        if c[1] in PG.DETERMINISTIC:
            continue
        p = board.cells.get(c, {})
        if base not in p or chal not in p:
            continue
        r = PG.compare(p[base], p[chal], metric)
        if r is None or not r["a"]:
            continue
        done = board.complete(c, base) and board.complete(c, chal)
        out.append(Gap(c, r["rel"], r["band"] / r["a"],
                       r["outcome"] if done else "pending", r["n"]))
    return out


def score(corpus: Corpus, base: str, chal: str, metric: str,
          skip: frozenset = frozenset()) -> dict:
    """paradigms.tally over the corpus' cells (wins, ties, losses, pending, p)."""
    return PG.tally(corpus.board, base, chal, metric, corpus.order, skip=skip)


def pending_cells(corpus: Corpus, base: str, chal: str,
                  skip: frozenset = frozenset()) -> list:
    """The cells score() counts as pending, in corpus order.

    paradigms.tally's rule: the comparison applies to the cell (a planned
    cluster pipeline counts, run or not), the model is not deterministic, and
    a side is short of seeds, still sweeping or has no run yet. gaps() draws
    those with a value on both sides; the rest have nothing to plot yet.
    """
    board = corpus.board
    return [c for c in corpus.order
            if c not in skip and c[1] not in PG.DETERMINISTIC
            and board.applies(c, base) and board.applies(c, chal)
            and not (board.complete(c, base) and board.complete(c, chal))]


def unswept(corpus: Corpus) -> list:
    """(dataset, model) cells with a planned cluster pipeline and no run yet."""
    board = corpus.board
    return [c for c in corpus.order if c[1] not in PG.DETERMINISTIC
            and board.applies(c, PG.BEST) and not board.sweeps.get(c)]


# --------------------------------------------------------------------------- #
# Clustering gain against k
# --------------------------------------------------------------------------- #

@dataclass
class KPoint:
    k: int
    rel: float           # seed-mean relative change against k = 1
    sd: float            # its seed standard deviation
    n: int               # matched seeds
    complete: bool       # every seed of the family


@dataclass
class KProfile:
    cell: tuple
    points: list         # KPoint per k, ascending
    chosen: Optional[int]      # the validation-chosen k (None while the sweep is partial)
    oracle: Optional[int]      # the k with the lowest test value over the chosen k's seeds


def k_profile(corpus: Corpus, cell: tuple, metric: str, protocol: str = "fixed") -> Optional[KProfile]:
    """A cell's sweep as the change against its own k = 1, seed by seed.

    Each k is compared to k = 1 on the seeds both have; the value is the mean
    of the per-seed relative changes and the ribbon their sample standard
    deviation. ``chosen`` is the board's validation-chosen k when the sweep is
    finished (never while it is partial: a partial ranking would move), and
    ``oracle`` the k the test split would pick among the same seeds, as
    tables.render_ccm_sweep reports it.
    """
    board = corpus.board if protocol == "fixed" else corpus.batch
    sweep = board.sweeps.get(cell)
    if not sweep or 1 not in sweep:
        return None
    base = sweep[1]
    points = []
    for k in sorted(sweep):
        seeds = sorted(set(sweep[k]) & set(base))
        rels = [sweep[k][s][metric] / base[s][metric] - 1.0 for s in seeds
                if base[s].get(metric) and sweep[k][s].get(metric) is not None]
        if not rels:
            continue
        points.append(KPoint(k, statistics.fmean(rels),
                             statistics.stdev(rels) if len(rels) > 1 else 0.0,
                             len(rels), set(sweep[k]) == set(board.seeds)))
    choice = board.choices.get(cell)
    chosen = choice.k if choice is not None and choice.k is not None and not choice.partial else None
    oracle = None
    if chosen is not None:
        scored = {k: statistics.fmean(v[metric] for v in s.values())
                  for k, s in sweep.items() if s and set(s) == set(choice.seeds)}
        oracle = min(scored, key=lambda kk: (scored[kk], kk)) if scored else None
    return KProfile(cell, points, chosen, oracle)


# --------------------------------------------------------------------------- #
# Per-band views (frames/bands.csv)
# --------------------------------------------------------------------------- #

@dataclass
class BandGrid:
    """log2(challenger / baseline) band MSE per (model, band) of one dataset."""
    dataset: str
    models: list                 # rows
    channels: list               # columns, loader order
    names: dict                  # {channel: band name}
    ratio: dict                  # {(model, channel): log2 ratio}
    diff: dict                   # {(model, channel): challenger - baseline band MSE}
    n: dict                      # {model: matched seeds}

    def share(self, skip=()) -> dict:
        """Each band's signed share of the dataset's band gaps.

        A band's gap is the sum over models of (challenger - baseline) band
        MSE, so models that disagree in sign on a band cancel there; its
        share is that gap over the sum of every band's |gap|. The shares'
        absolute values therefore sum to one, and a gap carried by a few
        bands shows as a few long bars. Positive = the baseline is better on
        that band. Models in ``skip`` (a model that does not train here,
        whose band gaps say nothing about the paradigms) enter no band's gap.
        """
        models = [m for m in self.models if m not in set(skip)]
        gap = {ch: sum(self.diff.get((m, ch), 0.0) for m in models) for ch in self.channels}
        total = sum(abs(v) for v in gap.values())
        if not total:
            return {}
        return {ch: v / total for ch, v in gap.items()}


def band_grids(corpus: Corpus, bands_csv: Path, base: str = PG.MV,
               chal: str = PG.PC) -> list:
    """One BandGrid per dataset of the corpus, from the pipelines' band vectors.

    Band MSE is averaged over the seeds both pipelines have with a full band
    vector (a cluster pipeline's fixed vector, under the fixed protocol).
    Only cells whose two sides are complete enter; a model with none is left
    out of the dataset's rows.
    """
    if not Path(bands_csv).exists():
        return []
    band_vals, names = PG.band_values(bands_csv, corpus.family, *corpus.window)
    board = corpus.board
    out = []
    for dataset in corpus.datasets:
        models, ratio, diff, ns, chans = [], {}, {}, {}, set()
        for c in corpus.order:
            if c[0] != dataset or c[1] in PG.DETERMINISTIC:
                continue
            if not (board.complete(c, base) and board.complete(c, chal)):
                continue
            sides = []
            for pipe in (base, chal):
                key = PG._source_key(board, c, pipe)
                per_seed = band_vals.get((key, c[0], c[1]), {}) if key else {}
                fixed = pipe in (PG.K1, PG.BEST) and board.protocol == "fixed"
                seeds = set(board.cells[c][pipe])
                sides.append({s: {ch: v[1] if fixed else v[0] for ch, v in per.items()}
                              for s, per in per_seed.items() if s in seeds})
            seeds = sorted(set(sides[0]) & set(sides[1]))
            common = None
            for s in seeds:
                keys = {ch for ch, v in sides[0][s].items() if v is not None} & \
                       {ch for ch, v in sides[1][s].items() if v is not None}
                common = keys if common is None else common & keys
            if not seeds or not common:
                continue
            models.append(c[1])
            ns[c[1]] = len(seeds)
            for ch in common:
                a = statistics.fmean(sides[0][s][ch] for s in seeds)
                b = statistics.fmean(sides[1][s][ch] for s in seeds)
                if a > 0 and b > 0:
                    ratio[(c[1], ch)] = math.log2(b / a)
                diff[(c[1], ch)] = b - a
                chans.add(ch)
        if models:
            out.append(BandGrid(dataset, models, sorted(chans),
                                {ch: names.get((dataset, ch), "") for ch in chans},
                                ratio, diff, ns))
    return out


# --------------------------------------------------------------------------- #
# Cost and calibration
# --------------------------------------------------------------------------- #

def _f(v) -> Optional[float]:
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None


@dataclass
class Cost:
    """Seed-mean training seconds of one pipeline; lo/hi bracket a per-channel estimate.

    ``foreign`` counts the seeds whose run trained while another user's
    process, or another launch of this repo, was on the GPU (runs.csv
    foreign_overlap > 0, specs/contention.yaml foreign and shared windows):
    their logged time may be inflated. Always 0 for the
    per-channel estimate, which is the solo pass's.
    """
    seconds: float
    lo: Optional[float] = None
    hi: Optional[float] = None
    n: int = 0
    foreign: int = 0


def costs(corpus: Corpus, runs_csv: Path, timing_csv: Path) -> dict:
    """{(cell, pipe): Cost} at the corpus' window.

    Multivariate and cluster runs train one at a time, so their logged
    running_time is their cost; a cluster pipeline costs what its own runs
    (the validation-chosen k) logged. A per-channel pipeline trains a dozen
    bands at once, so its logged time is contended: its cost is timing.csv's
    uncontended estimate (every band's epochs at the solo pass's seconds per
    epoch, summed over bands) with the fastest and slowest measured band as
    the bracket, averaged over the seeds the paradigm comparison keeps. No
    timing.csv, no per-channel cost. A logged cost counts the seeds whose run
    overlapped another user's GPU process (Cost.foreign).
    """
    board = corpus.board
    seq_len, pred_len = corpus.window
    times: dict = {}
    with open(runs_csv) as f:
        for r in csv.DictReader(f):
            if (r.get("family") or family_of(r["dataset"])) != corpus.family:
                continue
            if int(r["seq_len"]) != seq_len or int(r["pred_len"]) != pred_len:
                continue
            if r["setup"] not in ("multivariate", "cluster") or r["conf"] != "1":
                continue
            t = _f(r.get("running_time"))
            if t is None:
                continue
            k = int(r["k"]) if r["setup"] == "cluster" else None
            times[(r["setup"], r["dataset"], r["model"], k, r["seed"])] = \
                (t, bool(_f(r.get("foreign_overlap"))))
    out = {}
    for c in corpus.order:
        p = board.cells.get(c, {})
        for pipe, setup, k in ((PG.MV, "multivariate", None),
                               (PG.BEST, "cluster", board.choices[c].k
                                if c in board.choices else None)):
            if pipe not in p or (pipe == PG.BEST and k is None):
                continue
            ts = [times.get((setup, c[0], c[1], k, s)) for s in p[pipe]]
            ts = [t for t in ts if t is not None]
            if ts:
                out[(c, pipe)] = Cost(statistics.fmean(t for t, _ in ts), n=len(ts),
                                      foreign=sum(1 for _, f in ts if f))
    if Path(timing_csv).exists():
        solo: dict = {}
        with open(timing_csv) as f:
            for r in csv.DictReader(f):
                if r.get("family", "emf") != corpus.family:
                    continue
                if int(r["seq_len"]) != seq_len or int(r["pred_len"]) != pred_len:
                    continue
                vals = [_f(r.get(x)) for x in ("solo_total", "solo_lo", "solo_hi")]
                if None not in vals:
                    solo[(r["dataset"], r["model"], r["seed"])] = vals
        for c in corpus.order:
            seeds = board.cells.get(c, {}).get(PG.PC, {})
            rows = [solo[(c[0], c[1], s)] for s in seeds if (c[0], c[1], s) in solo]
            if rows:
                out[(c, PG.PC)] = Cost(statistics.fmean(r[0] for r in rows),
                                       statistics.fmean(r[1] for r in rows),
                                       statistics.fmean(r[2] for r in rows), len(rows))
    return out


def calibration(corpus: Corpus, runs_csv: Path) -> dict:
    """{(cell, pipe): {"coverage": Marks, "width": Marks}} for every pipeline.

    ``ic`` (per-channel coverage, percent) and ``interval_width`` of each run
    -- the fixed-protocol twins for a cluster run, as the fixed protocol is
    the primary one -- on the seeds the paradigm comparison keeps for the
    pipeline, so they pair with its accuracy; a per-channel seed is the mean
    over its bands.
    """
    board = corpus.board
    seq_len, pred_len = corpus.window
    raw: dict = {}
    with open(runs_csv) as f:
        for r in csv.DictReader(f):
            if (r.get("family") or family_of(r["dataset"])) != corpus.family:
                continue
            if int(r["seq_len"]) != seq_len or int(r["pred_len"]) != pred_len:
                continue
            if r["conf"] != "1" or r["setup"] not in ("multivariate", "perchannel", "cluster"):
                continue
            if r["setup"] == "cluster":
                ic, w = _f(r.get("ic_fixed")), _f(r.get("interval_width_fixed"))
                key = (("cluster", int(r["k"])), r["dataset"], r["model"])
            else:
                ic, w = _f(r.get("ic")), _f(r.get("interval_width"))
                key = (r["setup"], r["dataset"], r["model"])
            if ic is None or w is None:
                continue
            raw.setdefault(key, {}).setdefault(r["seed"], []).append((ic, w))
    out = {}
    for c in corpus.order:
        p = board.cells.get(c, {})
        for pipe in PG.PIPES:
            key = PG._source_key(board, c, pipe)
            if pipe not in p or key is None:
                continue
            by_seed = raw.get((key, c[0], c[1]), {})
            per = {s: by_seed[s] for s in p[pipe] if s in by_seed}
            if not per:
                continue
            complete = board.complete(c, pipe) and len(per) == len(p[pipe])
            k = (board.choices[c].k if pipe == PG.BEST else 1) if pipe in (PG.K1, PG.BEST) else None
            out[(c, pipe)] = {
                "coverage": Marks({s: statistics.fmean(ic for ic, _ in rs) for s, rs in per.items()},
                                  complete, k),
                "width": Marks({s: statistics.fmean(w for _, w in rs) for s, rs in per.items()},
                               complete, k)}
    return out


# --------------------------------------------------------------------------- #
# The single-series grid (a separate dataset)
# --------------------------------------------------------------------------- #

def series_values(runs_csv: Path, window: tuple = SERIES_WINDOW) -> dict:
    """{(dataset, model): {seed: {"mae", "mse"}}} of the single-series grid.

    A different target, dataset and calendar from the paradigm trees (audit
    section 7): never paired or ranked with them, only drawn beside itself.
    """
    out: dict = {}
    with open(runs_csv) as f:
        for r in csv.DictReader(f):
            if r["setup"] != "series" or not visible(r["dataset"]):
                continue
            if (int(r["seq_len"]), int(r["pred_len"])) != tuple(window):
                continue
            mae, mse = _f(r.get("mae")), _f(r.get("mse"))
            if mae is None or mse is None or not r["seed"]:
                continue
            out.setdefault((r["dataset"], r["model"]), {})[r["seed"]] = {"mae": mae, "mse": mse}
    return out


# --------------------------------------------------------------------------- #
# Spot check: the figures' numbers against cells.csv
# --------------------------------------------------------------------------- #

def spot_check(corpus: Corpus, cells_csv: Path, tol: float = 1e-9,
               runs_csv: Optional[Path] = None) -> tuple:
    """(checked, mismatches) of the seed means the figures draw against cells.csv.

    The seed values every accuracy figure is drawn from (P1, P2, P4, P5,
    P6, P9): every multivariate cell, every per-channel cell whose seeds are
    all complete (cells.csv averages every band x seed run, which equals the
    mean of per-seed band means only then) and every cluster k with a fixed
    value for all its seeds. With ``runs_csv``, also P8's coverage and width
    (calibration) and P7's multivariate and cluster training time (costs)
    under the same conditions. Not checked, having no cells.csv counterpart:
    P7's per-channel cost (timing.csv's solo estimate) and P3's band ratios
    (bands.csv); P10 has spot_check_series. Derived from the frames, so it
    holds after every re-extraction: a figure that drifted from the tables
    fails it.
    """
    if not Path(cells_csv).exists():
        return 0, []
    seq_len, pred_len = corpus.window
    table: dict = {}
    with open(cells_csv) as f:
        for r in csv.DictReader(f):
            if (r.get("family") or family_of(r["dataset"])) != corpus.family:
                continue
            if int(r["seq_len"]) != seq_len or int(r["pred_len"]) != pred_len or r["conf"] != "1":
                continue
            k = int(float(r["k"])) if r.get("k") not in ("", None) else None
            table[(r["setup"], r["dataset"], r["model"], k)] = r
    checked, bad = 0, []

    def check(key, seeds, cols):
        nonlocal checked
        row = table.get(key)
        if row is None or not seeds:
            return
        for metric, col in cols:
            want = _f(row.get(col))
            if want is None:
                continue
            got = statistics.fmean(s[metric] for s in seeds.values())
            checked += 1
            if abs(got - want) > tol * max(1.0, abs(want)):
                bad.append((key, metric, got, want))

    for (pipeline, dataset, model), seeds in corpus.values.items():
        if pipeline == PG.MV:
            check(("multivariate", dataset, model, None), seeds, (("mae", "mae"), ("mse", "mse")))
        elif pipeline == PG.PC:
            row = table.get(("perchannel", dataset, model, None))
            if row and _f(row.get("n_complete_seeds")) == _f(row.get("n_seeds")) == len(seeds):
                check(("perchannel", dataset, model, None), seeds,
                      (("mae", "mae"), ("mse", "mse")))
        else:
            fixed = PG.view(seeds, "fixed")
            row = table.get(("cluster", dataset, model, pipeline[1]))
            if row and len(fixed) == len(seeds) and _f(row.get("n_seeds")) == len(seeds):
                check(("cluster", dataset, model, pipeline[1]), fixed,
                      (("mae", "mae_fixed"), ("mse", "mse_fixed")))
    if runs_csv is None:
        return checked, bad

    def whole(key, n, pipe) -> bool:
        """cells.csv's plain mean is the seed mean only over every seed, all complete."""
        row = table.get(key)
        return bool(row) and _f(row.get("n_seeds")) == n and (
            pipe != PG.PC or _f(row.get("n_complete_seeds")) == n)

    setups = {PG.MV: "multivariate", PG.PC: "perchannel"}
    seen = set()
    for (cell, pipe), m in calibration(corpus, runs_csv).items():
        cov, width = m["coverage"], m["width"]
        if pipe in (PG.K1, PG.BEST):
            key, cols = ("cluster", *cell, cov.k), (("coverage", "ic_fixed"),
                                                    ("width", "interval_width_fixed"))
        else:
            key, cols = (setups[pipe], *cell, None), (("coverage", "ic"),
                                                      ("width", "interval_width"))
        if key in seen or not cov.complete or not whole(key, cov.n, pipe):
            continue
        seen.add(key)
        check(key, {s: {"coverage": cov.seeds[s], "width": width.seeds[s]} for s in cov.seeds},
              cols)
    timing = Path(runs_csv).with_name("timing.csv")
    for (cell, pipe), cost in costs(corpus, runs_csv, timing).items():
        if pipe == PG.PC:
            continue
        k = corpus.board.choices[cell].k if pipe == PG.BEST else None
        key = ("cluster" if pipe == PG.BEST else "multivariate", *cell, k)
        row = table.get(key)
        if not row or _f(row.get("n_seeds")) != cost.n:
            continue
        want = _f(row.get("running_time"))
        if want is None:
            continue
        checked += 1
        if abs(cost.seconds - want) > tol * max(1.0, abs(want)):
            bad.append((key, "running_time", cost.seconds, want))
    return checked, bad


def spot_check_series(runs_csv: Path, cells_csv: Path, tol: float = 1e-9) -> tuple:
    """(checked, mismatches) of P10's series seed means against cells.csv."""
    if not Path(cells_csv).exists():
        return 0, []
    seq_len, pred_len = SERIES_WINDOW
    table = {}
    with open(cells_csv) as f:
        for r in csv.DictReader(f):
            if r["setup"] == "series" and r["conf"] == "1" and \
                    (int(r["seq_len"]), int(r["pred_len"])) == (seq_len, pred_len):
                table[(r["dataset"], r["model"])] = r
    checked, bad = 0, []
    for key, seeds in series_values(runs_csv).items():
        row = table.get(key)
        if not row or _f(row.get("n_seeds")) != len(seeds):
            continue
        for metric in ("mae", "mse"):
            want = _f(row.get(metric))
            if want is None:
                continue
            got = statistics.fmean(v[metric] for v in seeds.values())
            checked += 1
            if abs(got - want) > tol * max(1.0, abs(want)):
                bad.append((("series", *key), metric, got, want))
    return checked, bad
