"""logs/ → tidy CSV. The numeric API everything downstream reads.

Today the chain is logs → extract_results → tables.md → four plotters that
re-parse that markdown with regexes keyed on auto-generated section numbers.
This module is the layer that ends it: one row per run in ``frames/runs.csv``,
one row per aggregated cell in ``frames/cells.csv``, and a manifest recording
exactly which logs went in.

The CSVs are meant to be tracked. They are small, they diff meaningfully
("MAE moved in the third decimal"), and they survive a logs/ pruning -- commit
e84faec4 already deleted log cells whose numbers had been published.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .runlogs import Run, equivalent_code, load_runs

# logs/ directory name -> the setup vocabulary results/ uses. "univariate"
# means three different things across this repo -- the per-band grid's CLI
# name, the single-series log tree, and a plot_metrics section -- so it is
# retired here in favour of two unambiguous names.
SETUPS = {
    "multivariate": "multivariate",
    "univariate": "series",
    "univariate (per channel)": "perchannel",
    "cluster": "cluster",
}

# logs/ subdirectories holding runs kept for provenance but out of the
# published numbers. The "_" prefix is the convention; logs/_archive holds the
# ED2022 lookback ladder (L = 48..480 by 48, O = 96, 81 cells over
# multivariate/per-channel/cluster), parked there 2026-08-14 with its L336
# rung left in the live tree as the benchmark cell, and since 2026-08-25 the
# retired upstream-network cluster runs (seeds 42/43 and 69421-3 alike) that
# the native-CCM rerun replaced. The unprefixed spellings
# are a safety net: an archive that misses this test does not vanish from the
# frames, it silently reappears as a setup named after its own directory.
ARCHIVE_DIRS = {"archive", "archived"}

# Dataset slug prefix -> corpus. The three paradigms run over two unrelated
# corpora that share every driver, log tree and metric: the Italy EMF
# measurements and the pooled VoIP call-quality traces. They are never
# comparable to each other -- different channels, different windows, different
# split -- so a table or figure renders one family at a time, and the column
# below is what makes that filter possible without re-deriving it from a
# dataset list that would go stale the next time a corpus is added.
VOIP_PREFIX = "voip_"


def family_of(dataset: str) -> str:
    """The corpus a dataset slug belongs to: "voip" or "emf"."""
    return "voip" if str(dataset).lower().startswith(VOIP_PREFIX) else "emf"


# The paths whose contents decide what a training run computes -- the same set
# forecasting/provenance.py checks for a dirty tree. Kept in step by hand: the
# results package does not import the training package.
CODE_PATHS = ("forecasting", "main.py", "pyproject.toml", "TimeSeriesCCM-main")

# Extensions a test's data fixture can carry: forecasting/test_golden.json holds
# the hashes test_golden.py pins the model builds to.
FIXTURE_SUFFIXES = (".json", ".csv", ".npy", ".npz")


def _is_test_module(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.startswith("test_") and name.endswith(".py")


def _folder(path: str) -> str:
    return path.rpartition("/")[0]


def _inert(path: str, test_dirs: frozenset = frozenset()) -> bool:
    """True for a file under CODE_PATHS that no training run ever reads.

    That is markdown, test modules (test_*.py), and the data fixtures those
    tests load: a test_* file with a FIXTURE_SUFFIXES extension in a folder of
    ``test_dirs``, the folders holding a test module. A fixture is inert only
    beside a test module, so a test_* data file a loader might read elsewhere
    still counts as training code.
    """
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".md") or _is_test_module(path):
        return True
    return (name.startswith("test_") and name.endswith(FIXTURE_SUFFIXES)
            and _folder(path) in test_dirs)


_FINGERPRINTS: dict = {}


def code_fingerprint(repo: Path, commit: Optional[str]) -> str:
    """A 12-hex id for the training code at ``commit``, or "" if unknown.

    Two commits that differ only in results/, logs/, data/ or documentation
    train identically, so provenance is judged on this rather than on the
    commit id: a run stamped before a results-only commit and one stamped after
    it are the same code version. Markdown, tests and the fixtures the tests
    load are left out for the same reason (see _inert) -- no training run reads
    them, so a tests-only commit keeps the version.
    """
    if not commit:
        return ""
    key = (str(repo), commit)
    if key not in _FINGERPRINTS:
        try:
            listing = subprocess.run(
                ["git", "-C", str(repo), "ls-tree", "-r", commit, "--", *CODE_PATHS],
                capture_output=True, text=True, check=True).stdout
        except Exception:
            _FINGERPRINTS[key] = ""
        else:
            blobs = [(path, meta.split()[2])
                     for meta, _, path in (l.partition("\t") for l in listing.splitlines())
                     if path]
            test_dirs = frozenset(_folder(p) for p, _ in blobs if _is_test_module(p))
            lines = sorted(f"{path} {sha}" for path, sha in blobs
                           if not _inert(path, test_dirs))
            _FINGERPRINTS[key] = (hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12]
                                  if lines else "")
    return _FINGERPRINTS[key]

RUN_FIELDS = [
    "setup", "family", "model", "dataset", "seq_len", "pred_len", "k", "channel",
    "seed", "conf", "mse", "mae", "val_loss",
    # Against the trivial forecast (emfres/baselines.py): the test MSE of
    # repeating each window's lookback mean on the same windows, and the run's
    # skill over it (1 - mse / lookback_mse).
    "lookback_mse", "skill",
    "running_time", "per_epoch", "epochs",
    # The 0-based epoch of val_loss, the checkpoint early stopping kept
    # (runlogs.Run.best_epoch); blank for the fit-once models.
    "best_epoch", "num_params",
    "ic", "jc", "interval_width",
    # Cluster runs only: the same metrics under the fixed train-mean membership
    # (the primary cluster protocol; mse, mae, val_loss, ic, jc and
    # interval_width above stay the logged batch protocol), and where they
    # came from -- runlogs.FIXED_SOURCES: log, sidecar (eval_fixed.json), k1
    # (the batch values: one cluster has no membership to fix) or none -- and
    # the fingerprint of the code that computed them: the run's own for log
    # and k1, the re-score's commit for a sidecar (blank when unknown, e.g. a
    # re-score on a dirty tree). Blank on every other setup.
    "mse_fixed", "mae_fixed", "val_loss_fixed",
    "ic_fixed", "jc_fixed", "interval_width_fixed", "fixed_source",
    "fixed_code_version",
    # Provenance (forecasting/provenance.py): the commit the run was stamped
    # with or dated to, whether the tree was dirty, how that commit was
    # established (see runlogs.Run.commit_source), the fingerprint of the
    # training code actually run (blank when it cannot be established), the
    # version it counts as under specs/provenance.yaml's equivalent_code
    # rules (code_version itself when no rule names it), and how many
    # trainings shared the GPU.
    "git_commit", "git_dirty", "commit_source", "code_version", "code_equivalent",
    "concurrency",
    "source", "path",
]

CELL_FIELDS = [
    "setup", "family", "model", "dataset", "seq_len", "pred_len", "k", "conf",
    # mse_std / mae_std are the SEED standard deviation in every setup: over
    # the runs of a joint cell (one per seed), over the per-seed band means of
    # a per-channel cell (see _seed_band_means). The spread over every
    # band x seed run of a per-channel cell -- what mse_std held there before
    # 2026-09-24 -- is mse_band_seed_sd / mae_band_seed_sd, blank elsewhere.
    "mse", "mse_std", "mae", "mae_std", "mse_band_seed_sd", "mae_band_seed_sd",
    "val_loss", "running_time", "per_epoch",
    "num_params", "ic", "jc", "interval_width",
    # Cluster cells: the fixed-membership twins of the columns above, blank
    # unless every run of the cell has a value; fixed_sources lists the
    # runs' fixed_source values, ';'-joined.
    "mse_fixed", "mse_fixed_std", "mae_fixed", "mae_fixed_std", "val_loss_fixed",
    "ic_fixed", "jc_fixed", "interval_width_fixed", "fixed_sources",
    # n_seeds counts every seed with a run, as mse and mae average every run
    # (extract_results.aggregate parity, what reconcile checks); a
    # per-channel seed still training counts there. n_complete_seeds is the
    # sample size of mse_std: the seeds of a joint cell with an MSE, the
    # complete seeds of a per-channel cell (complete_seeds).
    "n_runs", "n_seeds", "n_complete_seeds", "n_channels",
    # Distinct training-code versions behind the cell's runs, ';'-joined, with
    # "?" for any run whose code cannot be established; more than one means
    # the cell mixes code versions and is not a clean seed mean --
    # unless code_equivalents, the same under the equivalent_code rules,
    # shows one.
    "code_versions", "code_equivalents",
]

# One row per (run, band): the per-band test error behind every joint and
# per-channel cell. A joint (multivariate or cluster) run contributes one row
# per entry of its sl_test/mse_per_channel, a per-channel run its one band.
# channel is the loader's channel index in both, which is the same column of
# the same file (per-channel runs select it with data.target_channel), and
# band is that column's header: the frequency in MHz on the EMF corpus, the
# signal name on VoIP. Joint runs log no per-band MAE, so mae is blank there;
# mse_fixed is the fixed-membership twin on cluster rows (fixed_source as in
# runs.csv). lookback_mse is the band's lookback-mean baseline on the same
# windows. The series tree is one aggregate series, not bands, and has no rows.
BAND_FIELDS = [
    "setup", "family", "model", "dataset", "seq_len", "pred_len", "k", "seed",
    "conf", "channel", "band", "mse", "mae", "mse_fixed", "fixed_source",
    "lookback_mse",
]

# Uncontended per-channel training cost, one row per complete per-channel seed
# of a (model, dataset, window) the solo timing pass measured
# (specs/timing.yaml). The per-channel grid trains a dozen bands at once, so
# its logged running_time is a contended per-band time; the solo pass retrained
# three bands of every cell alone, and a seed's uncontended cost is estimated
# as its total epochs over every band times the solo seconds per epoch
# (solo_total), bracketed by the fastest and slowest measured band
# (solo_lo / solo_hi). contended_total is the logged sum over bands, at the
# concurrency the bands logged ("?" for bands logged before it was stamped),
# and mv_running_time the multivariate run of the same seed (trained alone,
# like every non-per-channel grid).
TIMING_FIELDS = [
    "family", "model", "dataset", "seq_len", "pred_len", "seed", "n_bands",
    "epochs", "solo_bands", "solo_per_epoch", "solo_total", "solo_lo", "solo_hi",
    "contention_mean", "contended_total", "concurrency", "mv_running_time",
    "solo_x_mv", "contended_x_mv",
]

EPOCH_FIELDS = ["setup", "model", "dataset", "seq_len", "pred_len", "k",
                "channel", "seed", "conf", "epoch", "train_loss", "val_loss"]


def setup_of(logs_dir: Path) -> str:
    """The setup slug for a logs/ subdirectory, falling back to its own name."""
    return SETUPS.get(logs_dir.name.lower(), logs_dir.name.lower())


def is_archive(logs_dir: Path) -> bool:
    """True for a logs/ subdirectory that is provenance rather than data.

    Matches the "_" prefix and the bare spellings alike, because the fallback
    in setup_of() is silent: anything not in SETUPS becomes a setup named
    after its own directory, so an archive that is not skipped here does not
    vanish from the frames -- it reappears under a setup nobody reads, with
    its paradigm collapsed into one label.
    """
    return logs_dir.name.startswith("_") or logs_dir.name.lower() in ARCHIVE_DIRS


def _baseline(run: Run) -> tuple:
    """(lookback_mse, skill) cells for one run, blank when unavailable."""
    from .baselines import lookback_mean_mse, skill

    if run.mse is None or run.train_split is None or run.val_split is None:
        return "", ""
    per_channel = lookback_mean_mse(run.dataset, run.seq_len, run.pred_len,
                                    run.train_split, run.val_split)
    if per_channel is None:
        return "", ""
    if run.channel is not None:
        if run.channel >= len(per_channel):
            return "", ""
        base = float(per_channel[run.channel])
    else:
        # A joint run's MSE averages every channel over the same windows.
        base = float(per_channel.mean())
    return _num(base), _num(skill(run.mse, base))


def _row(run: Run, setup: str, repo: Path,
         equivalent: Optional[list[dict]] = None) -> dict:
    lookback, skill_value = _baseline(run)
    version = code_fingerprint(repo, run.code_commit)
    return {
        "setup": setup,
        "family": family_of(run.dataset),
        "model": run.model,
        "dataset": run.dataset,
        "seq_len": run.seq_len,
        "pred_len": run.pred_len,
        "k": "" if run.k is None else run.k,
        "channel": "" if run.channel is None else run.channel,
        "seed": run.seed or "",
        "conf": int(run.conf),
        "mse": _num(run.mse),
        "mae": _num(run.mae),
        "val_loss": _num(run.val_loss),
        "lookback_mse": lookback,
        "skill": skill_value,
        "running_time": _num(run.running_time),
        "per_epoch": _num(run.per_epoch),
        "epochs": "" if run.epochs is None else run.epochs,
        "best_epoch": "" if run.best_epoch is None else run.best_epoch,
        "num_params": "" if run.num_params is None else run.num_params,
        "ic": _num(run.ic),
        "jc": _num(run.jc),
        "interval_width": _num(run.interval_width),
        "mse_fixed": _num(run.mse_fixed),
        "mae_fixed": _num(run.mae_fixed),
        "val_loss_fixed": _num(run.val_loss_fixed),
        "ic_fixed": _num(run.ic_fixed),
        "jc_fixed": _num(run.jc_fixed),
        "interval_width_fixed": _num(run.interval_width_fixed),
        "fixed_source": run.fixed_source,
        "fixed_code_version": (code_fingerprint(repo, run.fixed_commit)
                               if run.fixed_source else ""),
        "git_commit": run.git_commit or "",
        "git_dirty": "" if run.git_dirty is None else int(run.git_dirty),
        "commit_source": run.commit_source,
        "code_version": version,
        "code_equivalent": equivalent_code(equivalent or [], version),
        "concurrency": "" if run.concurrency is None else run.concurrency,
        "source": "log",
        "path": run.path.relative_to(repo).as_posix(),
    }


def _num(v: Optional[float]) -> str:
    """Full repr, so a round-trip through CSV loses nothing."""
    return "" if v is None else repr(float(v))


def load_all(logs_dir: Path, exclude: list[str],
             inferred: Optional[list[dict]] = None,
             attested: Optional[list[dict]] = None) -> list[tuple[str, Run]]:
    """(setup, Run) for every finished log.json under ``logs_dir``.

    Read once and shared by build_runs and build_bands, so the two frames
    describe exactly the same set of runs.
    """
    out = []
    for sub in sorted(p for p in logs_dir.iterdir() if p.is_dir()):
        if is_archive(sub):
            continue
        setup = setup_of(sub)
        out += [(setup, run) for run in load_runs(sub, exclude, setup=setup,
                                                  inferred=inferred, attested=attested)]
    return out


def build_runs(logs_dir: Path, repo: Path, exclude: list[str],
               inferred: Optional[list[dict]] = None,
               attested: Optional[list[dict]] = None,
               equivalent: Optional[list[dict]] = None,
               loaded: Optional[list[tuple[str, Run]]] = None) -> list[dict]:
    """One row per log.json across every setup under ``logs_dir``.

    ``inferred``, ``attested`` and ``equivalent`` are the rule lists from
    specs/provenance.yaml: the first dates runs logged before provenance
    stamping, the second resolves runs stamped on a dirty tree (see
    runlogs.infer_commit / attest_commit), the third counts one training-code
    version as another (runlogs.equivalent_code). ``loaded`` is load_all's
    result, when the caller already has it.
    """
    if loaded is None:
        loaded = load_all(logs_dir, exclude, inferred, attested)
    return [_row(run, setup, repo, equivalent) for setup, run in loaded]


def build_epochs(logs_dir: Path, repo: Path, exclude: list[str]) -> list[dict]:
    """Per-epoch train/val loss -- what the loss-curve family needs.

    Kept separate from runs.csv because it is the one genuinely long frame:
    a row per (run, epoch) rather than a row per run.
    """
    rows = []
    for sub in sorted(p for p in logs_dir.iterdir() if p.is_dir()):
        if is_archive(sub):
            continue
        setup = setup_of(sub)
        for run in load_runs(sub, exclude):
            with open(run.path) as f:
                d = json.load(f)
            train = d.get("sl_train/loss") or []
            val = d.get("sl_val/loss") or d.get("sl_valid/loss") or []
            for i in range(max(len(train), len(val))):
                rows.append({
                    "setup": setup, "model": run.model, "dataset": run.dataset,
                    "seq_len": run.seq_len, "pred_len": run.pred_len,
                    "k": "" if run.k is None else run.k,
                    "channel": "" if run.channel is None else run.channel,
                    "seed": run.seed or "", "conf": int(run.conf), "epoch": i,
                    "train_loss": _num(train[i]) if i < len(train) else "",
                    "val_loss": _num(val[i]) if i < len(val) else "",
                })
    return rows


def _mean(vals: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def _stdev(vals: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return statistics.stdev(vals) if len(vals) > 1 else None


def _f(cell: str) -> Optional[float]:
    return float(cell) if cell not in ("", None) else None


def _mean_all(vals: list[Optional[float]]) -> Optional[float]:
    """The mean, or None unless every value is present."""
    return statistics.mean(vals) if vals and all(v is not None for v in vals) else None


def _stdev_all(vals: list[Optional[float]]) -> Optional[float]:
    """The sample stdev, or None unless every value is present (and n > 1)."""
    return (statistics.stdev(vals)
            if len(vals) > 1 and all(v is not None for v in vals) else None)


def complete_seeds(by_seed: dict) -> set:
    """The seeds of a per-channel cell that have every one of its bands.

    ``by_seed`` maps each seed to its runs.csv rows. A seed is complete when
    its bands are exactly the union over the cell's seeds, one run each: a
    seed half-way through a sweep, averaged over whichever bands finished, is
    a different quantity from a full seed. The one completeness rule behind
    cells.csv's per-channel seed sd (_seed_band_means), timing.csv
    (build_timing) and the paradigm comparison (paradigms.seed_values).
    """
    bands = {seed: [r["channel"] for r in rs] for seed, rs in by_seed.items()}
    union = set().union(*map(set, bands.values())) if bands else set()
    return {seed for seed, chans in bands.items()
            if len(chans) == len(union) and set(chans) == union}


def _seed_band_means(rs: list[dict], metric: str) -> list[float]:
    """Per-seed means over the bands of a per-channel cell, complete seeds only.

    A seed counts only if it is complete (complete_seeds) and has a value on
    every band.
    """
    by_seed: dict = defaultdict(list)
    for r in rs:
        by_seed[r["seed"]].append(r)
    means = []
    for seed in sorted(complete_seeds(by_seed)):
        vals = [_f(r[metric]) for r in by_seed[seed]]
        if all(v is not None for v in vals):
            means.append(statistics.mean(vals))
    return means


def build_cells(runs: list[dict]) -> list[dict]:
    """Mean each (setup, model, dataset, L, O, k, conf) cell over its runs.

    Matches extract_results.aggregate exactly -- a plain mean over every run in
    the group, ignoring missing values, with num_params taken from the first
    run that logged one -- so the tables can be reconciled against the CSVs.

    mse_std and mae_std are the seed standard deviation: of the runs of a joint
    cell, one per seed, and of the per-seed band means of a per-channel cell
    (_seed_band_means). Until 2026-09-24 a per-channel cell's mse_std was the
    spread over all of its band x seed runs, which measures how different the
    bands are, not how repeatable the cell is (ED2022 DLinear: 0.494 against a
    seed sd of 1.3e-4); that number is kept as mse_band_seed_sd.
    n_complete_seeds is the sd's sample size; mse, mae and n_seeds still
    count every run, a per-channel seed still training included.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["setup"], r["model"], r["dataset"], r["seq_len"],
                r["pred_len"], r["k"], r["conf"])].append(r)

    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        setup, model, dataset, seq_len, pred_len, k, conf = key
        seeds = {r["seed"] for r in rs if r["seed"]}
        channels = {r["channel"] for r in rs if r["channel"] != ""}
        get = lambda name: [_f(r.get(name, "")) for r in rs]
        if channels:
            spread = {m: _num(_stdev(get(m))) for m in ("mse", "mae")}
            std = {m: _num(_stdev(_seed_band_means(rs, m))) for m in ("mse", "mae")}
            n_complete = len(_seed_band_means(rs, "mse"))
        else:
            spread = {"mse": "", "mae": ""}
            std = {m: _num(_stdev(get(m))) for m in ("mse", "mae")}
            n_complete = len({r["seed"] for r in rs
                              if r["seed"] and _f(r["mse"]) is not None})
        sources = sorted({r.get("fixed_source", "") for r in rs} - {""})
        out.append({
            "setup": setup, "family": family_of(dataset),
            "model": model, "dataset": dataset,
            "seq_len": seq_len, "pred_len": pred_len, "k": k, "conf": conf,
            "mse": _num(_mean(get("mse"))), "mse_std": std["mse"],
            "mae": _num(_mean(get("mae"))), "mae_std": std["mae"],
            "mse_band_seed_sd": spread["mse"], "mae_band_seed_sd": spread["mae"],
            "val_loss": _num(_mean(get("val_loss"))),
            "running_time": _num(_mean(get("running_time"))),
            "per_epoch": _num(_mean(get("per_epoch"))),
            "num_params": next((r["num_params"] for r in rs if r["num_params"]), ""),
            "ic": _num(_mean(get("ic"))), "jc": _num(_mean(get("jc"))),
            "interval_width": _num(_mean(get("interval_width"))),
            "mse_fixed": _num(_mean_all(get("mse_fixed")) if sources else None),
            "mse_fixed_std": _num(_stdev_all(get("mse_fixed")) if sources else None),
            "mae_fixed": _num(_mean_all(get("mae_fixed")) if sources else None),
            "mae_fixed_std": _num(_stdev_all(get("mae_fixed")) if sources else None),
            "val_loss_fixed": _num(_mean_all(get("val_loss_fixed")) if sources else None),
            "ic_fixed": _num(_mean_all(get("ic_fixed")) if sources else None),
            "jc_fixed": _num(_mean_all(get("jc_fixed")) if sources else None),
            "interval_width_fixed": _num(_mean_all(get("interval_width_fixed"))
                                         if sources else None),
            "fixed_sources": ";".join(sources),
            "n_runs": len(rs),
            "n_seeds": len(seeds) or "",
            "n_complete_seeds": n_complete if seeds else "",
            "n_channels": len(channels) or "",
            "code_versions": ";".join(sorted({r["code_version"] or "?" for r in rs})),
            "code_equivalents": ";".join(sorted({r.get("code_equivalent", r["code_version"])
                                                 or "?" for r in rs})),
        })
    return out


def band_rows(setup: str, run: Run, names: Optional[list],
              lookback: Optional[list]) -> list[dict]:
    """bands.csv rows of one run (see BAND_FIELDS).

    ``names`` are the dataset's band labels and ``lookback`` its per-band
    lookback-mean MSE on the run's windows, each in channel order (None when
    unknown). A joint run's per-band vector must have one entry per band.
    """
    base = {
        "setup": setup, "family": family_of(run.dataset), "model": run.model,
        "dataset": run.dataset, "seq_len": run.seq_len, "pred_len": run.pred_len,
        "k": "" if run.k is None else run.k, "seed": run.seed or "",
        "conf": int(run.conf), "fixed_source": run.fixed_source,
    }

    def one(channel: int, mse, mae, mse_fixed) -> dict:
        row = {**base, "channel": channel,
               "band": names[channel] if names and channel < len(names) else "",
               "mse": _num(mse), "mae": _num(mae), "mse_fixed": _num(mse_fixed),
               "lookback_mse": (_num(float(lookback[channel]))
                                if lookback is not None and channel < len(lookback)
                                else "")}
        return {f: row[f] for f in BAND_FIELDS}

    if run.channel is not None:
        return [one(run.channel, run.mse, run.mae, None)]
    per = run.mse_per_channel or []
    if names and per and len(per) != len(names):
        raise ValueError(f"{run.path}: {len(per)} per-channel MSEs for "
                         f"{len(names)} bands of {run.dataset}")
    fixed = run.mse_per_channel_fixed
    if fixed is not None and len(fixed) != len(per):
        raise ValueError(f"{run.path}: {len(fixed)} fixed per-channel MSEs "
                         f"for {len(per)} batch ones")
    return [one(i, v, None, fixed[i] if fixed is not None else None)
            for i, v in enumerate(per)]


def build_bands(loaded: list[tuple[str, Run]]) -> list[dict]:
    """One row per (run, band) over the joint-channel corpora (BAND_FIELDS).

    Only datasets with named bands take part (baselines.band_names): the
    series tree's det/ptv/ptv2 are single aggregate series.
    """
    from .baselines import band_names, lookback_mean_mse

    rows = []
    for setup, run in loaded:
        names = band_names(run.dataset)
        if names is None:
            continue
        lookback = None
        if run.train_split is not None and run.val_split is not None:
            lookback = lookback_mean_mse(run.dataset, run.seq_len, run.pred_len,
                                         run.train_split, run.val_split)
        rows += band_rows(setup, run, names, lookback)
    rows.sort(key=lambda r: (r["setup"], r["model"], r["dataset"], r["seq_len"],
                             r["pred_len"], -1 if r["k"] == "" else r["k"],
                             r["seed"], r["channel"]))
    return rows


def build_timing(runs: list[dict], solo: Optional[dict]) -> list[dict]:
    """TIMING_FIELDS rows: the uncontended cost of each complete per-channel seed.

    ``solo`` is specs/timing.yaml's ``solo_pass`` block. Only cells it
    measured get rows, at its window, and only seeds that have every band of
    the cell (complete_seeds) and an epoch count on each.
    """
    if not solo or not solo.get("bands"):
        return []
    seq_len, pred_len = int(solo["seq_len"]), int(solo["pred_len"])
    measured: dict = defaultdict(list)
    for b in solo["bands"]:
        measured[(str(b["model"]), str(b["dataset"]))].append(b)

    pc: dict = defaultdict(lambda: defaultdict(list))
    mv: dict = {}
    for r in runs:
        if int(r["seq_len"]) != seq_len or int(r["pred_len"]) != pred_len:
            continue
        if r["setup"] == "perchannel" and r["seed"]:
            pc[(r["model"], r["dataset"])][r["seed"]].append(r)
        elif r["setup"] == "multivariate" and r["seed"] and r["running_time"]:
            mv[(r["model"], r["dataset"], r["seed"])] = float(r["running_time"])

    out = []
    for (model, dataset), bands in sorted(measured.items()):
        per_epoch = [float(b["seconds"]) / int(b["epochs"]) for b in bands]
        pooled = (sum(float(b["seconds"]) for b in bands)
                  / sum(int(b["epochs"]) for b in bands))
        contention = statistics.mean(
            (float(b["twin_seconds"]) / int(b["twin_epochs"])) / pe
            for b, pe in zip(bands, per_epoch))
        by_seed = pc.get((model, dataset), {})
        complete = complete_seeds(by_seed)
        for seed, rs in sorted(by_seed.items()):
            if seed not in complete or any(not r.get("epochs") for r in rs):
                continue
            epochs = sum(int(r["epochs"]) for r in rs)
            times = [_f(r["running_time"]) for r in rs]
            contended = sum(times) if all(t is not None for t in times) else None
            mv_t = mv.get((model, dataset, seed))
            total = epochs * pooled
            out.append({
                "family": family_of(dataset), "model": model, "dataset": dataset,
                "seq_len": seq_len, "pred_len": pred_len, "seed": seed,
                "n_bands": len(rs), "epochs": epochs, "solo_bands": len(bands),
                "solo_per_epoch": _num(pooled), "solo_total": _num(total),
                "solo_lo": _num(epochs * min(per_epoch)),
                "solo_hi": _num(epochs * max(per_epoch)),
                "contention_mean": _num(contention),
                "contended_total": _num(contended),
                # "?" for bands logged before concurrency was stamped.
                "concurrency": ";".join(sorted({str(r["concurrency"]) or "?" for r in rs})),
                "mv_running_time": _num(mv_t),
                "solo_x_mv": _num(total / mv_t if mv_t else None),
                "contended_x_mv": _num(contended / mv_t
                                       if mv_t and contended is not None else None),
            })
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _logs_commit(repo: Path) -> str:
    """The last commit that touched logs/ -- the only git fact worth recording.

    Recording the checked-out HEAD instead would rewrite this file on every
    unrelated commit, leaving `git status` permanently dirty after a rebuild,
    which is how a provenance file stops being read. This changes exactly when
    the inputs change.
    """
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "log", "-1", "--format=%H", "--", "logs"],
            capture_output=True, text=True, check=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_manifest(path: Path, repo: Path, runs: list[dict], counts: dict,
                   exclude: list[str]) -> None:
    """Record what went in, so a stale artifact can be spotted rather than trusted.

    Hashing every log.json would be slow and is not the point: what matters is
    the set of inputs and the numbers derived from them, so the digest runs
    over the sorted run paths plus their metric values.
    """
    digest = hashlib.sha256()
    for r in sorted(runs, key=lambda r: r["path"]):
        line = f"{r['path']}|{r['mse']}|{r['mae']}|{r['running_time']}"
        # A cluster run's fixed-membership numbers are inputs too (its log or
        # an eval_fixed.json that appears later); other runs keep the line
        # they always had.
        if r.get("fixed_source"):
            line += f"|{r['fixed_source']}|{r['mse_fixed']}|{r['mae_fixed']}"
        digest.update(f"{line}\n".encode())
    sources = {"stamped": 0, "attested": 0, "inferred": 0, "dirty": 0, "unstamped": 0}
    versions: dict = {}
    for r in runs:
        src = r.get("commit_source") or "unstamped"
        sources["stamped" if src == "log" else src] += 1
        if r.get("code_version"):
            v = versions.setdefault(r["code_version"], {"commits": set(), "runs": 0,
                                                        "equivalent_to": set()})
            v["commits"].add(r["git_commit"])
            v["runs"] += 1
            if r.get("code_equivalent", r["code_version"]) != r["code_version"]:
                v["equivalent_to"].add(r["code_equivalent"])
    equivalents: dict = {}
    for r in runs:
        if r.get("code_version"):
            e = equivalents.setdefault(r.get("code_equivalent") or r["code_version"],
                                       {"runs": 0, "versions": set()})
            e["runs"] += 1
            e["versions"].add(r["code_version"])

    def version_entry(v: dict) -> dict:
        entry = {"commits": sorted(v["commits"]), "runs": v["runs"]}
        if v["equivalent_to"]:
            entry["equivalent_to"] = ";".join(sorted(v["equivalent_to"]))
        return entry

    fixed_sources: dict = {}
    fixed_versions: dict = {}
    for r in runs:
        if r.get("fixed_source"):
            fixed_sources[r["fixed_source"]] = fixed_sources.get(r["fixed_source"], 0) + 1
        if r.get("fixed_source") not in (None, "", "none"):
            v = fixed_versions.setdefault(r.get("fixed_code_version") or "?",
                                          {"runs": 0, "sources": set()})
            v["runs"] += 1
            v["sources"].add(r["fixed_source"])
    payload = {
        "logs_commit": _logs_commit(repo),
        # The training-code versions behind the runs, each with the commit ids
        # that carry it. Several commits can share one version (they differ only
        # outside the code paths); one version is the goal -- contract C1 in
        # .personal/PLAN.md. A version an equivalent_code rule in
        # specs/provenance.yaml counts as another names it in equivalent_to.
        "code_versions": {fp: version_entry(v) for fp, v in sorted(versions.items())},
        # The same runs under those rules: each version they count as, with the
        # fingerprints it gathers. One entry here is what contract C1 asks for.
        "code_equivalents": {fp: {"runs": e["runs"], "versions": sorted(e["versions"])}
                             for fp, e in sorted(equivalents.items())},
        "unknown_code_runs": sum(1 for r in runs if not r.get("code_version")),
        # Where the cluster runs' fixed-membership metrics came from
        # (runlogs.FIXED_SOURCES), and the code versions that computed them
        # ("?" where that is unknown): a sidecar written later by other
        # re-score code shows up here as a second version.
        "fixed_sources": dict(sorted(fixed_sources.items())),
        "fixed_code_versions": {fp: {"runs": v["runs"], "sources": sorted(v["sources"])}
                                for fp, v in sorted(fixed_versions.items())},
        "provenance": sources,
        "n_runs": len(runs),
        "runs_digest": digest.hexdigest(),
        "counts": counts,
        "exclude": exclude,
        "setups": sorted({r["setup"] for r in runs}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
