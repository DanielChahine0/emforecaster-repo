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
    "running_time", "per_epoch", "num_params",
    "ic", "jc", "interval_width",
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
    "mse", "mse_std", "mae", "mae_std", "running_time", "per_epoch",
    "num_params", "ic", "jc", "interval_width",
    "n_runs", "n_seeds", "n_channels",
    # Distinct training-code versions behind the cell's runs, ';'-joined, with
    # "?" for any run whose code cannot be established; more than one means
    # the cell mixes code versions and is not a clean seed mean --
    # unless code_equivalents, the same under the equivalent_code rules,
    # shows one.
    "code_versions", "code_equivalents",
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
        "num_params": "" if run.num_params is None else run.num_params,
        "ic": _num(run.ic),
        "jc": _num(run.jc),
        "interval_width": _num(run.interval_width),
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


def build_runs(logs_dir: Path, repo: Path, exclude: list[str],
               inferred: Optional[list[dict]] = None,
               attested: Optional[list[dict]] = None,
               equivalent: Optional[list[dict]] = None) -> list[dict]:
    """One row per log.json across every setup under ``logs_dir``.

    ``inferred``, ``attested`` and ``equivalent`` are the rule lists from
    specs/provenance.yaml: the first dates runs logged before provenance
    stamping, the second resolves runs stamped on a dirty tree (see
    runlogs.infer_commit / attest_commit), the third counts one training-code
    version as another (runlogs.equivalent_code).
    """
    rows = []
    for sub in sorted(p for p in logs_dir.iterdir() if p.is_dir()):
        if is_archive(sub):
            continue
        setup = setup_of(sub)
        for run in load_runs(sub, exclude, setup=setup, inferred=inferred,
                             attested=attested):
            rows.append(_row(run, setup, repo, equivalent))
    return rows


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


def build_cells(runs: list[dict]) -> list[dict]:
    """Mean each (setup, model, dataset, L, O, k, conf) cell over its runs.

    Matches extract_results.aggregate exactly -- a plain mean over every run in
    the group, ignoring missing values, with num_params taken from the first
    run that logged one -- so the tables can be reconciled against the CSVs.
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
        get = lambda name: [_f(r[name]) for r in rs]
        out.append({
            "setup": setup, "family": family_of(dataset),
            "model": model, "dataset": dataset,
            "seq_len": seq_len, "pred_len": pred_len, "k": k, "conf": conf,
            "mse": _num(_mean(get("mse"))), "mse_std": _num(_stdev(get("mse"))),
            "mae": _num(_mean(get("mae"))), "mae_std": _num(_stdev(get("mae"))),
            "running_time": _num(_mean(get("running_time"))),
            "per_epoch": _num(_mean(get("per_epoch"))),
            "num_params": next((r["num_params"] for r in rs if r["num_params"]), ""),
            "ic": _num(_mean(get("ic"))), "jc": _num(_mean(get("jc"))),
            "interval_width": _num(_mean(get("interval_width"))),
            "n_runs": len(rs),
            "n_seeds": len(seeds) or "",
            "n_channels": len(channels) or "",
            "code_versions": ";".join(sorted({r["code_version"] or "?" for r in rs})),
            "code_equivalents": ";".join(sorted({r.get("code_equivalent", r["code_version"])
                                                 or "?" for r in rs})),
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
        digest.update(f"{r['path']}|{r['mse']}|{r['mae']}|{r['running_time']}\n".encode())
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
        "provenance": sources,
        "n_runs": len(runs),
        "runs_digest": digest.hexdigest(),
        "counts": counts,
        "exclude": exclude,
        "setups": sorted({r["setup"] for r in runs}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
