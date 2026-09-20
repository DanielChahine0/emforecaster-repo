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

from .runlogs import Run, load_runs

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

RUN_FIELDS = [
    "setup", "model", "dataset", "seq_len", "pred_len", "k", "channel", "seed",
    "conf", "mse", "mae", "running_time", "per_epoch", "num_params",
    "ic", "jc", "interval_width",
    # Provenance (forecasting/provenance.py): the commit that produced the run
    # and how many trainings shared the GPU. Blank on runs logged before
    # 2026-09-19.
    "git_commit", "commit_source", "concurrency",
    "source", "path",
]

CELL_FIELDS = [
    "setup", "model", "dataset", "seq_len", "pred_len", "k", "conf",
    "mse", "mse_std", "mae", "mae_std", "running_time", "per_epoch",
    "num_params", "ic", "jc", "interval_width",
    "n_runs", "n_seeds", "n_channels",
    # Distinct commits behind the cell's runs, ';'-joined; more than one means
    # the cell mixes code versions and is not a clean seed mean.
    "commits",
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


def _row(run: Run, setup: str, repo: Path) -> dict:
    return {
        "setup": setup,
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
        "running_time": _num(run.running_time),
        "per_epoch": _num(run.per_epoch),
        "num_params": "" if run.num_params is None else run.num_params,
        "ic": _num(run.ic),
        "jc": _num(run.jc),
        "interval_width": _num(run.interval_width),
        "git_commit": run.git_commit or "",
        "commit_source": run.commit_source,
        "concurrency": "" if run.concurrency is None else run.concurrency,
        "source": "log",
        "path": run.path.relative_to(repo).as_posix(),
    }


def _num(v: Optional[float]) -> str:
    """Full repr, so a round-trip through CSV loses nothing."""
    return "" if v is None else repr(float(v))


def build_runs(logs_dir: Path, repo: Path, exclude: list[str],
               inferred: Optional[list[dict]] = None) -> list[dict]:
    """One row per log.json across every setup under ``logs_dir``.

    ``inferred`` is the rule list from specs/provenance.yaml, used to assign a
    commit to runs logged before provenance stamping (see runlogs.infer_commit).
    """
    rows = []
    for sub in sorted(p for p in logs_dir.iterdir() if p.is_dir()):
        if is_archive(sub):
            continue
        setup = setup_of(sub)
        for run in load_runs(sub, exclude, setup=setup, inferred=inferred):
            rows.append(_row(run, setup, repo))
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
            "setup": setup, "model": model, "dataset": dataset,
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
            "commits": ";".join(sorted({r["git_commit"] for r in rs if r["git_commit"]})),
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
    commits = sorted({r["git_commit"] for r in runs if r.get("git_commit")})
    sources = {"stamped": 0, "inferred": 0, "unstamped": 0}
    for r in runs:
        sources[{"log": "stamped", "inferred": "inferred"}.get(r.get("commit_source"), "unstamped")] += 1
    payload = {
        "logs_commit": _logs_commit(repo),
        # The code commits behind the runs: stamped by the run itself (since
        # 2026-09-19) or inferred from specs/provenance.yaml for older runs.
        # One entry is the goal: contract C1 in .personal/PLAN.md.
        "code_commits": commits,
        "provenance": sources,
        "n_runs": len(runs),
        "runs_digest": digest.hexdigest(),
        "counts": counts,
        "exclude": exclude,
        "setups": sorted({r["setup"] for r in runs}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
