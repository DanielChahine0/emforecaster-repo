"""The one reader of a logs/ tree: log.json in, Run records out.

Nothing else in results/ opens a log file. Keeping the schema knowledge here
means the path conventions forecasting/grid.py stamps -- the ``ch<N>`` band
directories, the ``k<N>`` cluster tag, the numeric seed directory -- are
described once.

Moved verbatim out of extract_results.py (Phase 1 of the results/ restructure).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .vocab import parse_duration

_CHANNEL_RE = re.compile(r"^ch(\d+)$")
# The run-dir tag forecasting/grid.py stamps on a cluster cell (exp.id = "k3").
_K_RE = re.compile(r"^k(\d+)$")

# The last key a finished run logs (forecasting/grid.py MARKER, the grids'
# resume check): without it the run is still training, or crashed before its
# test, and is not a result.
MARKER = "sl_test/interval_width_mean"

# The sidecar forecasting/ccm_rescore.py writes into a cluster run trained
# before the fixed-membership evaluation: sl_val_fixed/* and sl_test_fixed/*
# under log.json's key names and list format. log.json is never edited.
FIXED_SIDECAR = "eval_fixed.json"

# Where a cluster run's fixed-membership metrics come from. "log": the run
# logged them itself (trained with the fixed evaluation); "sidecar": its
# eval_fixed.json re-score; "k1": neither, but k = 1 -- one cluster, so the
# membership has nothing to fix and the fixed metrics are the batch ones by
# construction; "none": a k > 1 run with neither. Blank on non-cluster runs.
FIXED_SOURCES = ("log", "sidecar", "k1", "none")


def parse_running_time(raw: Optional[str]) -> Optional[float]:
    """Strict read of a logged running_time; raises on anything unreadable."""
    return parse_duration(raw, strict=True)


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
    # Number of channel clusters, or None for an ordinary (non-cluster) run.
    k: Optional[int]
    seed: Optional[str]
    path: Path
    # Stamped by Experiment.init_logger since 2026-09-19 (forecasting/provenance.py);
    # None on every run logged before that.
    git_commit: Optional[str] = None
    concurrency: Optional[int] = None
    # Where git_commit came from: "log" (stamped, clean tree), "attested" (stamped
    # on a dirty tree, resolved by a specs/provenance.yaml rule), "dirty"
    # (stamped on a dirty tree, unresolved), "inferred" (unstamped, dated by a
    # rule), or "" (none of these).
    commit_source: str = ""
    # Whether the tree had uncommitted training-path changes when the run
    # started. None for runs logged before stamping.
    git_dirty: Optional[bool] = None
    # The commit whose training code the run actually used, when that is known:
    # the stamp itself on a clean tree, the attested or inferred commit
    # otherwise. None when it cannot be established.
    code_commit: Optional[str] = None
    # The split the run was scored on; the lookback-mean baseline has to use
    # the same one (VoIP runs use 0.6/0.2, the EMF jobs 0.7/0.1).
    train_split: Optional[float] = None
    val_split: Optional[float] = None
    # Best (lowest) validation loss over the run's epochs -- the quantity early
    # stopping selected on. What a model-selection choice made *without* the
    # test split (e.g. a cluster count) has to be made from. None for the
    # fit-once models, which have no epoch loop.
    val_loss: Optional[float] = None
    # sl_test/mse_per_channel: the test MSE of every channel, in the loader's
    # channel order. A per-channel run logs its one band here.
    mse_per_channel: Optional[list] = None
    # Cluster runs only: the same metrics under the fixed train-mean membership
    # (forecasting/multivariate/models/ccm.py), and where they came from (see
    # FIXED_SOURCES). mse_fixed is sl_test_fixed/loss, as mse is sl_test/loss;
    # val_loss_fixed is sl_val_fixed/loss at the restored best checkpoint.
    mse_fixed: Optional[float] = None
    mae_fixed: Optional[float] = None
    val_loss_fixed: Optional[float] = None
    ic_fixed: Optional[float] = None
    jc_fixed: Optional[float] = None
    interval_width_fixed: Optional[float] = None
    mse_per_channel_fixed: Optional[list] = None
    fixed_source: str = ""


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


def excluded(log_path: Path, exclude: list[str]) -> bool:
    """True if any --exclude pattern names a directory on this run's path.

    Matching is per path component, not by substring: a plain substring test
    made ``--exclude ch1`` swallow ch10 through ch19 as well. A pattern that
    contains a separator still matches as a substring, so a path fragment
    like ``ed2022/L336_O96_cp`` keeps working.
    """
    if not exclude:
        return False
    parts = set(log_path.parts)
    return any(
        (pat in str(log_path)) if ("/" in pat or "\\" in pat) else (pat in parts)
        for pat in exclude
    )


def _run_time(exp_time) -> Optional[datetime]:
    try:
        return datetime.strptime(str(exp_time), "%Y-%m-%d_%H-%M-%S")
    except (TypeError, ValueError):
        return None


def _in_window(rule: dict, when: datetime) -> bool:
    """Whether ``when`` falls in the rule's half-open [from, to) window."""
    try:
        lo = datetime.fromisoformat(str(rule["from"]))
        hi = datetime.fromisoformat(str(rule["to"]))
    except (KeyError, ValueError):
        return False
    return lo <= when < hi


def infer_commit(rules: list[dict], setup: Optional[str], model: str,
                 exp_time: Optional[str]) -> Optional[str]:
    """The commit a specs/provenance.yaml rule assigns to an unstamped run.

    A rule matches on the frames setup slug, an optional model list, and a
    half-open exp/time window [from, to); the first match wins. None when no
    rule applies or the run has no parseable parameters/exp/time.
    """
    when = _run_time(exp_time) if rules else None
    if when is None:
        return None
    for rule in rules:
        if rule.get("setup") not in (None, setup):
            continue
        models = rule.get("models")
        if models and model not in models:
            continue
        if _in_window(rule, when):
            return str(rule["commit"])
    return None


def attest_commit(rules: list[dict], stamped: str,
                  exp_time: Optional[str]) -> Optional[str]:
    """The commit whose code a dirty-tree run used, if a rule vouches for it.

    A run stamped on a dirty tree records the commit it sat on, not the code it
    ran, so its code is unknown unless something else establishes it. An
    ``attested`` rule in specs/provenance.yaml does that for a window of runs:
    it names the stamped commit (a prefix is enough), the [from, to) window,
    and the commit whose training code those runs are shown to have used,
    with the evidence written next to it.
    """
    when = _run_time(exp_time) if rules else None
    if when is None or not stamped:
        return None
    for rule in rules:
        prefix = str(rule.get("stamped") or "")
        if not prefix or not stamped.startswith(prefix):
            continue
        if _in_window(rule, when):
            return str(rule["code"])
    return None


def valid_equivalence(rule) -> bool:
    """Whether an ``equivalent_code`` rule is usable: from, to and evidence.

    A rule with no evidence written next to it vouches for nothing -- the
    whole point of the rule kind is that merging two code versions is a
    claim somebody checked.
    """
    if not isinstance(rule, dict):
        return False
    evidence = rule.get("evidence")
    if isinstance(evidence, str):
        evidence = [evidence]
    return (bool(str(rule.get("from") or "").strip())
            and bool(str(rule.get("to") or "").strip())
            and bool(evidence) and all(str(e).strip() for e in evidence))


def equivalent_code(rules: list[dict], fingerprint: str) -> str:
    """The code version a training-code fingerprint is counted as.

    An ``equivalent_code`` rule in specs/provenance.yaml maps one fingerprint
    (frames.code_fingerprint) onto another whose runs it is shown to
    reproduce, with the evidence beside it: the code changed on disk -- a new
    fingerprint -- but not in anything a run computes. Rules chain (a -> b,
    b -> c counts a as c); a cycle stops where it would repeat. Fingerprints
    match exactly, and a rule without evidence is ignored (valid_equivalence).
    A fingerprint no rule names is its own version; "" stays "".
    """
    table = {str(r["from"]).strip(): str(r["to"]).strip()
             for r in rules or [] if valid_equivalence(r)}
    seen = {fingerprint}
    while fingerprint in table and table[fingerprint] not in seen:
        fingerprint = table[fingerprint]
        seen.add(fingerprint)
    return fingerprint


def _first(value):
    """The scalar a log.json list holds (a one-shot test metric), else None."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _fixed_block(src: dict, conf: bool) -> dict:
    """The fixed-membership fields of a log.json or eval_fixed.json dict."""
    block = {
        "mse_fixed": _first(src.get("sl_test_fixed/loss")),
        "mae_fixed": _first(src.get("sl_test_fixed/mae")),
        "val_loss_fixed": _first(src.get("sl_val_fixed/loss")),
        "mse_per_channel_fixed": src.get("sl_test_fixed/mse_per_channel") or None,
    }
    if conf:
        block.update(ic_fixed=_first(src.get("sl_test_fixed/ic")),
                     jc_fixed=_first(src.get("sl_test_fixed/jc")),
                     interval_width_fixed=_first(src.get("sl_test_fixed/interval_width_mean")))
    return block


def _sidecar_passed(sidecar: dict) -> bool:
    """Whether a re-score's own self-checks passed (ccm_rescore writes only then)."""
    rescore = sidecar.get("rescore") or {}
    checks = [rescore.get(name) for name in ("batch_check", "fixed_check")
              if name in rescore]
    return bool(checks) and all(isinstance(c, dict) and c.get("passed") for c in checks)


def fixed_metrics(d: dict, run_dir: Path, k: Optional[int], conf: bool,
                  batch: dict) -> tuple[dict, str]:
    """(fields, source) of a cluster run's fixed-membership metrics.

    log.json's own sl_test_fixed/* win: those were computed on the device the
    run trained on. Otherwise the run dir's eval_fixed.json, if its re-score
    passed its self-checks. Otherwise, at k = 1, the batch metrics themselves:
    with one cluster every channel is in it whatever the batch, so the two
    protocols are the same forecast (``batch`` holds the run's batch values
    under the fixed field names). A k > 1 run with neither has no fixed
    metrics ("none").
    """
    if any(key.startswith("sl_test_fixed/") for key in d):
        return _fixed_block(d, conf), "log"
    path = run_dir / FIXED_SIDECAR
    if path.is_file():
        try:
            with open(path) as f:
                sidecar = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: unreadable {path} ({exc}); ignoring it", file=sys.stderr)
        else:
            if _sidecar_passed(sidecar):
                return _fixed_block(sidecar, conf), "sidecar"
            print(f"WARNING: {path} does not record a passed re-score; ignoring it",
                  file=sys.stderr)
    if k == 1:
        return dict(batch), "k1"
    return {}, "none"


def load_runs(logs_root: Path, exclude: list[str], *, setup: Optional[str] = None,
              inferred: Optional[list[dict]] = None,
              attested: Optional[list[dict]] = None) -> list[Run]:
    """A Run per finished log.json under logs_root.

    A log without MARKER, or one that does not parse (log.json is written in
    one go at the end of a run, but a reader can catch the write), is a run
    still training or one that crashed: it is skipped and counted in a NOTE,
    never read as a result.
    """
    runs = []
    incomplete = []
    for log_path in sorted(logs_root.rglob("log.json")):
        if excluded(log_path, exclude):
            continue
        try:
            with open(log_path) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            incomplete.append(log_path)
            continue
        if MARKER not in d:
            incomplete.append(log_path)
            continue

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

        channel = k = None
        for part in log_path.parts:
            m = _CHANNEL_RE.fullmatch(part)
            if m:
                channel = int(m.group(1))
            m = _K_RE.fullmatch(part)
            if m:
                k = int(m.group(1))
        # The logged value wins over the path tag; k_clusters is only meaningful
        # when cluster.enabled, since it defaults to 1 on every ordinary run.
        if d.get("parameters/cluster/enabled"):
            k = d.get("parameters/cluster/k_clusters", k)
        elif k is not None:
            print(f"WARNING: {log_path} sits under a 'k{k}' directory but "
                  f"parameters/cluster/enabled is not set; treating it as k={k}", file=sys.stderr)

        seed = log_path.parent.name if log_path.parent.name.isdigit() else None

        exp_time = d.get("parameters/exp/time")
        git_commit = d.get("parameters/exp/git_commit") or None
        git_dirty = d.get("parameters/exp/git_dirty")
        git_dirty = bool(git_dirty) if git_dirty is not None else None
        code_commit, commit_source = None, ""
        if git_commit and not git_dirty:
            code_commit, commit_source = git_commit, "log"
        elif git_commit:
            code_commit = attest_commit(attested or [], git_commit, exp_time)
            commit_source = "attested" if code_commit else "dirty"
        else:
            hit = infer_commit(inferred or [], setup, model, exp_time)
            if hit:
                git_commit = code_commit = hit
                commit_source = "inferred"
        concurrency = d.get("parameters/exp/concurrency")
        concurrency = int(concurrency) if concurrency not in (None, "") else None
        val_curve = d.get("sl_val/loss") or d.get("sl_valid/loss")
        val_loss = (min(v for v in val_curve if v is not None)
                    if isinstance(val_curve, list) and val_curve else None)

        mse_per_channel = d.get("sl_test/mse_per_channel") or None

        fixed, fixed_source = {}, ""
        if d.get("parameters/cluster/enabled"):
            batch = {"mse_fixed": mse, "mae_fixed": mae, "val_loss_fixed": val_loss,
                     "mse_per_channel_fixed": mse_per_channel}
            if conf:
                batch.update(ic_fixed=ic, jc_fixed=jc, interval_width_fixed=interval_width)
            fixed, fixed_source = fixed_metrics(d, log_path.parent, k, conf, batch)

        runs.append(Run(model, dataset, seq_len, pred_len, conf, mse, mae,
                         running_time, per_epoch, num_params, ic, jc, interval_width,
                         channel, k, seed, log_path,
                         git_commit=git_commit, concurrency=concurrency,
                         commit_source=commit_source, val_loss=val_loss,
                         git_dirty=git_dirty, code_commit=code_commit,
                         train_split=d.get("parameters/data/train_split"),
                         val_split=d.get("parameters/data/val_split"),
                         mse_per_channel=mse_per_channel,
                         fixed_source=fixed_source, **fixed))
    if incomplete:
        print(f"NOTE: skipped {len(incomplete)} unfinished run(s) under {logs_root} "
              f"(no {MARKER}: still training, or crashed), e.g. {incomplete[0]}",
              file=sys.stderr)
    return runs
