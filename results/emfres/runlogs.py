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
from pathlib import Path
from typing import Optional

from .vocab import parse_duration

_CHANNEL_RE = re.compile(r"^ch(\d+)$")
# The run-dir tag forecasting/grid.py stamps on a cluster cell (exp.id = "k3").
_K_RE = re.compile(r"^k(\d+)$")


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


def load_runs(logs_root: Path, exclude: list[str]) -> list[Run]:
    runs = []
    for log_path in sorted(logs_root.rglob("log.json")):
        if excluded(log_path, exclude):
            continue
        with open(log_path) as f:
            d = json.load(f)

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

        runs.append(Run(model, dataset, seq_len, pred_len, conf, mse, mae,
                         running_time, per_epoch, num_params, ic, jc, interval_width,
                         channel, k, seed, log_path))
    return runs
