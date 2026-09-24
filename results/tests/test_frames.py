"""The frames: the fixed cluster protocol.

It decides what a cluster number in frames/ means, and fails quietly: a cluster
run reading its batch-protocol numbers where the fixed ones exist publishes the
sensitivity row as the result. So it is pinned here on small synthetic logs
whose right answer is known.
"""

import json

import pytest

from emfres import frames
from emfres.runlogs import FIXED_SIDECAR, MARKER, load_runs


def _log(**extra):
    log = {
        "parameters/exp/model_id": "DLinear",
        "parameters/data/dataset": "TOY",
        "parameters/data/seq_len": 336,
        "parameters/data/pred_len": 96,
        "parameters/conf/conf": True,
        "sl_train/loss": [1.0, 0.9],
        "sl_val/loss": [0.8, 0.7],
        "sl_test/loss": [0.5],
        "sl_test/mae": [0.4],
        "sl_test/ic": [90.0],
        "sl_test/jc": [10.0],
        MARKER: [1.5],
    }
    log.update(extra)
    return log


def _write(root, rel, log):
    run = root / rel
    run.mkdir(parents=True, exist_ok=True)
    (run / "log.json").write_text(json.dumps(log))
    return run


# --------------------------------------------------------------- fixed protocol

BATCH = {"sl_test/mse_per_channel": [0.4, 0.6]}
FIXED = {"sl_val_fixed/loss": [0.61], "sl_test_fixed/loss": [0.45],
         "sl_test_fixed/mae": [0.35], "sl_test_fixed/mse_per_channel": [0.3, 0.6],
         "sl_test_fixed/ic": [91.0], "sl_test_fixed/jc": [11.0],
         "sl_test_fixed/interval_width_mean": [1.4]}
SIDECAR = {"sl_val_fixed/loss": [0.62], "sl_test_fixed/loss": [0.44],
           "sl_test_fixed/mae": [0.34], "sl_test_fixed/mse_per_channel": [0.28, 0.6],
           "sl_test_fixed/ic": [92.0], "sl_test_fixed/jc": [12.0],
           "sl_test_fixed/interval_width_mean": [1.3],
           "rescore": {"batch_check": {"passed": True}}}


def _cluster(root, k, seed, *, fixed=None, sidecar=None, **extra):
    log = _log(**BATCH, **{"parameters/cluster/enabled": True,
                           "parameters/cluster/k_clusters": k}, **(fixed or {}), **extra)
    run = _write(root, f"DLinear/toy/L336_O96_cp/k{k}/{seed}", log)
    if sidecar is not None:
        (run / FIXED_SIDECAR).write_text(json.dumps(sidecar))
    return run


def _by_seed(root, **kw):
    return {int(r.seed): r for r in load_runs(root, [], setup="cluster", **kw)}


def test_the_log_wins_then_the_sidecar_then_k1(tmp_path):
    _cluster(tmp_path, 3, 1, fixed=FIXED, sidecar=SIDECAR)   # both: the log wins
    _cluster(tmp_path, 3, 2, sidecar=SIDECAR)                # re-scored
    _cluster(tmp_path, 3, 3)                                 # neither, k > 1
    _cluster(tmp_path, 1, 4)                                 # neither, k = 1
    _cluster(tmp_path, 1, 5, sidecar=SIDECAR)                # k = 1, re-scored
    runs = _by_seed(tmp_path)
    assert {s: r.fixed_source for s, r in runs.items()} == {
        1: "log", 2: "sidecar", 3: "none", 4: "k1", 5: "sidecar"}

    log = runs[1]
    assert (log.mse_fixed, log.mae_fixed, log.val_loss_fixed) == (0.45, 0.35, 0.61)
    assert (log.ic_fixed, log.jc_fixed, log.interval_width_fixed) == (91.0, 11.0, 1.4)
    assert log.mse_per_channel_fixed == [0.3, 0.6]
    side = runs[2]
    assert (side.mse_fixed, side.mae_fixed, side.val_loss_fixed) == (0.44, 0.34, 0.62)
    assert side.mse_per_channel_fixed == [0.28, 0.6]

    # The batch protocol is untouched whatever the fixed source.
    assert all((r.mse, r.mae, r.ic, r.jc, r.interval_width, r.val_loss)
               == (0.5, 0.4, 90.0, 10.0, 1.5, 0.7) for r in runs.values())

    none = runs[3]
    assert none.mse_fixed is None and none.mse_per_channel_fixed is None
    # k = 1: one cluster, nothing to fix -- the fixed metrics ARE the batch ones.
    one = runs[4]
    assert ((one.mse_fixed, one.mae_fixed, one.val_loss_fixed, one.ic_fixed,
             one.jc_fixed, one.interval_width_fixed, one.mse_per_channel_fixed)
            == (0.5, 0.4, 0.7, 90.0, 10.0, 1.5, [0.4, 0.6]))


def test_a_failed_or_unreadable_sidecar_is_not_used(tmp_path, capsys):
    failed = {**SIDECAR, "rescore": {"batch_check": {"passed": True},
                                     "fixed_check": {"passed": False}}}
    _cluster(tmp_path, 3, 1, sidecar=failed)
    _cluster(tmp_path, 3, 2, sidecar={k: v for k, v in SIDECAR.items() if k != "rescore"})
    run = _cluster(tmp_path, 3, 3)
    (run / FIXED_SIDECAR).write_text('{"sl_test_fixed/loss": [0.4')
    runs = _by_seed(tmp_path)
    assert {r.fixed_source for r in runs.values()} == {"none"}
    assert all(r.mse_fixed is None for r in runs.values())
    err = capsys.readouterr().err
    assert err.count("does not record a passed re-score") == 2
    assert "unreadable" in err


def test_a_non_cluster_run_has_no_fixed_source(tmp_path):
    _write(tmp_path, "DLinear/toy/L336_O96_cp/1", _log(**FIXED))
    (run,) = load_runs(tmp_path, [], setup="multivariate")
    assert run.fixed_source == "" and run.mse_fixed is None


def test_unfinished_runs_are_skipped(tmp_path, capsys):
    _write(tmp_path, "DLinear/toy/L336_O96_cp/1", _log())
    unfinished = _log()
    del unfinished[MARKER]
    _write(tmp_path, "DLinear/toy/L336_O96_cp/2", unfinished)
    torn = _write(tmp_path, "DLinear/toy/L336_O96_cp/3", _log())
    (torn / "log.json").write_text('{"parameters/exp/model_id": "DLi')
    runs = load_runs(tmp_path, [], setup="multivariate")
    assert [r.seed for r in runs] == ["1"]
    assert "skipped 2 unfinished run(s)" in capsys.readouterr().err


def test_runs_csv_carries_fixed_beside_batch(tmp_path):
    _cluster(tmp_path, 3, 1, fixed=FIXED)
    _cluster(tmp_path, 3, 2, fixed=FIXED)
    (run,) = [r for r in load_runs(tmp_path, [], setup="cluster") if r.seed == "1"]
    row = frames._row(run, "cluster", tmp_path)
    assert list(row) == frames.RUN_FIELDS
    assert (row["mse"], row["mse_fixed"], row["fixed_source"]) == ("0.5", "0.45", "log")
    assert (row["val_loss"], row["val_loss_fixed"]) == ("0.7", "0.61")


# --------------------------------------------------------------- cells

def _row(setup, seed, mse, *, channel="", k="", mae=None, **extra):
    row = {f: "" for f in frames.RUN_FIELDS}
    row.update(setup=setup, family="emf", model="DLinear", dataset="ed2022",
               seq_len=336, pred_len=96, k=k, channel=channel, seed=str(seed),
               conf=1, mse=repr(mse), mae=repr(mse / 2 if mae is None else mae),
               code_version="aaaaaaaaaaaa", code_equivalent="aaaaaaaaaaaa")
    row.update(extra)
    return row


def _cells(rows):
    return {(c["setup"], c["k"]): c for c in frames.build_cells(rows)}


def test_a_cluster_cell_has_fixed_means_only_when_every_run_has_one():
    full = [_row("cluster", s, 0.5, k=3, mse_fixed=repr(v), mae_fixed=repr(v / 2),
                 fixed_source=src)
            for s, v, src in ((1, 0.40, "log"), (2, 0.42, "sidecar"), (3, 0.44, "sidecar"))]
    cell = _cells(full)[("cluster", 3)]
    assert float(cell["mse_fixed"]) == pytest.approx(0.42)
    assert float(cell["mse_fixed_std"]) == pytest.approx(0.02)
    assert cell["fixed_sources"] == "log;sidecar"
    assert float(cell["mse"]) == pytest.approx(0.5)          # batch, as before

    partial = full[:2] + [_row("cluster", 3, 0.5, k=3, fixed_source="none")]
    cell = _cells(partial)[("cluster", 3)]
    assert cell["mse_fixed"] == "" and cell["mae_fixed"] == ""
    assert cell["fixed_sources"] == "log;none;sidecar"
