"""The frames: the fixed cluster protocol, seed spread, bands and timing.

Each of these decides what a number in frames/ means, and each fails quietly:
a cluster run reading its batch-protocol numbers where the fixed ones exist
publishes the sensitivity row as the result; a per-channel spread over bands
read as seed noise makes every per-channel cell look unrepeatable; a band
vector off by one channel puts one frequency's error on another. So each is
pinned here on small synthetic logs whose right answer is known.
"""

import hashlib
import json
import statistics

import pytest

from emfres import baselines, cli, frames
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
           "rescore": {"batch_check": {"passed": True}, "git_commit": "r" * 40,
                       "git_dirty": False}}


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


def test_the_fixed_numbers_carry_the_commit_that_computed_them(tmp_path):
    stamp = {"parameters/exp/git_commit": "c" * 40, "parameters/exp/git_dirty": False}
    _cluster(tmp_path, 3, 1, fixed=FIXED, **stamp)              # log: the run's code
    _cluster(tmp_path, 3, 2, sidecar=SIDECAR, **stamp)          # sidecar: the re-score's
    _cluster(tmp_path, 1, 3, **stamp)                           # k1: the run's code
    _cluster(tmp_path, 3, 4, **stamp)                           # none
    dirty = {**SIDECAR, "rescore": {**SIDECAR["rescore"], "git_dirty": True}}
    _cluster(tmp_path, 3, 5, sidecar=dirty, **stamp)            # a dirty re-score
    runs = _by_seed(tmp_path)
    assert {s: (r.fixed_source, r.fixed_commit) for s, r in runs.items()} == {
        1: ("log", "c" * 40), 2: ("sidecar", "r" * 40), 3: ("k1", "c" * 40),
        4: ("none", None), 5: ("sidecar", None)}


def test_a_sidecar_of_another_checkpoint_is_not_used(tmp_path, capsys):
    def with_checkpoint(seed, weights, recorded):
        sidecar = {**SIDECAR, "rescore": {**SIDECAR["rescore"],
                                          "checkpoint": "supervised.pth",
                                          "checkpoint_sha256": recorded}}
        run = _cluster(tmp_path, 3, seed, sidecar=sidecar)
        if weights is not None:
            (run / "supervised.pth").write_bytes(weights)

    same = hashlib.sha256(b"weights").hexdigest()
    with_checkpoint(1, b"weights", same)             # the checkpoint it scored
    with_checkpoint(2, b"retrained", same)           # retrained in place since
    with_checkpoint(3, None, same)                   # checkpoint gone: cannot check
    runs = _by_seed(tmp_path)
    assert {s: r.fixed_source for s, r in runs.items()} == {
        1: "sidecar", 2: "none", 3: "sidecar"}
    assert "scored another checkpoint than the run's own" in capsys.readouterr().err


def test_runs_csv_and_the_manifest_name_the_fixed_code(tmp_path, monkeypatch):
    monkeypatch.setattr(frames, "code_fingerprint",
                        lambda repo, commit: commit[:3] if commit else "")
    stamp = {"parameters/exp/git_commit": "c" * 40, "parameters/exp/git_dirty": False}
    _cluster(tmp_path, 3, 1, fixed=FIXED, **stamp)
    _cluster(tmp_path, 3, 2, sidecar=SIDECAR, **stamp)
    _cluster(tmp_path, 3, 3, **stamp)
    _write(tmp_path, "DLinear/toy/L336_O96_cp/9", _log(**stamp))
    rows = {r.seed: frames._row(r, "cluster" if r.k else "multivariate", tmp_path)
            for r in load_runs(tmp_path, [])}
    assert {s: (r["fixed_source"], r["fixed_code_version"], r["code_version"])
            for s, r in rows.items()} == {
        "1": ("log", "ccc", "ccc"), "2": ("sidecar", "rrr", "ccc"),
        "3": ("none", "", "ccc"), "9": ("", "", "ccc")}
    frames.write_manifest(tmp_path / "MANIFEST.json", tmp_path, list(rows.values()),
                          {}, [])
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert manifest["fixed_code_versions"] == {
        "ccc": {"runs": 1, "sources": ["log"]}, "rrr": {"runs": 1, "sources": ["sidecar"]}}


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
    assert row["epochs"] == 2


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


def test_per_channel_std_is_the_seed_sd_of_band_means():
    # Two bands far apart (the band spread) and seeds that barely move.
    values = {1: (0.10, 2.00), 2: (0.11, 2.01), 3: (0.12, 2.05)}
    rows = [_row("perchannel", s, v, channel=ch)
            for s, pair in values.items() for ch, v in enumerate(pair)]
    cell = _cells(rows)[("perchannel", "")]
    means = [statistics.mean(pair) for pair in values.values()]
    assert float(cell["mse_std"]) == pytest.approx(statistics.stdev(means))
    assert float(cell["mae_std"]) == pytest.approx(statistics.stdev(means) / 2)
    # The old spread over every band x seed is kept, under its own name.
    spread = statistics.stdev(v for pair in values.values() for v in pair)
    assert float(cell["mse_band_seed_sd"]) == pytest.approx(spread)
    assert float(cell["mse_band_seed_sd"]) > 10 * float(cell["mse_std"])
    # The mean itself keeps its meaning: over every band x seed run.
    assert float(cell["mse"]) == pytest.approx(statistics.mean(means))


def test_a_seed_missing_a_band_is_left_out_of_the_seed_sd():
    rows = [_row("perchannel", s, v, channel=ch)
            for s, pair in {1: (0.1, 2.0), 2: (0.2, 2.2)}.items()
            for ch, v in enumerate(pair)]
    rows.append(_row("perchannel", 3, 9.0, channel=0))      # half a seed
    cell = _cells(rows)[("perchannel", "")]
    assert float(cell["mse_std"]) == pytest.approx(statistics.stdev([1.05, 1.2]))
    assert cell["n_seeds"] == 3
    # One complete seed has no seed spread at all.
    one = _cells([_row("perchannel", 1, v, channel=ch) for ch, v in enumerate((0.1, 2.0))])
    assert one[("perchannel", "")]["mse_std"] == ""


def test_a_joint_cell_keeps_its_std_and_has_no_band_spread():
    rows = [_row("multivariate", s, v) for s, v in ((1, 0.50), (2, 0.52), (3, 0.51))]
    cell = _cells(rows)[("multivariate", "")]
    assert float(cell["mse_std"]) == pytest.approx(statistics.stdev([0.50, 0.52, 0.51]))
    assert cell["mse_band_seed_sd"] == "" and cell["mae_band_seed_sd"] == ""
    assert cell["fixed_sources"] == "" and cell["mse_fixed"] == ""


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


# --------------------------------------------------------------- bands

def test_band_labels_follow_the_loader_channel_order():
    ed = baselines.band_names("ed2022")
    assert len(ed) == 27 and ed[0] == "796.0" and ed[26] == "3760.0"
    assert baselines.band_names("uh2022")[21] == "3568.0"
    emf = baselines.band_names("emf_2023_university_5_with_ptv")
    assert len(emf) == 29 and emf[24] == "3547.5" and "PTV" not in emf
    assert baselines.band_names("voip_mobile") == ["MOS", "BW", "RTT", "JIT", "DJB", "SNR"]
    assert baselines.band_names("det") is None
    assert not any(n.startswith("EMF_") for n in ed + emf)


@pytest.mark.parametrize("dataset,split", [("ed2022", (0.7, 0.1)),
                                           ("uh2022", (0.7, 0.1)),
                                           ("emf_2023_university_5_with_ptv", (0.7, 0.1)),
                                           ("voip_mobile", (0.6, 0.2))])
def test_band_labels_count_the_loaders_channels(dataset, split):
    window = (48, 12) if dataset.startswith("voip") else (336, 96)
    per_channel = baselines.lookback_mean_mse(dataset, *window, *split)
    assert len(baselines.band_names(dataset)) == len(per_channel)


def _run(tmp_path, k=None, **extra):
    log = _log(**{"parameters/data/dataset": "VOIP_MOBILE",
                  "parameters/data/seq_len": 48, "parameters/data/pred_len": 12,
                  "sl_test/mse_per_channel": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}, **extra)
    rel = "M/voip_mobile/L48_O12_cp/" + (f"k{k}/" if k else "") + "1"
    if k:
        log.update({"parameters/cluster/enabled": True, "parameters/cluster/k_clusters": k})
    _write(tmp_path, rel, log)
    (run,) = load_runs(tmp_path, [])
    return run


def test_a_joint_run_is_one_row_per_band(tmp_path):
    run = _run(tmp_path, k=3, **{"sl_test_fixed/loss": [0.3],
                                 "sl_test_fixed/mse_per_channel": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]})
    names = baselines.band_names("voip_mobile")
    rows = frames.band_rows("cluster", run, names, [1.0] * 6)
    assert [list(r) for r in rows] == [frames.BAND_FIELDS] * 6
    assert [r["channel"] for r in rows] == list(range(6))
    assert [r["band"] for r in rows] == names
    assert [float(r["mse"]) for r in rows] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert [float(r["mse_fixed"]) for r in rows] == [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # Joint runs log no per-band MAE; it is left blank, never filled in.
    assert {r["mae"] for r in rows} == {""}
    assert {(r["k"], r["fixed_source"], r["lookback_mse"]) for r in rows} == {
        (3, "log", "1.0")}


def test_a_per_channel_run_is_its_one_band(tmp_path):
    band = tmp_path / "PC"
    _write(band, "M/voip_mobile/L48_O12_cp/ch2/1",
           _log(**{"parameters/data/dataset": "VOIP_MOBILE", "sl_test/loss": [0.33],
                   "sl_test/mae": [0.22], "sl_test/mse_per_channel": [0.33]}))
    (run,) = load_runs(band, [])
    (row,) = frames.band_rows("perchannel", run, baselines.band_names("voip_mobile"),
                              [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert (row["channel"], row["band"], row["mse"], row["mae"]) == (2, "RTT", "0.33", "0.22")
    assert row["lookback_mse"] == "0.7" and row["mse_fixed"] == ""


def test_a_band_vector_of_the_wrong_length_is_refused(tmp_path):
    run = _run(tmp_path)
    with pytest.raises(ValueError, match="6 per-channel MSEs for 5 bands"):
        frames.band_rows("multivariate", run, ["a", "b", "c", "d", "e"], None)


def test_the_series_tree_has_no_bands(tmp_path):
    _write(tmp_path, "M/det/L336_O96_cp/1", _log(**{"parameters/data/dataset": "rf_emf_det"}))
    loaded = [("series", r) for r in load_runs(tmp_path, [])]
    assert frames.build_bands(loaded) == []


# --------------------------------------------------------------- timing

SOLO = {"seq_len": 336, "pred_len": 96, "bands": [
    {"model": "DLinear", "dataset": "ed2022", "channel": 0, "epochs": 10,
     "seconds": 10.0, "twin_epochs": 10, "twin_seconds": 20.0},
    {"model": "DLinear", "dataset": "ed2022", "channel": 1, "epochs": 10,
     "seconds": 30.0, "twin_epochs": 20, "twin_seconds": 90.0},
]}


def test_timing_is_the_solo_per_epoch_cost_times_every_band_epoch():
    rows = [_row("perchannel", s, 0.1, channel=ch, epochs=e, running_time=repr(t),
                 concurrency=12)
            for s, bands in {1: ((0, 10, 20.0), (1, 30, 70.0)),
                             2: ((0, 20, 40.0),)}.items()      # seed 2 is incomplete
            for ch, e, t in bands]
    rows.append(_row("multivariate", 1, 0.1, running_time="25.0"))
    (row,) = frames.build_timing(rows, SOLO)
    assert list(row) == frames.TIMING_FIELDS
    assert (row["seed"], row["n_bands"], row["epochs"]) == ("1", 2, 40)
    assert float(row["solo_per_epoch"]) == pytest.approx(2.0)     # 40 s over 20 epochs
    assert float(row["solo_total"]) == pytest.approx(80.0)
    assert float(row["solo_lo"]) == pytest.approx(40.0)           # 1 s/epoch band
    assert float(row["solo_hi"]) == pytest.approx(120.0)          # 3 s/epoch band
    assert float(row["contention_mean"]) == pytest.approx((2.0 + 1.5) / 2)
    assert float(row["contended_total"]) == pytest.approx(90.0)
    assert float(row["solo_x_mv"]) == pytest.approx(80.0 / 25.0)
    assert row["concurrency"] == "12"
    # Nothing measured, nothing estimated.
    assert frames.build_timing(rows, {}) == []


def test_the_timing_spec_covers_every_measured_cell():
    solo = cli.spec_timing()
    assert solo["seq_len"] == 336 and solo["pred_len"] == 96
    cells = {(b["model"], b["dataset"]) for b in solo["bands"]}
    assert len(solo["bands"]) == 54 and len(cells) == 18
    assert all(b["epochs"] > 0 and b["seconds"] > 0 for b in solo["bands"])
