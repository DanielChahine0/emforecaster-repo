"""What the paradigm figures plot: the paradigm comparison's numbers, nothing else.

Each test builds a small synthetic frames directory whose right answer is
known -- the helpers are test_paradigms' -- so a figure that drifts from the
table beside it, draws a hidden dataset, or ranks a sweep still training
shows up here rather than in a published PNG.
"""

from pathlib import Path

import pytest

from emfres import figdata as FD
from emfres import frames
from emfres.paradigms import BEST, K1, MV, PC

from test_paradigms import SEEDS, _band, _run


def _frames(tmp_path, rows, bands=()) -> Path:
    frames.write_csv(tmp_path / "runs.csv", rows, frames.RUN_FIELDS)
    frames.write_csv(tmp_path / "cells.csv", frames.build_cells(rows), frames.CELL_FIELDS)
    if bands:
        frames.write_csv(tmp_path / "bands.csv", list(bands), frames.BAND_FIELDS)
    return tmp_path


def _corpus_rows():
    """ED2022: DLinear in every paradigm, k=2 chosen on validation (k=3 is the
    test-best k); LSTM's sweep has one seed of k=1 so far; an ED2024 run."""
    rows = []
    for s, v in zip(SEEDS, (0.400, 0.410, 0.420)):
        rows.append(_run("multivariate", s, v, lookback=1.0))
        rows.append(_run("multivariate", s, v + 0.05, model="LSTM", lookback=1.0))
        rows.append(_run("multivariate", s, v, dataset="ed2024"))
        for ch in (0, 1):
            rows.append(_run("perchannel", s, 0.43 + 0.01 * ch, channel=ch, lookback=1.0))
            rows.append(_run("perchannel", s, 0.47 + 0.01 * ch, channel=ch, model="LSTM",
                             lookback=1.0))
        rows.append(_run("cluster", s, v + 0.001, k=1, val_loss=0.50, lookback=1.0))
        rows.append(_run("cluster", s, v - 0.010, k=2, val_loss=0.30, lookback=1.0))
        rows.append(_run("cluster", s, v - 0.020, k=3, val_loss=0.40, lookback=1.0))
    rows.append(_run("cluster", SEEDS[0], 0.44, k=1, val_loss=0.5, model="LSTM",
                     source="log", lookback=1.0))
    return rows


def _load(tmp_path, rows=None, bands=()):
    d = _frames(tmp_path, _corpus_rows() if rows is None else rows, bands)
    return FD.load(d / "runs.csv", "emf", (336, 96)), d


def test_a_hidden_dataset_never_reaches_a_figure(tmp_path):
    corpus, _ = _load(tmp_path)
    assert "ed2024" not in corpus.datasets
    assert all(key[1] != "ed2024" for key in corpus.values)


def test_the_marks_are_the_boards_seed_values(tmp_path):
    corpus, _ = _load(tmp_path)
    mv = FD.marks(corpus, ("ed2022", "DLinear"), MV, "mae")
    assert mv.mean == pytest.approx(0.41) and mv.n == 3 and mv.complete
    # A per-channel seed is the mean over its channels.
    assert FD.marks(corpus, ("ed2022", "DLinear"), PC, "mae").mean == pytest.approx(0.435)
    best = FD.marks(corpus, ("ed2022", "DLinear"), BEST, "mae")
    assert best.k == 2 and best.complete          # validation's k, not the test-best 3
    assert FD.marks(corpus, ("ed2022", "DLinear"), K1, "mae").k == 1
    # Skill is 1 - MSE / lookback-mean MSE, seed by seed.
    skill = FD.marks(corpus, ("ed2022", "DLinear"), MV, "skill")
    assert skill.seeds["69421"] == pytest.approx(1 - 0.4 * 1.5)


def test_a_sweep_still_training_is_drawn_but_marked_incomplete(tmp_path):
    corpus, _ = _load(tmp_path)
    best = FD.marks(corpus, ("ed2022", "LSTM"), BEST, "mae")
    assert best is not None and not best.complete and best.n == 1
    gaps = {g.cell: g for g in FD.gaps(corpus, K1, BEST, "mae")}
    assert gaps[("ed2022", "LSTM")].outcome == "pending"
    assert gaps[("ed2022", "DLinear")].outcome == "challenger"
    # The scoreboard counts the pending cell as pending, not as a result.
    t = FD.score(corpus, K1, BEST, "mae")
    assert t["pending"] == 1 and t["cells"] == 1


def test_pending_cells_are_the_ones_the_scoreboard_counts_pending(tmp_path):
    # PatchTST is planned for the cluster grid but has no cluster run yet.
    rows = _corpus_rows() + [_run(setup, s, 0.45, model="PatchTST", lookback=1.0, **kw)
                             for s in SEEDS for setup, kw in
                             (("multivariate", {}), ("perchannel", {"channel": 0}))]
    corpus, _ = _load(tmp_path, rows)
    for base, chal, _ in FD.GAP_COMPARISONS:
        waiting = FD.pending_cells(corpus, base, chal)
        assert len(waiting) == FD.score(corpus, base, chal, "mae")["pending"]
    waiting = FD.pending_cells(corpus, K1, BEST)
    assert waiting == [("ed2022", "LSTM"), ("ed2022", "PatchTST")]
    # LSTM's has a gap to plot; PatchTST's has none yet, only a count.
    drawn = {g.cell for g in FD.gaps(corpus, K1, BEST, "mae") if g.outcome == "pending"}
    assert drawn == {("ed2022", "LSTM")}


def test_the_k_profile_is_relative_to_k1_and_rings_only_a_finished_sweep(tmp_path):
    corpus, _ = _load(tmp_path)
    prof = FD.k_profile(corpus, ("ed2022", "DLinear"), "mae")
    assert [p.k for p in prof.points] == [1, 2, 3]
    assert prof.points[0].rel == 0.0
    expected = sum((v - 0.010) / (v + 0.001) - 1 for v in (0.400, 0.410, 0.420)) / 3
    assert prof.points[1].rel == pytest.approx(expected)
    assert prof.chosen == 2 and prof.oracle == 3
    partial = FD.k_profile(corpus, ("ed2022", "LSTM"), "mae")
    assert partial.chosen is None and not partial.points[0].complete


def test_band_shares_carry_the_sign_and_sum_to_one_in_absolute_value(tmp_path):
    rows, bands = [], []
    for s in SEEDS:
        rows.append(_run("multivariate", s, 0.5, dataset="uh2022", mse=1.5))
        bands += [_band("multivariate", s, ch, v) for ch, v in enumerate((1.0, 2.0, 1.5))]
        for ch, v in enumerate((1.1, 2.5, 1.4)):
            rows.append(_run("perchannel", s, 0.5, dataset="uh2022", channel=ch, mse=v))
            bands.append(_band("perchannel", s, ch, v))
    corpus, d = _load(tmp_path, rows, bands)
    (grid,) = FD.band_grids(corpus, d / "bands.csv")
    assert grid.models == ["DLinear"] and grid.channels == [0, 1, 2]
    share = grid.share()
    assert share[1] == pytest.approx(0.5 / 0.7)          # the band that carries the gap
    assert share[2] == pytest.approx(-0.1 / 0.7)         # per-channel better there
    assert sum(abs(v) for v in share.values()) == pytest.approx(1.0)


def test_a_model_left_out_of_the_shares_leaves_the_rest_summing_to_one(tmp_path):
    rows, bands = [], []
    for s in SEEDS:
        for model, pc in (("DLinear", (1.1, 2.5, 1.4)), ("RF_EMF_Transformer", (4.0, 2.0, 1.5))):
            rows.append(_run("multivariate", s, 0.5, model=model, dataset="uh2022", mse=1.5))
            bands += [_band("multivariate", s, ch, v, model=model)
                      for ch, v in enumerate((1.0, 2.0, 1.5))]
            for ch, v in enumerate(pc):
                rows.append(_run("perchannel", s, 0.5, model=model, dataset="uh2022",
                                 channel=ch, mse=v))
                bands.append(_band("perchannel", s, ch, v, model=model))
    corpus, d = _load(tmp_path, rows, bands)
    (grid,) = FD.band_grids(corpus, d / "bands.csv")
    # With the Transformer, its band 0 (+3.0) carries most of the gap ...
    assert grid.share()[0] == pytest.approx((0.1 + 3.0) / 3.7)
    # ... left out, the shares are DLinear's alone, as if it were the only row.
    share = grid.share(skip={"RF_EMF_Transformer"})
    assert share[1] == pytest.approx(0.5 / 0.7) and share[0] == pytest.approx(0.1 / 0.7)
    assert sum(abs(v) for v in share.values()) == pytest.approx(1.0)


def test_the_spot_check_holds_the_figures_to_cells_csv(tmp_path):
    corpus, d = _load(tmp_path)
    n, bad = FD.spot_check(corpus, d / "cells.csv")
    assert n > 0 and bad == []
    # A plotted mean that drifted from the tables is reported.
    corpus.values[(MV, "ed2022", "DLinear")]["69421"]["mae"] += 0.01
    _, bad = FD.spot_check(corpus, d / "cells.csv")
    assert [(key[0], metric) for key, metric, *_ in bad] == [("multivariate", "mae")]


def test_the_spot_check_also_holds_coverage_width_and_cost_to_cells_csv(tmp_path):
    rows = _corpus_rows()
    for i, r in enumerate(rows):
        r.update(ic=repr(88.0 + i % 3), interval_width=repr(1.0 + 0.01 * i),
                 running_time=repr(60.0 + i))
        if r["setup"] == "cluster":
            r.update(ic_fixed=repr(87.0 + i % 3), interval_width_fixed=repr(1.1 + 0.01 * i))
    corpus, d = _load(tmp_path, rows)
    n_means, _ = FD.spot_check(corpus, d / "cells.csv")
    n, bad = FD.spot_check(corpus, d / "cells.csv", runs_csv=d / "runs.csv")
    assert n > n_means and bad == []
    # A coverage the figure would draw that no longer matches cells.csv.
    frames.write_csv(d / "runs.csv", [dict(r, ic="95.0") if r["setup"] == "multivariate"
                                      and r["model"] == "DLinear" else r for r in rows],
                     frames.RUN_FIELDS)
    _, bad = FD.spot_check(corpus, d / "cells.csv", runs_csv=d / "runs.csv")
    assert {(key[0], metric) for key, metric, *_ in bad} == {("multivariate", "coverage")}


def test_the_series_spot_check_holds_its_seed_means(tmp_path):
    rows = [_run("series", s, 0.3 + 0.01 * i, dataset="det") for i, s in enumerate(SEEDS)]
    d = _frames(tmp_path, rows)
    n, bad = FD.spot_check_series(d / "runs.csv", d / "cells.csv")
    assert n == 2 and bad == []


def test_the_series_grid_is_read_on_its_own(tmp_path):
    rows = [_run("series", s, 0.3 + 0.01 * i, dataset="det")
            for i, s in enumerate(SEEDS)]
    rows.append(_run("series", SEEDS[0], 0.9, dataset="det", seq_len=192))
    rows += [_run("multivariate", s, 0.4) for s in SEEDS]
    d = _frames(tmp_path, rows)
    vals = FD.series_values(d / "runs.csv")
    assert list(vals) == [("det", "DLinear")] and len(vals[("det", "DLinear")]) == 3
    # Never a paradigm pipeline.
    assert all(key[1] != "det" for key in FD.load(d / "runs.csv", "emf", (336, 96)).values)


# The multivariate numbers the sweep does not retrain: pinned against the
# tracked frames, so a regression in how a figure reads them shows up even
# before the frames are next regenerated. (Per-channel and cluster values are
# checked against cells.csv by spot_check instead: the sweep moves them.)
PINNED_MV_MAE = {
    ("emf", (336, 96), "ed2022", "EMForecaster"): 0.3358,
    ("emf", (336, 96), "uh2022", "PatchTST"): 0.4922,
    ("voip", (48, 12), "voip_mobile", "DLinear"): 0.3250,
}


@pytest.mark.parametrize("key", sorted(PINNED_MV_MAE))
def test_multivariate_values_of_the_tracked_frames(key):
    runs_csv = Path(__file__).resolve().parents[1] / "frames" / "runs.csv"
    if not runs_csv.exists():
        pytest.skip("no tracked frames/runs.csv")
    family, window, dataset, model = key
    corpus = FD.load(runs_csv, family, window)
    m = FD.marks(corpus, (dataset, model), MV, "mae")
    assert m is not None and round(m.mean, 4) == PINNED_MV_MAE[key]
