"""The paradigm comparison: seed-aware, matched, validation-chosen k.

Each test builds a small synthetic runs.csv whose right answer is known, so a
regression in how seeds are averaged, matched, tied, selected or counted shows
up here rather than as a quietly wrong headline.
"""

from pathlib import Path

import pytest

from emfres import frames, paradigms
from emfres.paradigms import BEST, K1, MV, PC


def _run(setup, seed, mae, *, model="DLinear", dataset="ed2022", k="", channel="",
         val_loss="", mse=None, lookback=None, best_epoch="", fixed=None,
         source="sidecar", val_fixed=None, family="emf", seq_len=336, pred_len=96):
    """One runs.csv row. A cluster row's fixed metrics default to its batch ones
    (``fixed`` overrides mae_fixed, mse_fixed follows as 1.5x) with the batch
    val_loss, from a re-score."""
    mse = mae * 1.5 if mse is None else mse
    row = {f: "" for f in frames.RUN_FIELDS}
    row.update(setup=setup, family=family, model=model, dataset=dataset, seq_len=seq_len,
               pred_len=pred_len, k=k, channel=channel, seed=str(seed), conf=1,
               mae=repr(mae), mse=repr(mse),
               val_loss=repr(val_loss) if val_loss != "" else "",
               lookback_mse="" if lookback is None else repr(lookback),
               best_epoch=best_epoch)
    if setup == "cluster":
        fmae = mae if fixed is None else fixed
        vf = val_loss if val_fixed is None else val_fixed
        row.update(mae_fixed=repr(fmae), mse_fixed=repr(fmae * 1.5 if fixed is not None else mse),
                   val_loss_fixed=repr(vf) if vf != "" else "", fixed_source=source)
    return row


def _csv(tmp_path, rows, name="runs.csv") -> Path:
    path = tmp_path / name
    frames.write_csv(path, rows, frames.RUN_FIELDS)
    return path


SEEDS = (69421, 69422, 69423)


def _sweep():
    rows = []
    for s, v in zip(SEEDS, (0.400, 0.410, 0.420)):
        rows.append(_run("multivariate", s, v))
    # Per-channel: two channels, but seed 69423 is missing channel 1.
    for s in SEEDS:
        for ch in (0, 1):
            if s == 69423 and ch == 1:
                continue
            rows.append(_run("perchannel", s, 0.30 + 0.01 * ch, channel=ch))
    # Cluster: k=1 ties the joint network; k=2 has the best validation loss
    # but the worse test error; k=3 has the best test error but worse validation.
    for s, v in zip(SEEDS, (0.401, 0.409, 0.421)):
        rows.append(_run("cluster", s, v, k=1, val_loss=0.50))
    for s in SEEDS:
        rows.append(_run("cluster", s, 0.405, k=2, val_loss=0.30))
        rows.append(_run("cluster", s, 0.350, k=3, val_loss=0.40))
    return rows


def _values(tmp_path, rows, family="emf", window=(336, 96)):
    return paradigms.seed_values(_csv(tmp_path, rows), family, *window)


# --------------------------------------------------------------- seeds and k

def test_a_per_channel_seed_missing_a_channel_is_dropped(tmp_path):
    pc = _values(tmp_path, _sweep())[("perchannel", "ed2022", "DLinear")]
    assert sorted(pc) == ["69421", "69422"]
    # And a complete per-channel seed is the mean over its channels.
    assert pc["69421"]["mae"] == pytest.approx(0.305)


def test_best_k_is_chosen_on_validation_not_test(tmp_path):
    cells, chosen = paradigms.pipelines(_values(tmp_path, _sweep()))
    # k=3 has the best test error; picking it would be an oracle choice.
    assert chosen[("ed2022", "DLinear")] == 2


def test_the_fixed_protocol_is_primary_and_ranks_on_its_own_validation(tmp_path):
    rows = [_run("multivariate", s, 0.40) for s in SEEDS]
    for s in SEEDS:
        # Batch: k=2 validates best. Fixed: k=3 does, and scores other numbers.
        rows.append(_run("cluster", s, 0.40, k=1, val_loss=0.5))
        rows.append(_run("cluster", s, 0.39, k=2, val_loss=0.30, fixed=0.38, val_fixed=0.45))
        rows.append(_run("cluster", s, 0.37, k=3, val_loss=0.35, fixed=0.36, val_fixed=0.40))
    values = _values(tmp_path, rows)
    cells, chosen = paradigms.pipelines(values)
    assert chosen[("ed2022", "DLinear")] == 3
    assert cells[("ed2022", "DLinear")][BEST]["69421"]["mae"] == pytest.approx(0.36)
    batch_cells, batch_chosen = paradigms.pipelines(values, "batch")
    assert batch_chosen[("ed2022", "DLinear")] == 2
    assert batch_cells[("ed2022", "DLinear")][BEST]["69421"]["mae"] == pytest.approx(0.39)


def test_a_run_without_fixed_metrics_is_not_averaged_in(tmp_path):
    rows = [_run("cluster", s, 0.40, k=2, val_loss=0.3) for s in SEEDS]
    rows[2].update(mse_fixed="", mae_fixed="", val_loss_fixed="", fixed_source="none")
    values = _values(tmp_path, rows)
    fixed = paradigms.view(values[(("cluster", 2), "ed2022", "DLinear")], "fixed")
    assert sorted(fixed) == ["69421", "69422"]


def test_fixed_metrics_of_two_origins_choose_no_k(tmp_path):
    rows = [_run("multivariate", s, 0.40) for s in SEEDS]
    for s in SEEDS:
        rows.append(_run("cluster", s, 0.40, k=1, val_loss=0.5, source="log"))
        rows.append(_run("cluster", s, 0.39, k=2, val_loss=0.4, source="sidecar"))
    board = paradigms.build(_values(tmp_path, rows))
    choice = board.choices[("ed2022", "DLinear")]
    assert choice.k is None and "two origins" in choice.reason
    assert BEST not in board.cells[("ed2022", "DLinear")]
    # Counted as pending, never as a result.
    t = paradigms.tally(board, K1, BEST, "mae", [("ed2022", "DLinear")])
    assert (t["cells"], t["pending"]) == (0, 1)
    # The batch protocol has no origin to check.
    assert paradigms.build(_values(tmp_path, rows), "batch").choices[
        ("ed2022", "DLinear")].k == 2


def test_a_k_still_training_is_not_ranked_and_the_cell_is_pending(tmp_path):
    rows = [_run("multivariate", s, 0.40) for s in SEEDS]
    for s in SEEDS:
        rows.append(_run("cluster", s, 0.40, k=1, val_loss=0.5))
        rows.append(_run("cluster", s, 0.39, k=2, val_loss=0.4))
    # k=3 would validate best, but has one seed so far.
    rows.append(_run("cluster", 69421, 0.30, k=3, val_loss=0.1))
    board = paradigms.build(_values(tmp_path, rows))
    choice = board.choices[("ed2022", "DLinear")]
    assert choice.k == 2 and choice.ranked == (1, 2) and choice.partial
    assert "ranked 1,2 of 1,2,3" in choice.tag()
    order = [("ed2022", "DLinear")]
    assert paradigms.tally(board, K1, BEST, "mae", order)["pending"] == 1
    # k=1 itself is complete, so the cluster-module question is answered.
    assert paradigms.tally(board, MV, K1, "mae", order)["cells"] == 1


def test_a_cell_short_of_the_family_seeds_is_shown_but_not_counted(tmp_path):
    rows = _sweep() + [_run("multivariate", s, 0.40, model="TSMixer") for s in SEEDS]
    rows += [_run("perchannel", 69421, 0.39, model="TSMixer", channel=0)]
    board = paradigms.build(_values(tmp_path, rows))
    order = [("ed2022", "DLinear"), ("ed2022", "TSMixer")]
    t = paradigms.tally(board, MV, PC, "mae", order)
    # DLinear's per-channel cell has 2 of 3 seeds, TSMixer's 1: both pending.
    assert (t["cells"], t["pending"]) == (0, 2)
    md = paradigms.render(_csv(tmp_path, rows), "emf", (336, 96))
    assert "(n=1)" in md and "3/1/-/-" in md


# --------------------------------------------------------------- comparisons

def test_comparisons_use_matched_seeds_and_a_seed_noise_tie_band(tmp_path):
    cells, _ = paradigms.pipelines(_values(tmp_path, _sweep()))
    c = cells[("ed2022", "DLinear")]

    pc = paradigms.compare(c[MV], c[PC], "mae")
    assert pc["n"] == 2                          # per-channel has two complete seeds
    assert pc["outcome"] == "challenger"         # 0.30 vs 0.40, far outside the noise

    k1 = paradigms.compare(c[MV], c[K1], "mae")
    assert k1["n"] == 3
    assert k1["outcome"] == "tie"                # a 0.0003 gap inside a 0.01 spread


def test_identical_arithmetic_is_a_duplicate_not_evidence():
    same = {"1": {"mae": 0.6, "mse": 1.0}, "2": {"mae": 0.6, "mse": 1.0}}
    r = paradigms.compare(same, same, "mae")
    assert r["duplicate"] and r["outcome"] == "tie"


def test_wilcoxon_needs_something_to_test():
    assert paradigms.wilcoxon_p([]) is None
    assert paradigms.wilcoxon_p([0.0, 0.0, 0.0]) is None
    p = paradigms.wilcoxon_p([-0.05, -0.04, -0.06, -0.03, -0.05, -0.02, -0.04])
    assert p is not None and p < 0.05


def test_deterministic_models_are_identical_by_construction_not_ties(tmp_path):
    rows = []
    for model, gap in (("DLinear", 0.01), ("ARIMA", 0.0)):
        for s in SEEDS:
            rows.append(_run("multivariate", s, 0.60 + 0.001 * SEEDS.index(s), model=model))
            rows.append(_run("perchannel", s, 0.60 + gap + 0.001 * SEEDS.index(s),
                             model=model, channel=0))
    board = paradigms.build(_values(tmp_path, rows))
    t = paradigms.tally(board, MV, PC, "mae", [("ed2022", "DLinear"), ("ed2022", "ARIMA")])
    assert (t["cells"], t["identical"]) == (1, 1)
    assert t["count"]["baseline"] == 1 and t["count"]["tie"] == 0
    md = paradigms.render(_csv(tmp_path, rows), "emf", (336, 96))
    arima = next(line for line in md.splitlines() if line.startswith("| ed2022 | ARIMA"))
    assert "identical" in arima and "n/a (per-band)" in arima and "1/1/n/a/n/a" in arima
    assert "±" not in arima


# --------------------------------------------------------------- rendering

def test_render_names_the_chosen_k_and_its_origin(tmp_path):
    md = paradigms.render(_csv(tmp_path, _sweep()), "emf", (336, 96))
    assert md.startswith("# Paradigm Comparison")
    assert "(k=2, sidecar)" in md
    assert "batch protocol" in md                     # the sensitivity row

    only_joint = [_run("multivariate", s, 0.4) for s in SEEDS]
    assert paradigms.render(_csv(tmp_path, only_joint), "emf", (336, 96)) == ""


def test_other_families_and_windows_are_left_out(tmp_path):
    rows = _sweep() + [dict(_run("multivariate", 69421, 9.9), family="voip",
                            dataset="voip_mobile")]
    values = _values(tmp_path, rows)
    assert not any(ds == "voip_mobile" for _, ds, _ in values)
    assert _values(tmp_path, rows, window=(192, 96)) == {}
