"""The paradigm comparison: seed-aware, matched, validation-chosen k.

Each test builds a small synthetic runs.csv whose right answer is known, so a
regression in how seeds are averaged, matched, tied or selected shows up here
rather than as a quietly wrong headline.
"""

from pathlib import Path

import pytest

from emfres import frames, paradigms


def _run(setup, seed, mae, *, model="DLinear", dataset="ed2022", k="", channel="",
         val_loss="", mse=None):
    row = {f: "" for f in frames.RUN_FIELDS}
    row.update(setup=setup, family="emf", model=model, dataset=dataset, seq_len=336,
               pred_len=96, k=k, channel=channel, seed=str(seed), conf=1,
               mae=repr(mae), mse=repr(mae * 1.5 if mse is None else mse),
               val_loss=repr(val_loss) if val_loss != "" else "")
    return row


def _csv(tmp_path, rows) -> Path:
    path = tmp_path / "runs.csv"
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


def test_a_per_channel_seed_missing_a_channel_is_dropped(tmp_path):
    values = paradigms.seed_values(_csv(tmp_path, _sweep()), "emf", 336, 96)
    pc = values[("perchannel", "ed2022", "DLinear")]
    assert sorted(pc) == ["69421", "69422"]
    # And a complete per-channel seed is the mean over its channels.
    assert pc["69421"]["mae"] == pytest.approx(0.305)


def test_best_k_is_chosen_on_validation_not_test(tmp_path):
    cells, chosen = paradigms.pipelines(
        paradigms.seed_values(_csv(tmp_path, _sweep()), "emf", 336, 96))
    # k=3 has the best test error; picking it would be an oracle choice.
    assert chosen[("ed2022", "DLinear")] == 2


def test_comparisons_use_matched_seeds_and_a_seed_noise_tie_band(tmp_path):
    cells, _ = paradigms.pipelines(
        paradigms.seed_values(_csv(tmp_path, _sweep()), "emf", 336, 96))
    c = cells[("ed2022", "DLinear")]

    pc = paradigms.compare(c[paradigms.MV], c[paradigms.PC], "mae")
    assert pc["n"] == 2                          # per-channel has two complete seeds
    assert pc["outcome"] == "challenger"         # 0.30 vs 0.40, far outside the noise

    k1 = paradigms.compare(c[paradigms.MV], c[paradigms.K1], "mae")
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


def test_render_names_the_chosen_k_and_skips_when_nothing_is_shared(tmp_path):
    md = paradigms.render(_csv(tmp_path, _sweep()), "emf", (336, 96))
    assert md.startswith("# Paradigm Comparison")
    assert "(k=2)" in md

    only_joint = [_run("multivariate", s, 0.4) for s in SEEDS]
    assert paradigms.render(_csv(tmp_path, only_joint), "emf", (336, 96)) == ""


def test_other_families_and_windows_are_left_out(tmp_path):
    rows = _sweep() + [dict(_run("multivariate", 69421, 9.9), family="voip",
                            dataset="voip_mobile")]
    values = paradigms.seed_values(_csv(tmp_path, rows), "emf", 336, 96)
    assert not any(ds == "voip_mobile" for _, ds, _ in values)
    assert paradigms.seed_values(_csv(tmp_path, rows), "emf", 192, 96) == {}
