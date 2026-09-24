"""The value gate: what it holds, how it compares, and which changes it forgives.

The gate used to hold only the per-setup MSE/MAE cells, text-diffed every
other section, and printed "every published value still matches" beside a
section whose values it had never looked at. These pin that the paradigm
scoreboard and the CCM sweep are held value by value (in both layouts the
sweep has had), that a changed section says what was held of it, and that
only a change specs/gate.yaml does not declare fails the gate.
"""

import io
from pathlib import Path

from emfres import frames, reconcile as R, tables

RESULTS = Path(__file__).resolve().parents[1]
SNAPSHOT = RESULTS / "archive" / "2026-09-22_tables.md"


def _scoreboard(rows, *, extra_sensitivity=()):
    """A minimal Paradigm Comparison section: (question, metric, W, T, L, cells, p)."""
    out = ["# Paradigm Comparison", "", "## 1 · Scoreboard", "",
           "| Question | Metric | Challenger better | Tie | Baseline better | Cells "
           "| Median gap | Wilcoxon p |",
           "|:---|:---|---:|---:|---:|---:|---:|---:|"]
    for q, metric, w, t, l, n, p in rows:
        out.append(f"| {q} | {metric} | {w} | {t} | {l} | {n} | +1.00% | {p} |")
    if extra_sensitivity:
        out += ["", "## 2 · Sensitivity", "",
                "| Comparison | Variant | Metric | Challenger better | Tie "
                "| Baseline better | Cells | Median gap | Wilcoxon p |",
                "|:---|:---|:---|---:|---:|---:|---:|---:|---:|"]
        out += [f"| {c} | {v} | MSE | 1 | 1 | 1 | 3 | +0.50% | 0.5 |"
                for c, v in extra_sensitivity]
    return "\n".join(out) + "\n"


PC = "Per-channel vs Multivariate — Is forecasting each channel alone better than jointly?"
BEST = "Cluster (best k) vs Cluster (k=1) — Does clustering itself help?"


def _gate(tmp_path, published_md, current_md, rules=None, *, snapshot=None):
    published = tmp_path / "2026-01-01_tables.md"
    published.write_text(published_md)
    cells = tmp_path / "cells.csv"
    frames.write_csv(cells, [], frames.CELL_FIELDS)
    spec = tmp_path / "gate.yaml"
    import yaml
    spec.write_text(yaml.safe_dump({"snapshot": snapshot or published.name,
                                    "expected": rules or []}))
    sections = [(f"s{i}", heading, block)
                for i, (heading, block) in enumerate(R.split_blocks(current_md))]
    return R.gate(published, cells, sections, runs_spec=tmp_path / "none.yaml",
                  gate_spec=spec, out=io.StringIO())


# --------------------------------------------------------------------- compare

def test_values_agree_at_the_precision_the_snapshot_printed():
    assert R.same("0.00391", "0.003906")           # p at 3 significant digits
    assert R.same("0.3768 ±0.0002", "0.37684 ±0.0002")
    assert R.same("-0.0%", "+0.0%")
    assert not R.same("+1.03%", "+1.04%")
    assert not R.same("9", "8")


def test_a_p_printed_without_its_trailing_zeros_is_not_a_coarse_number():
    # A Wilcoxon p is printed with :.3g, which drops trailing zeros: "1" is
    # 1.00, not a p known to the unit. The snapshot's k=1-vs-multivariate MSE
    # p was "1" and a later render printed "0.5".
    assert not R.same("1", "0.5")
    assert not R.same("1", "0.625")
    assert not R.same("0.5", "0.54")
    assert not R.same("7.63e-06", "7.6e-06")       # 7.6e-06 is 7.60e-06
    assert R.same("1", "1")
    assert R.same("0.438", "0.4375")


def test_the_text_around_a_value_is_part_of_it():
    assert not R.same("≈ +0.6%", "+0.6%")          # a tie mark lost
    assert not R.same("0.3768 ±0.0002", "0.3768 ±0.0002 (n=2)")   # now incomplete
    assert not R.same("0.3768 ±0.0002", "0.3768")  # the sd vanished


# --------------------------------------------------------------------- headings

def test_a_sweep_rendered_per_dataset_is_found_by_its_prefix():
    blocks = [(f"{tables.SWEEP_TITLE} · ED2022 · MAE", "a"), ("Multivariate", "b")]
    assert R.find_blocks(blocks, tables.SWEEP_TITLE) == [blocks[0]]


def test_an_exact_heading_wins_and_a_claimed_one_is_never_borrowed():
    both = [("Paradigm Comparison", "a"), ("Paradigm Comparison · L24/O6", "b")]
    assert R.find_blocks(both, "Paradigm Comparison") == [both[0]]
    assert R.find_blocks(both, "Paradigm Comparison · L24/O6") == [both[1]]
    only_l24 = [("Paradigm Comparison · L24/O6", "b")]
    claimed = frozenset({"Paradigm Comparison · L24/O6"})
    assert R.find_blocks(only_l24, "Paradigm Comparison", claimed=claimed) == []


# --------------------------------------------------------------------- reading

def test_scoreboard_rows_are_keyed_by_comparison_not_by_wording():
    a = R.scoreboard_values(_scoreboard([(PC, "MAE", 1, 10, 10, 21, "0.0268")]))
    b = R.scoreboard_values(_scoreboard(
        [("Per-channel vs Multivariate — reworded", "MAE", 1, 10, 10, 21, "0.0268")]))
    assert a == b
    key = ("scoreboard", "Paradigm Comparison", "scoreboard",
           "Per-channel vs Multivariate", "", "MAE", "Wilcoxon p")
    assert a[key] == "0.0268"


def test_the_frozen_snapshot_reads_as_published():
    text = SNAPSHOT.read_text()
    board = R.scoreboard_values(text)
    assert len(board) == 48                        # 8 rows x 6 values
    assert board[("scoreboard", "Paradigm Comparison", "scoreboard",
                  "Cluster (best k) vs Cluster (k=1)", "", "MSE", "Wilcoxon p")] == "0.00391"
    sweep = R.sweep_values(text)
    head = ("sweep", "ED2022", "L336/O96", "MAE", "DLinear")
    assert sweep[head + ("k", "2")] == "0.3768 ±0.0002"
    assert sweep[head + ("plain", "")] == "0.3771 ±0.0001"
    # The old layout underlined the test-best k: today's oracle k.
    assert sweep[head + ("oracle k", "")] == "2"
    assert sweep[("sweep", "ED2022", "L336/O96", "MAE", "PatchTST", "oracle k", "")] == "9"
    assert len(sweep) == 21                        # 3 models x (plain, 5 k, oracle)


def _row(setup, seed, mae, *, k="", val=None):
    row = {f: "" for f in frames.RUN_FIELDS}
    row.update(setup=setup, family="emf", model="DLinear", dataset="ed2022", seq_len=336,
               pred_len=96, k=k, seed=str(seed), conf=1, mae=repr(mae), mse=repr(mae * 1.5))
    if setup == "cluster":
        row.update(mae_fixed=repr(mae), mse_fixed=repr(mae * 1.5), val_loss=repr(val),
                   val_loss_fixed=repr(val), fixed_source="sidecar")
    return row


def test_todays_sweep_layout_reads_to_the_same_keys(tmp_path):
    rows = []
    for s in (1, 2, 3):
        rows.append(_row("multivariate", s, 0.40 + 0.001 * s))
        # k=2 validates best, k=3 tests best.
        for k, mae, val in ((1, 0.40, 0.5), (2, 0.38, 0.3), (3, 0.36, 0.4)):
            rows.append(_row("cluster", s, mae + 0.001 * s, k=k, val=val))
    runs = tmp_path / "runs.csv"
    frames.write_csv(runs, rows, frames.RUN_FIELDS)
    sweep = R.sweep_values(tables.render_ccm_sweep(runs, "emf", [(336, 96)]))
    head = ("sweep", "ed2022", "L336/O96", "MAE", "DLinear")
    assert sweep[head + ("plain", "")] == "0.4020 ±0.0010"
    assert sweep[head + ("k", "2")] == "0.3820 ±0.0010"
    assert sweep[head + ("val k", "")] == "2"
    assert sweep[head + ("oracle k", "")] == "3"
    assert head + ("Δ vs k=1", "") in sweep


# --------------------------------------------------------------------- the gate

def test_only_an_undeclared_change_fails_the_gate(tmp_path):
    before = _scoreboard([(PC, "MAE", 1, 10, 10, 21, "0.0268"),
                          (BEST, "MAE", 9, 0, 0, 9, "0.00391")])
    after = _scoreboard([(PC, "MAE", 1, 7, 10, 18, "0.0268"),
                         (BEST, "MAE", 8, 1, 0, 9, "0.00781")])
    tie = {"kind": "scoreboard", "comparison": "Per-channel vs Multivariate",
           "column": ["Tie", "Cells"], "reason": "ARIMA is identical by construction."}
    g = _gate(tmp_path, before, after, [tie])
    assert not g.ok
    assert g.n_expected == 2
    assert sorted(c.key["column"] for c in g.unexpected) == [
        "Challenger better", "Tie", "Wilcoxon p"]

    best = {"kind": "scoreboard", "comparison": ["Cluster (best k) vs Cluster (k=1)"],
            "reason": "The primary protocol changed."}
    g = _gate(tmp_path, before, after, [tie, best])
    assert g.ok and g.n_expected == 5


def test_no_rule_forgives_a_value_that_vanished_or_turned_into_a_dash(tmp_path):
    before = _scoreboard([(BEST, "MAE", 9, 0, 0, 9, "0.00391")])
    after = _scoreboard([(BEST, "MAE", 8, 1, 0, 0, "-")])
    everything = {"kind": "scoreboard", "reason": "The primary protocol changed."}
    g = _gate(tmp_path, before, after, [everything])
    assert not g.ok
    assert [c.key["column"] for c in g.unexpected] == ["Wilcoxon p"]
    assert g.n_expected == 3                       # the counts moved; that is declared
    g = _gate(tmp_path, before, _scoreboard([]), [everything])
    assert len(g.unexpected) == 6 and g.n_expected == 0


BATCH = "batch protocol (k chosen on the logged `sl_val/loss`)"


def _recount(rows):
    """Today's paradigm comparison re-counted: its sensitivity rows only.

    rows: (comparison, variant, W, T, L, cells, median, p).
    """
    out = ["# Paradigm Comparison", "", "## 2 · Sensitivity", "",
           "| Comparison | Variant | Metric | Challenger better | Tie | Baseline better "
           "| Cells | Median gap | Wilcoxon p |",
           "|:---|:---|:---|---:|---:|---:|---:|---:|---:|"]
    out += [f"| {c} | {v} | MAE | {w} | {t} | {l} | {n} | {m} | {p} |"
            for c, v, w, t, l, n, m, p in rows]
    return "\n".join(out) + "\n"


def _held(rules=(), **entry):
    spec = {"comparison": "Cluster (best k) vs Cluster (k=1)", "variant": BATCH,
            "models": ["DLinear", "PatchTST", "TSMixer"],
            "reason": "The snapshot's rows are the batch protocol."}
    spec.update(entry)
    return {"held_elsewhere": [spec], "expected": list(rules)}


def _gate_held(tmp_path, before, after, recount, spec):
    published = tmp_path / "2026-01-01_tables.md"
    published.write_text(before)
    cells = tmp_path / "cells.csv"
    frames.write_csv(cells, [], frames.CELL_FIELDS)
    import yaml
    path = tmp_path / "gate.yaml"
    path.write_text(yaml.safe_dump(dict(spec, snapshot=published.name)))
    asked = []

    def render_models(models):
        asked.append(models)
        return recount

    sections = [(f"s{i}", h, b) for i, (h, b) in enumerate(R.split_blocks(after))]
    g = R.gate(published, cells, sections, runs_spec=tmp_path / "none.yaml",
               gate_spec=path, render_models=None if recount is None else render_models,
               out=io.StringIO())
    return g, asked


def test_a_row_held_elsewhere_is_held_against_the_recounted_row(tmp_path):
    before = _scoreboard([(PC, "MAE", 1, 10, 10, 21, "0.0268"),
                          (BEST, "MAE", 9, 0, 0, 9, "0.00391")])
    # Today's scoreboard: another protocol over more models.
    after = _scoreboard([(PC, "MAE", 1, 10, 10, 21, "0.0268"),
                         (BEST, "MAE", 10, 1, 0, 11, "0.00195")])
    recount = _recount([("Cluster (best k) vs Cluster (k=1)", BATCH,
                         9, 0, 0, 9, "+1.00%", "0.00391")])
    g, asked = _gate_held(tmp_path, before, after, recount, _held())
    assert g.ok and not g.held["scoreboard"].changes
    assert asked == [("DLinear", "PatchTST", "TSMixer")]
    # Today's scoreboard values under the old name were never published: new.
    new = {c.key["column"]: c.current for c in g.held["scoreboard"].new}
    assert new["Cells"] == "11" and new["Wilcoxon p"] == "0.00195"
    assert any("6 of them held against" in n for n in g.held["scoreboard"].notes)

    # A recounted value that moved is a change like any other, and says where
    # it was read.
    moved = _recount([("Cluster (best k) vs Cluster (k=1)", BATCH,
                       8, 1, 0, 9, "+1.00%", "0.00781")])
    g, _ = _gate_held(tmp_path, before, after, moved, _held())
    assert sorted(c.key["column"] for c in g.unexpected) == [
        "Challenger better", "Tie", "Wilcoxon p"]
    assert "held against today's 'batch protocol" in g.unexpected[0].label()


def test_a_row_held_elsewhere_with_nothing_to_hold_it_against_fails(tmp_path):
    before = _scoreboard([(BEST, "MAE", 9, 0, 0, 9, "0.00391")])
    everything = {"kind": "scoreboard", "reason": "anything"}
    g, _ = _gate_held(tmp_path, before, before, None, _held([everything]))
    assert len(g.unexpected) == 6 and {c.current for c in g.unexpected} == {None}
    assert any("nothing to re-count" in n for n in g.notes)


def test_a_malformed_held_elsewhere_entry_is_dropped(tmp_path):
    before = _scoreboard([(BEST, "MAE", 9, 0, 0, 9, "0.00391")])
    after = _scoreboard([(BEST, "MAE", 10, 1, 0, 11, "0.00195")])
    recount = _recount([("Cluster (best k) vs Cluster (k=1)", BATCH,
                         9, 0, 0, 9, "+1.00%", "0.00391")])
    g, asked = _gate_held(tmp_path, before, after, recount, _held(models=[]))
    assert not asked and len(g.unexpected) == 4    # held under its own name again
    assert any("held_elsewhere 1" in n for n in g.notes)


def test_k_min_matches_only_the_clustered_ks():
    rule = {"kind": "sweep", "column": "k", "k_min": 2, "reason": "protocol."}
    key = dict(dataset="ED2022", window="L336/O96", metric="MAE", model="DLinear",
               column="k")
    assert R.matches(rule, R.Change("sweep", dict(key, k="2"), "0.1", "0.2"))
    assert not R.matches(rule, R.Change("sweep", dict(key, k="1"), "0.1", "0.2"))
    plain = R.Change("sweep", dict(key, column="plain", k=""), "0.1", "0.2")
    assert not R.matches(rule, plain)


def test_rules_written_for_another_snapshot_do_not_apply(tmp_path):
    before = _scoreboard([(PC, "MAE", 1, 10, 10, 21, "0.0268")])
    after = _scoreboard([(PC, "MAE", 1, 7, 10, 18, "0.0268")])
    rule = {"kind": "scoreboard", "reason": "anything"}
    g = _gate(tmp_path, before, after, [rule], snapshot="2025-01-01_tables.md")
    assert not g.ok
    assert any("none applies" in n for n in g.notes)


def test_a_malformed_rule_is_dropped_and_the_gate_stays_strict(tmp_path):
    before = _scoreboard([(PC, "MAE", 1, 10, 10, 21, "0.0268")])
    after = _scoreboard([(PC, "MAE", 1, 7, 10, 18, "0.0268")])
    rules = [{"kind": "scoreboard", "model": "DLinear", "reason": "no such field"},
             {"kind": "scoreboard"},                                # no reason
             {"kind": "figures", "reason": "no such kind"}]
    g = _gate(tmp_path, before, after, rules)
    assert not g.ok and len(g.unexpected) == 2
    assert len([n for n in g.notes if n.startswith("WARNING")]) == 3


def test_a_value_that_vanished_is_a_change_and_a_new_one_is_not(tmp_path):
    before = _scoreboard([(PC, "MAE", 1, 10, 10, 21, "0.0268"),
                          (BEST, "MAE", 9, 0, 0, 9, "0.00391")])
    after = _scoreboard([(PC, "MAE", 1, 10, 10, 21, "0.0268")],
                        extra_sensitivity=[("Per-channel vs Multivariate", "a variant")])
    g = _gate(tmp_path, before, after)
    assert {c.current for c in g.unexpected} == {None}
    assert len(g.unexpected) == 6                  # the vanished row's six values
    assert len(g.held["scoreboard"].new) == 6      # and the new row's six


def test_a_section_says_what_the_gate_held_of_it(tmp_path):
    before = _scoreboard([(PC, "MAE", 1, 10, 10, 21, "0.0268")])
    after = _scoreboard([(PC, "MAE", 1, 7, 10, 18, "0.0268")])
    rule = {"kind": "scoreboard", "column": "Tie", "reason": "declared."}
    g = _gate(tmp_path, before, after, [rule])
    assert "2 of its 6 gated values" in g.coverage("Paradigm Comparison")
    assert "1 expected, 1 unexpected" in g.coverage("Paradigm Comparison")
    assert g.coverage("CCM vs Base (No Clustering)") == (
        "text only -- none of its values is gated")


def test_a_dash_that_now_has_a_value_is_new_not_a_change(tmp_path):
    snapshot = "\n".join([
        "# Multivariate", "", "### 1.1 · MSE (`sl_test/loss`)", "",
        "| Dataset | $L$ | $O$ | DLinear (2023) | TSMixer (2023) |",
        "|:---|---:|---:|---:|---:|",
        "| ED2022 | 336 | 96 | 0.5353 | - |", ""])
    published = tmp_path / "2026-01-01_tables.md"
    published.write_text(snapshot)
    cells = tmp_path / "cells.csv"
    base = {f: "" for f in frames.CELL_FIELDS}
    frames.write_csv(cells, [
        dict(base, setup="multivariate", model="DLinear", dataset="ed2022", seq_len=336,
             pred_len=96, k="", conf=1, mse="0.53531", mae="0.3"),
        dict(base, setup="multivariate", model="TSMixer", dataset="ed2022", seq_len=336,
             pred_len=96, k="", conf=1, mse="0.54", mae="0.3")], frames.CELL_FIELDS)
    g = R.gate(published, cells, [], runs_spec=tmp_path / "none.yaml",
               gate_spec=tmp_path / "none.yaml", out=io.StringIO())
    held = g.held["cells"]
    assert g.ok and held.published == 1 and len(held.new) == 1
    assert g.coverage("Multivariate").startswith("its 1 gated values")


# --------------------------------------------------------------------- the spec

def test_the_declared_changes_are_for_the_frozen_snapshot_and_no_wider():
    from emfres import cli

    rules, notes = R.load_expected(RESULTS / "specs" / "gate.yaml", cli.PUBLISHED.name)
    assert rules and notes == []
    sweep = dict(dataset="ED2022", window="L336/O96", metric="MAE", model="DLinear")
    cells = dict(setup="cluster", model="DLinear", dataset="ED2022", window="L336/O96",
                 metric="MSE")
    must_reproduce = [
        R.Change("sweep", dict(sweep, column="k", k="1"), "0.3771", "0.3772"),
        R.Change("sweep", dict(sweep, column="plain", k=""), "0.3771", "0.3772"),
        # The Cluster section still prints the batch protocol: its runs did not change.
        R.Change("cells", dict(cells, k="2"), "0.5350", "0.5352"),
        R.Change("cells", dict(cells, setup="multivariate", k=""), "0.5353", "0.5354"),
        R.Change("scoreboard", dict(section="Paradigm Comparison", table="scoreboard",
                                    comparison="Per-channel vs Multivariate", variant="",
                                    metric="MAE", column="Wilcoxon p"), "0.0268", "0.03"),
    ]
    for change in must_reproduce:
        assert not any(R.matches(rule, change) for rule in rules), change.label()
