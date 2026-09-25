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
    spec.write_text(yaml.safe_dump({"emf": {"snapshot": snapshot or published.name,
                                            "expected": rules or []}}))
    sections = [(f"s{i}", heading, block)
                for i, (heading, block) in enumerate(R.split_blocks(current_md))]
    return R.gate(published, cells, sections, family="emf",
                  runs_spec=tmp_path / "none.yaml", gate_spec=spec, out=io.StringIO())


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


def test_a_value_printed_to_more_places_agrees_at_the_fewer():
    # Each side is known at least to its third significant digit, and the two
    # are compared at the coarser of those steps: at the snapshot's places
    # when today prints more of them, and the other way round.
    assert R.same("0.3765 ±0.0002", "0.37649 ±0.0002")
    assert R.same("+1.04%", "+1.035%")
    assert R.same("1235", "1234.6")
    assert R.same("1234.6", "1235")
    assert R.same("7.6e-06", "7.6049e-06")
    assert not R.same("1235", "1233.6")
    assert not R.same("+1.04%", "+1.034%")


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
    path.write_text(yaml.safe_dump({"emf": dict(spec, snapshot=published.name)}))
    asked = []

    def render_models(models):
        asked.append(models)
        return recount

    sections = [(f"s{i}", h, b) for i, (h, b) in enumerate(R.split_blocks(after))]
    g = R.gate(published, cells, sections, family="emf", runs_spec=tmp_path / "none.yaml",
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
    assert "; 6 of them against today's 'batch protocol" in g.coverage("Paradigm Comparison")

    # A recounted value that moved is a change like any other, and says where
    # it was read.
    moved = _recount([("Cluster (best k) vs Cluster (k=1)", BATCH,
                       8, 1, 0, 9, "+1.00%", "0.00781")])
    g, _ = _gate_held(tmp_path, before, after, moved, _held())
    assert sorted(c.key["column"] for c in g.unexpected) == [
        "Challenger better", "Tie", "Wilcoxon p"]
    assert "held against today's 'batch protocol" in g.unexpected[0].label()


def test_a_row_held_elsewhere_still_has_to_be_printed_under_its_own_name(tmp_path):
    before = _scoreboard([(BEST, "MAE", 9, 0, 0, 9, "0.00391")])
    recount = _recount([("Cluster (best k) vs Cluster (k=1)", BATCH,
                         9, 0, 0, 9, "+1.00%", "0.00391")])
    everything = {"kind": "scoreboard", "reason": "The primary protocol changed."}

    # Other numbers under the old name: a quantity the snapshot never
    # published, so new, not a change.
    other = _scoreboard([(BEST, "MAE", 10, 1, 0, 11, "0.00195")])
    g, _ = _gate_held(tmp_path, before, other, recount, _held([everything]))
    assert g.ok and len(g.held["scoreboard"].new) == 6

    # The row collapses to nothing counted: the recount still reproduces, but
    # its own row prints '-' where the snapshot printed a value.
    collapsed = _scoreboard([(BEST, "MAE", 0, 0, 0, 0, "-")]).replace("+1.00%", "-")
    g, _ = _gate_held(tmp_path, before, collapsed, recount, _held([everything]))
    assert not g.ok and g.n_expected == 0
    assert sorted(c.key["column"] for c in g.unexpected) == ["Median gap", "Wilcoxon p"]
    assert all(c.via == R.OWN_ROW and c.vanished for c in g.unexpected)
    assert "held against its own row" in g.unexpected[0].label()
    assert not any(c.key["column"] == "Wilcoxon p" for c in g.held["scoreboard"].new)

    # The row is gone: each of its six values fails, whatever a rule says.
    g, _ = _gate_held(tmp_path, before, _scoreboard([]), recount, _held([everything]))
    assert not g.ok and g.n_unexpected == 6 and g.n_expected == 0
    assert {c.current for c in g.unexpected} == {None}
    assert g.held["scoreboard"].changed == 6 and not g.held["scoreboard"].new

    # Moved at the recount and gone from its own row: one value, two changes.
    moved = _recount([("Cluster (best k) vs Cluster (k=1)", BATCH,
                       9, 0, 0, 9, "+1.00%", "0.00781")])
    g, _ = _gate_held(tmp_path, before, _scoreboard([]), moved, _held())
    assert len(g.unexpected) == 7 and g.n_unexpected == 6
    assert g.held["scoreboard"].changed == 6
    assert g.coverage("Paradigm Comparison").startswith("6 of its 6 gated values")
    assert "0 expected, 6 unexpected" in g.coverage("Paradigm Comparison")


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


def test_a_pinned_rule_forgives_only_the_value_it_declares(tmp_path):
    before = _scoreboard([(PC, "MAE", 1, 10, 10, 21, "0.0268"),
                          (PC, "MSE", 2, 8, 11, 21, "0.0182")])
    rule = {"kind": "scoreboard", "comparison": "Per-channel vs Multivariate",
            "column": ["Tie", "Cells"], "reason": "ARIMA is identical by construction.",
            "now": [{"column": "Tie", "metric": "MAE", "value": 7},
                    {"column": "Tie", "metric": "MSE", "value": "5"},
                    {"column": "Cells", "value": "18"}]}
    declared = _scoreboard([(PC, "MAE", 1, 7, 10, 18, "0.0268"),
                            (PC, "MSE", 2, 5, 11, 18, "0.0182")])
    g = _gate(tmp_path, before, declared, [rule])
    assert g.ok and g.n_expected == 4
    # The same columns, other values: Tie 10 -> 0, Cells 21 -> 5, and the
    # MSE tie at the MAE row's pinned value.
    other = _scoreboard([(PC, "MAE", 1, 0, 10, 5, "0.0268"),
                         (PC, "MSE", 2, 7, 11, 18, "0.0182")])
    g = _gate(tmp_path, before, other, [rule])
    assert sorted(c.label().split("] ")[1] for c in g.unexpected) == [
        "Cells: 21 -> 5", "Tie: 10 -> 0", "Tie: 8 -> 7"]
    assert g.n_expected == 1                       # MSE Cells 21 -> 18 is declared


def test_one_pinned_value_holds_for_everything_the_rule_matches():
    rule = {"kind": "scoreboard", "comparison": "Per-channel vs Multivariate",
            "metric": "MSE", "column": "Median gap", "now": "+1.04%", "reason": "x"}
    key = dict(section="Paradigm Comparison", table="scoreboard",
               comparison="Per-channel vs Multivariate", variant="", metric="MSE",
               column="Median gap")
    assert R.matches(rule, R.Change("scoreboard", key, "+1.03%", "+1.04%"))
    assert not R.matches(rule, R.Change("scoreboard", key, "+1.03%", "+1.05%"))
    assert not R.matches(rule, R.Change("scoreboard", dict(key, metric="MAE"),
                                        "+1.31%", "+1.04%"))
    assert not R.matches(rule, R.Change("scoreboard", key, "+1.03%", None))


def test_a_malformed_pin_drops_its_rule(tmp_path):
    before = _scoreboard([(PC, "MAE", 1, 10, 10, 21, "0.0268")])
    after = _scoreboard([(PC, "MAE", 1, 7, 10, 18, "0.0268")])
    base = {"kind": "scoreboard", "column": ["Tie", "Cells"], "reason": "declared."}
    rules = [dict(base, now=[]),                                   # pins nothing
             dict(base, now=[{"column": "Tie"}]),                  # no value
             dict(base, now=[{"model": "DLinear", "value": "7"}]), # no such field
             dict(base, now=True)]
    g = _gate(tmp_path, before, after, rules)
    assert not g.ok and len(g.unexpected) == 2
    assert len([n for n in g.notes if "`now` must be" in n]) == 4


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


def test_a_familys_rules_never_apply_to_another_family(tmp_path):
    before = _scoreboard([(PC, "MAE", 1, 10, 10, 21, "0.0268")])
    after = _scoreboard([(PC, "MAE", 1, 7, 10, 18, "0.0268")])
    published = tmp_path / "2026-01-01_tables.md"
    published.write_text(before)
    cells = tmp_path / "cells.csv"
    frames.write_csv(cells, [], frames.CELL_FIELDS)
    import yaml
    spec = tmp_path / "gate.yaml"
    rule = {"kind": "scoreboard", "reason": "anything"}
    # Written for this very snapshot name, but in the other family's block.
    spec.write_text(yaml.safe_dump({"voip": {"snapshot": published.name,
                                             "expected": [rule]}}))
    sections = [(f"s{i}", h, b) for i, (h, b) in enumerate(R.split_blocks(after))]

    def run(family):
        return R.gate(published, cells, sections, family=family,
                      runs_spec=tmp_path / "none.yaml", gate_spec=spec, out=io.StringIO())

    g = run("emf")
    assert not g.ok and g.n_expected == 0 and g.notes == []
    g = run("voip")
    assert g.ok and g.n_expected == 2 and g.family == "voip"


def test_a_list_outside_every_family_block_is_not_read(tmp_path):
    # The single-snapshot layout gate.yaml had before it kept a block per
    # family: its lists are left out, loudly, and the gate is the stricter.
    import yaml
    spec = tmp_path / "gate.yaml"
    spec.write_text(yaml.safe_dump({"snapshot": "2026-01-01_tables.md",
                                    "expected": [{"kind": "scoreboard", "reason": "x"}],
                                    "held_elsewhere": []}))
    rules, notes = R.load_expected(spec, "2026-01-01_tables.md", "emf")
    assert rules == [] and len(notes) == 1 and "outside a family block" in notes[0]
    held, notes = R.load_held_elsewhere(spec, "2026-01-01_tables.md", "emf")
    assert held == [] and len(notes) == 1 and "`held_elsewhere`" in notes[0]
    # A family with no block expects nothing, and says nothing.
    spec.write_text(yaml.safe_dump({"emf": {"snapshot": "a.md", "expected": []}}))
    assert R.load_expected(spec, "b.md", "voip") == ([], [])


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
    # Partly gated: the section says which part is not.
    assert g.coverage("Paradigm Comparison").endswith(
        "1 expected, 1 unexpected; its per-cell tables are text only")
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
    g = R.gate(published, cells, [], family="emf", runs_spec=tmp_path / "none.yaml",
               gate_spec=tmp_path / "none.yaml", out=io.StringIO())
    held = g.held["cells"]
    assert g.ok and held.published == 1 and len(held.new) == 1
    assert g.coverage("Multivariate") == (
        "its 1 gated values (MSE/MAE tables vs frames/cells.csv) all reproduce; "
        "its tables other than MSE/MAE are text only")


def test_a_cell_still_training_is_held_at_the_mean_it_printed(tmp_path):
    # "(n=…)" marks a cell whose seeds are still training. It is held at the
    # mean it printed, so it changes as they finish, rather than read as '-'.
    published = tmp_path / "2026-01-01_tables.md"
    published.write_text("\n".join([
        "# Multivariate", "", "### 1.1 · MSE (`sl_test/loss`)", "",
        "| Dataset | $L$ | $O$ | DLinear (2023) | TSMixer (2023) | PatchTST (2023) |",
        "|:---|---:|---:|---:|---:|---:|",
        "| ED2022 | 336 | 96 | 0.5353 (n=1) | **0.5100 (n=2 + partial)** | - |", ""]))

    def gate(dlinear):
        cells = tmp_path / "cells.csv"
        base = {f: "" for f in frames.CELL_FIELDS}
        frames.write_csv(cells, [
            dict(base, setup="multivariate", model=model, dataset="ed2022", seq_len=336,
                 pred_len=96, k="", conf=1, mse=mse)
            for model, mse in (("DLinear", dlinear), ("TSMixer", "0.51"),
                               ("PatchTST", "0.52"))], frames.CELL_FIELDS)
        return R.gate(published, cells, [], family="emf", runs_spec=tmp_path / "none.yaml",
                      gate_spec=tmp_path / "none.yaml", out=io.StringIO())

    g = gate("0.53531")
    held = g.held["cells"]
    assert g.ok and held.published == 2 and len(held.new) == 1      # PatchTST's '-'
    assert "2 cells the snapshot printed with (n=...)" in " ".join(held.notes)
    assert "1 cells are '-'" in " ".join(held.notes)
    g = gate("0.5360")
    assert [c.label() for c in g.unexpected] == [
        "multivariate/DLinear/ED2022 L336/O96 [MSE]: 0.5353 (n=1) -> 0.5360"]


def test_every_dataset_label_reads_back_to_its_cells_slug():
    import extract_results as er

    for slug in er.DATASET_ORDER:
        assert R.dataset_slug(er.dataset_label(slug)) == slug


def test_a_voip_cell_is_held_against_its_cells_row(tmp_path):
    # A VoIP label is not its slug lower-cased: "VoIP Mobile" is voip_mobile's
    # row, not a value that is no longer published.
    published = tmp_path / "2026-01-01_voip_tables.md"
    published.write_text("\n".join([
        "# Univariate (Per Channel)", "", "### 1.1 · MSE (`sl_test/loss`)", "",
        "| Dataset | $L$ | $O$ | Transformer (2017) | Persistence |",
        "|:---|---:|---:|---:|---:|",
        "| VoIP Mobile | 48 | 12 | 0.2948 | 0.2663 |",
        "| VoIP Fixed | 24 | 6 | 0.3451 | 0.2152 |", ""]))
    cells = tmp_path / "cells.csv"
    base = {f: "" for f in frames.CELL_FIELDS}
    frames.write_csv(cells, [
        dict(base, setup="perchannel", family="voip", model=model, dataset=ds, seq_len=sl,
             pred_len=pl, k="", conf=1, mse=mse)
        for model, ds, sl, pl, mse in (("RF_EMF_Transformer", "voip_mobile", 48, 12, "0.29481"),
                                       ("Persistence", "voip_mobile", 48, 12, "0.2663"),
                                       ("RF_EMF_Transformer", "voip_fixed", 24, 6, "0.3451"),
                                       ("Persistence", "voip_fixed", 24, 6, "0.21524"))],
        frames.CELL_FIELDS)
    g = R.gate(published, cells, [], family="voip", runs_spec=tmp_path / "none.yaml",
               gate_spec=tmp_path / "none.yaml", out=io.StringIO())
    assert g.ok and g.held["cells"].published == 4 and not g.held["cells"].changes


def test_an_archived_rule_names_a_dataset_by_its_label_or_its_slug():
    # The tables print "VoIP Mobile", cells.csv has voip_mobile: a rule written
    # either way withdraws the cell, and names no other dataset.
    for name in ("VoIP Mobile", "voip_mobile", "VOIP_MOBILE"):
        rules = [{"setups": ["cluster"], "dataset": name, "horizon": 12}]
        assert R.is_archived(rules, "cluster", "VoIP Mobile", 48, 12)
        assert not R.is_archived(rules, "cluster", "VoIP Fixed", 48, 12)
        assert not R.is_archived(rules, "cluster", "VoIP Mobile", 24, 6)
    for name in ("ED2022", "ed2022", "emf_2023_university_5_with_ptv"):
        label = "EMF2023" if name.startswith("emf") else "ED2022"
        assert R.is_archived([{"dataset": name}], "multivariate", label, 336, 96)
    assert not R.is_archived([{"dataset": "ed2022"}], "multivariate", "UH2022", 336, 96)


def _cluster_gate(tmp_path, *, batch_key="sl_test/loss", mse="0.5350", mse_fixed="0.5352",
                  rules=()):
    """A snapshot publishing one cluster k=2 MSE of 0.5350 under ``batch_key``."""
    snapshot = "\n".join([
        "# Cluster (k-Grouped Channels)", "", f"### 1.1 · MSE (`{batch_key}`)", "",
        "| Dataset | $L$ | $O$ | $k$ | DLinear (2023) |",
        "|:---|---:|---:|---:|---:|",
        "| ED2022 | 336 | 96 | 2 | 0.5350 |", ""])
    published = tmp_path / "2026-01-01_tables.md"
    published.write_text(snapshot)
    cells = tmp_path / "cells.csv"
    base = {f: "" for f in frames.CELL_FIELDS}
    frames.write_csv(cells, [dict(base, setup="cluster", model="DLinear", dataset="ed2022",
                                  seq_len=336, pred_len=96, k="2", conf=1, mse=mse,
                                  mae="0.3", mse_fixed=mse_fixed, mae_fixed="0.3")],
                     frames.CELL_FIELDS)
    import yaml
    spec = tmp_path / "gate.yaml"
    spec.write_text(yaml.safe_dump({"emf": {"snapshot": published.name,
                                            "expected": list(rules)}}))
    return R.gate(published, cells, [], family="emf", runs_spec=tmp_path / "none.yaml",
                  gate_spec=spec, out=io.StringIO())


def test_a_cluster_cell_is_held_against_its_fixed_mean_and_the_batch_one(tmp_path):
    rule = {"kind": "cells", "setup": "cluster", "k_min": 2, "reason": "fixed protocol.",
            "now": [{"model": "DLinear", "value": "0.5352"}]}
    # Printed under the batch protocol: today's primary is the fixed mean.
    g = _cluster_gate(tmp_path)
    assert not g.ok and [(c.current, c.via) for c in g.unexpected] == [("0.5352", "")]
    assert "held against today's fixed-protocol cells" in " ".join(g.held["cells"].notes)
    g = _cluster_gate(tmp_path, rules=[rule])
    assert g.ok and g.n_expected == 1
    # The batch mean it was published from still has to reproduce, and no
    # rule forgives it there -- not even one pinning the value it moved to.
    g = _cluster_gate(tmp_path, mse="0.5352", rules=[rule])
    assert not g.ok and [c.via for c in g.unexpected] == [R.BATCH_VIA]
    assert g.n_expected == 1
    # A snapshot that printed the fixed protocol is held against it alone.
    g = _cluster_gate(tmp_path, batch_key="sl_test_fixed/loss", mse="0.9",
                      mse_fixed="0.5350")
    assert g.ok and g.held["cells"].published == 1
    assert R.cluster_printed_batch(SNAPSHOT.read_text())


def _fixed_first_gate(tmp_path, *, mse="0.5350", mse_fixed="0.5352", rules=()):
    """A snapshot whose cluster section leads with the fixed protocol (k=2 MSE
    0.5352) and prints the batch one (0.5350) in its batch-protocol group, as
    extract_results renders it; one cell of each, and a '-' in the group."""
    import extract_results as er

    table = ["| Dataset | $L$ | $O$ | $k$ | DLinear (2023) | TSMixer (2023) |",
             "|:---|---:|---:|---:|---:|---:|"]
    published = tmp_path / "2026-01-01_tables.md"
    published.write_text("\n".join([
        "# Cluster (k-Grouped Channels)", "", "## 1 · Forecast accuracy", "",
        "### 1.1 · MSE (`sl_test_fixed/loss`)", "", *table,
        "| ED2022 | 336 | 96 | 2 | **0.5352** | - |", "",
        "## 4 · Model size", "", "### 4.1 · Trainable weights", "",
        "| Model | ED2022 |", "|:---|---:|", "| DLinear (2023) | 1,000 |", "",
        f"## 5 · {er.BATCH_GROUP}", "", "### 5.1 · MSE (`sl_test/loss`)", "", *table,
        "| ED2022 | 336 | 96 | 2 | **0.5350** | - |", "",
        "### 5.3 · Marginal coverage % (`sl_test/ic`)", "", *table,
        "| ED2022 | 336 | 96 | 2 | 85.00 | - |", ""]))
    cells = tmp_path / "cells.csv"
    base = {f: "" for f in frames.CELL_FIELDS}
    frames.write_csv(cells, [dict(base, setup="cluster", model="DLinear", dataset="ed2022",
                                  seq_len=336, pred_len=96, k="2", conf=1, mse=mse,
                                  mae="0.3", mse_fixed=mse_fixed, mae_fixed="0.3")],
                     frames.CELL_FIELDS)
    import yaml
    spec = tmp_path / "gate.yaml"
    spec.write_text(yaml.safe_dump({"emf": {"snapshot": published.name,
                                            "expected": list(rules)}}))
    return R.gate(published, cells, [], family="emf", runs_spec=tmp_path / "none.yaml",
                  gate_spec=spec, out=io.StringIO())


def test_a_fixed_first_snapshots_batch_protocol_group_is_held_too(tmp_path):
    # Its first table is held against the fixed mean, its batch-protocol
    # group's against the batch mean: two published values, not one.
    g = _fixed_first_gate(tmp_path)
    assert g.ok and g.held["cells"].published == 2
    assert "1 cluster cells of the snapshot's batch-protocol group" in " ".join(
        g.held["cells"].notes)
    assert [len(R.batch_group_records(p.read_text()))
            for p in tmp_path.glob("*_tables.md")] == [2]      # MSE only, '-' kept
    # A batch value that moved fails, and no rule forgives it, not even one
    # pinning the value it moved to.
    rule = {"kind": "cells", "setup": "cluster", "reason": "moved.", "now": "0.5360"}
    g = _fixed_first_gate(tmp_path, mse="0.5360", rules=[rule])
    assert not g.ok and [(c.current, c.via) for c in g.unexpected] == [
        ("0.5360", R.BATCH_GROUP_VIA)]
    assert "cluster/DLinear/ED2022 L336/O96 k2 [MSE] (held against cells.csv's " \
           "batch-protocol mse/mae" in g.unexpected[0].label()
    # Both protocols moved: two published values changed, each counted once;
    # the rule forgives the fixed one alone.
    g = _fixed_first_gate(tmp_path, mse="0.5360", mse_fixed="0.5360", rules=[rule])
    assert g.n_expected == 1 and g.n_unexpected == 1
    assert g.coverage("Cluster (k-Grouped Channels)").startswith(
        "2 of its 2 gated values (MSE/MAE tables vs frames/cells.csv) changed: "
        "1 expected, 1 unexpected")
    g = _fixed_first_gate(tmp_path, mse="0.5360", mse_fixed="0.5360")
    assert g.n_unexpected == 2
    # A snapshot that printed the batch protocol first has no such group.
    assert not R.batch_group_records(SNAPSHOT.read_text())


# --------------------------------------------------------------------- the spec

def test_the_declared_changes_are_for_the_frozen_snapshot_and_no_wider():
    from emfres import cli

    # A re-freeze retires the lists (2026-09-25: both are empty); whatever
    # rules a later sweep declares must load cleanly and stay this narrow.
    rules, notes = R.load_expected(RESULTS / "specs" / "gate.yaml", cli.PUBLISHED.name,
                                   "emf")
    assert notes == []
    held, notes = R.load_held_elsewhere(RESULTS / "specs" / "gate.yaml",
                                        cli.PUBLISHED.name, "emf")
    assert notes == [] and all(h.models for h in held)
    sweep = dict(dataset="ED2022", window="L336/O96", metric="MAE", model="DLinear")
    cells = dict(setup="cluster", model="DLinear", dataset="ED2022", window="L336/O96",
                 metric="MSE")
    board = dict(section="Paradigm Comparison", table="scoreboard", variant="", metric="MSE")
    must_reproduce = [
        # The cluster comparisons are held against the batch-protocol row over
        # the snapshot's three models: only their median moved (the median
        # over every counted cell), and only for k=1.
        R.Change("scoreboard", dict(board, comparison="Cluster (k=1) vs Multivariate",
                                    column="Cells"), "9", "11"),
        R.Change("scoreboard", dict(board, comparison="Cluster (best k) vs Cluster (k=1)",
                                    column="Wilcoxon p"), "0.00391", "0.00781"),
        R.Change("scoreboard", dict(board, comparison="Cluster (best k) vs Multivariate",
                                    column="Median gap"), "-0.73%", "-1.67%"),
        R.Change("sweep", dict(sweep, column="k", k="1"), "0.3771", "0.3772"),
        R.Change("sweep", dict(sweep, column="plain", k=""), "0.3771", "0.3772"),
        # The Cluster section's k >= 2 cells moved to their fixed means, each
        # to the one it is pinned to; no k = 1 cell moved, and none moved in
        # the batch protocol the snapshot printed it under.
        R.Change("cells", dict(cells, k="2"), "0.5350", "0.5351"),
        R.Change("cells", dict(cells, k="1"), "0.5353", "0.5354"),
        R.Change("cells", dict(cells, k="2"), "0.5350", "0.5352", R.BATCH_VIA),
        R.Change("cells", dict(cells, setup="multivariate", k=""), "0.5353", "0.5354"),
        R.Change("scoreboard", dict(section="Paradigm Comparison", table="scoreboard",
                                    comparison="Per-channel vs Multivariate", variant="",
                                    metric="MAE", column="Wilcoxon p"), "0.0268", "0.03"),
    ]
    for change in must_reproduce:
        assert not any(R.matches(rule, change) for rule in rules), change.label()


def test_the_spec_has_a_block_per_family_for_its_frozen_snapshot():
    # Each family's block names the snapshot the family is gated against, so
    # its lists apply; an unfrozen family's block has nothing to apply.
    import yaml
    from emfres import cli

    doc = yaml.safe_load((RESULTS / "specs" / "gate.yaml").read_text())
    assert set(doc) <= set(cli.WINDOWS)
    for family, block in doc.items():
        assert set(block) <= set(R.BLOCK_FIELDS), family
        snapshot = cli.SNAPSHOTS.get(family)
        if snapshot is None:
            assert not block.get("snapshot") and not block.get("expected")
            assert not block.get("held_elsewhere")
        else:
            assert snapshot.exists() and block["snapshot"] == snapshot.name


def test_each_declared_change_is_pinned_to_a_value_the_snapshot_published(tmp_path):
    from emfres import cli

    rules, _ = R.load_expected(RESULTS / "specs" / "gate.yaml", cli.PUBLISHED.name, "emf")
    assert all("now" in rule for rule in rules)

    # Every value the snapshot publishes, by kind (the cells against no frames
    # at all, so each comes back as a change carrying its key).
    empty = tmp_path / "cells.csv"
    frames.write_csv(empty, [], frames.CELL_FIELDS)
    text = SNAPSHOT.read_text()
    published = {"cells": [(c.key, c.published)
                           for c in R.hold_cells(SNAPSHOT, empty, []).changes]}
    for kind, parse in (("scoreboard", R.scoreboard_values), ("sweep", R.sweep_values)):
        published[kind] = [(dict(zip(R.KINDS[kind], k[1:])), v)
                           for k, v in parse(text).items()]

    # Each pin is the transition of a value the snapshot published: no pin
    # names a value that is not there, so none can go stale unnoticed.
    for rule in rules:
        for pin in R._pins(rule):
            hits = [key for key, was in published[rule["kind"]]
                    if R.matches(rule, R.Change(rule["kind"], key, was, pin["value"]))
                    and R._fields_match({n: v for n, v in pin.items() if n != "value"},
                                        key)]
            assert hits, (rule["_n"], pin)

    # And the same columns at any other value are not forgiven.
    board = dict(section="Paradigm Comparison", table="scoreboard",
                 comparison="Per-channel vs Multivariate", variant="")
    sweep = dict(dataset="ED2022", window="L336/O96", metric="MAE")
    others = [
        R.Change("scoreboard", dict(board, metric="MAE", column="Tie"), "10", "0"),
        R.Change("scoreboard", dict(board, metric="MSE", column="Tie"), "8", "7"),
        R.Change("scoreboard", dict(board, metric="MSE", column="Cells"), "21", "5"),
        R.Change("scoreboard", dict(board, metric="MAE", column="Median gap"),
                 "+1.31%", "+1.04%"),
        R.Change("scoreboard", dict(board, comparison="Cluster (k=1) vs Multivariate",
                                    metric="MAE", column="Median gap"), "+0.09%", "+0.50%"),
        R.Change("sweep", dict(sweep, model="DLinear", column="k", k="2"),
                 "0.3768 ±0.0002", "0.3780 ±0.0001"),
        R.Change("sweep", dict(sweep, model="PatchTST", column="oracle k", k=""), "9", "5"),
        R.Change("cells", dict(setup="perchannel", model="EMForecaster", dataset="UH2022",
                               window="L336/O96", k="", metric="MSE"), "0.8316", "0.8320"),
        R.Change("cells", dict(setup="perchannel", model="EMForecaster", dataset="ED2022",
                               window="L336/O96", k="", metric="MSE"), "0.4879", "0.4890"),
        R.Change("cells", dict(setup="cluster", model="TSMixer", dataset="UH2022",
                               window="L336/O96", k="9", metric="MSE"), "1.2268", "1.1588"),
        R.Change("cells", dict(setup="cluster", model="DLinear", dataset="EMF2023",
                               window="L336/O96", k="2", metric="MAE"), "0.5030", "0.5031"),
    ]
    for change in others:
        assert not any(R.matches(rule, change) for rule in rules), change.label()


# --------------------------------------------------------------------- the command

def _voip_frames(path):
    """frames/ for one VoIP cell per paradigm (multivariate, per-channel), 3 seeds."""
    rows = []
    for seed in (1, 2, 3):
        for setup, mae in (("multivariate", 0.40), ("perchannel", 0.30)):
            row = {f: "" for f in frames.RUN_FIELDS}
            row.update(setup=setup, family="voip", model="DLinear", dataset="voip_mobile",
                       seq_len=48, pred_len=12, k="", seed=str(seed), conf=1,
                       channel="0" if setup == "perchannel" else "",
                       mae=repr(mae), mse=repr(mae * 1.5))
            rows.append(row)
    frames.write_csv(path / "runs.csv", rows, frames.RUN_FIELDS)
    frames.write_csv(path / "cells.csv", frames.build_cells(rows), frames.CELL_FIELDS)
    frames.write_csv(path / "bands.csv", [], frames.BAND_FIELDS)


def test_reconcile_gates_every_frozen_family_and_names_the_rest(tmp_path, monkeypatch,
                                                                capsys):
    import csv

    import yaml
    from emfres import cli

    _voip_frames(tmp_path)
    monkeypatch.setattr(cli, "FRAMES", tmp_path)
    specs = tmp_path / "specs"
    specs.mkdir()
    monkeypatch.setattr(cli, "SPECS", specs)
    # A VoIP snapshot frozen the way specs/gate.yaml describes: today's
    # render, named in SNAPSHOTS and in the family's block.
    snapshot = tmp_path / "2026-01-01_voip_tables.md"
    snapshot.write_text("\n\n".join(md.strip() for _, _, md in cli.render_family("voip")))
    (specs / "gate.yaml").write_text(yaml.safe_dump(
        {"voip": {"snapshot": snapshot.name, "held_elsewhere": [], "expected": []}}))
    monkeypatch.setattr(cli, "SNAPSHOTS", {"emf": None, "voip": snapshot})
    capsys.readouterr()

    assert cli.main(["reconcile"]) == 0
    out = capsys.readouterr().out
    assert "value gate (emf): no frozen snapshot yet" in out
    assert "value gate (voip): 2026-01-01_voip_tables.md against today's frames" in out
    assert out.rstrip().endswith(
        "reconcile: emf not gated (no frozen snapshot yet); voip PASS")
    assert "cells         4 published, 4 reproduce" in out   # MV and PC, MSE and MAE

    # A VoIP value that moved fails the run. The family with no snapshot
    # still does not: it is named, not gated.
    with open(tmp_path / "cells.csv") as f:
        cells = list(csv.DictReader(f))
    next(c for c in cells if c["setup"] == "multivariate")["mse"] = "0.7"
    frames.write_csv(tmp_path / "cells.csv", cells, frames.CELL_FIELDS)
    assert cli.main(["reconcile", "--all"]) == 1
    out = capsys.readouterr().out
    assert "multivariate/DLinear/VoIP Mobile L48/O12 [MSE]: 0.6000 -> 0.7000" in out
    assert out.rstrip().endswith("voip FAIL (1 unexpected changes)")

    # Asked for by name, a family with no snapshot is an error, not a pass;
    # --tables is one family's snapshot; and a frozen snapshot that is gone fails.
    assert cli.main(["reconcile", "--family", "emf"]) == 1
    assert "no frozen snapshot for family 'emf' yet" in capsys.readouterr().err
    assert cli.main(["reconcile", "--tables", str(snapshot)]) == 1
    assert "pass --family with it" in capsys.readouterr().err
    monkeypatch.setattr(cli, "SNAPSHOTS", {"emf": None, "voip": tmp_path / "gone.md"})
    assert cli.main(["reconcile"]) == 1
    assert capsys.readouterr().out.rstrip().endswith("voip FAIL (gone.md is missing)")
    monkeypatch.setattr(cli, "SNAPSHOTS", {"emf": None, "voip": None})
    assert cli.main(["reconcile"]) == 1
    assert "nothing was gated" in capsys.readouterr().err


def test_tables_tells_no_snapshot_from_a_missing_one(tmp_path, monkeypatch, capsys):
    # No snapshot yet renders ungated; a frozen one whose file is gone fails
    # --check and refuses to write, as reconcile fails on it.
    from emfres import cli

    _voip_frames(tmp_path)
    monkeypatch.setattr(cli, "FRAMES", tmp_path)
    monkeypatch.setattr(cli, "SPECS", tmp_path)
    monkeypatch.setattr(cli, "SNAPSHOTS", {"emf": None, "voip": None})
    assert cli.main(["tables", "--family", "voip", "--check"]) == 0
    assert "family 'voip' has no frozen snapshot" in capsys.readouterr().out

    monkeypatch.setattr(cli, "SNAPSHOTS", {"emf": None, "voip": tmp_path / "gone.md"})
    assert cli.main(["tables", "--family", "voip", "--check"]) == 1
    out = capsys.readouterr().out
    assert "frozen snapshot gone.md is missing" in out and "has no frozen" not in out
    # Write mode refuses before it writes anything (outdir would be tables/voip/).
    assert cli.main(["tables", "--family", "voip"]) == 1
    assert "refusing to write" in capsys.readouterr().err


def test_every_family_has_a_snapshot_entry():
    # None until one is frozen; a family missing here would never be gated
    # and never be named as ungated either.
    from emfres import cli

    assert list(cli.SNAPSHOTS) == list(cli.WINDOWS)
    assert cli.SNAPSHOTS["emf"] == cli.PUBLISHED
