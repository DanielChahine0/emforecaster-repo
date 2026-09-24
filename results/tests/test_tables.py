"""The cluster sweep table: every dataset and window, k chosen on validation.

The sweep used to underline the k with the best *test* value beside prose
saying k was chosen on validation. These pin that the underlined k is the
validation one -- the same one the paradigm comparison names -- and that a k
the test split would have picked is printed as an oracle, never selected.
"""

from pathlib import Path

from emfres import frames, tables


def _run(setup, seed, mae, *, model="DLinear", dataset="ed2022", k="", val=None,
         seq_len=336, pred_len=96, family="emf"):
    row = {f: "" for f in frames.RUN_FIELDS}
    row.update(setup=setup, family=family, model=model, dataset=dataset, seq_len=seq_len,
               pred_len=pred_len, k=k, seed=str(seed), conf=1, mae=repr(mae),
               mse=repr(mae * 1.5))
    if setup == "cluster":
        row.update(mae_fixed=repr(mae), mse_fixed=repr(mae * 1.5), val_loss=repr(val),
                   val_loss_fixed=repr(val), fixed_source="sidecar")
    return row


SEEDS = (1, 2, 3)


def _sweep(tmp_path, *, dataset="ed2022", window=(336, 96), family="emf",
           extra=()) -> Path:
    rows = list(extra)
    for s in SEEDS:
        rows.append(_run("multivariate", s, 0.40 + 0.001 * s, dataset=dataset,
                         seq_len=window[0], pred_len=window[1], family=family))
        # k=2 validates best, k=3 tests best.
        for k, mae, val in ((1, 0.40, 0.5), (2, 0.38, 0.3), (3, 0.36, 0.4)):
            rows.append(_run("cluster", s, mae + 0.001 * s, k=k, val=val, dataset=dataset,
                             seq_len=window[0], pred_len=window[1], family=family))
    path = tmp_path / "runs.csv"
    frames.write_csv(path, rows, frames.RUN_FIELDS)
    return path


def test_the_validation_k_is_underlined_and_the_test_k_is_an_oracle(tmp_path):
    md = tables.render_ccm_sweep(_sweep(tmp_path), "emf", [(336, 96)])
    assert md.startswith(f"# {tables.SWEEP_TITLE}\n")
    row = next(line for line in md.splitlines() if line.startswith("| DLinear"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    # Model, Plain, k=1, k=2, k=3, Val k, Δ vs k=1, Δ vs plain, Oracle k
    assert cells[3].startswith("<u>") and not cells[4].startswith("<u>")
    assert cells[5] == "k=2, sidecar"
    assert cells[8] == "3 (oracle)"


def test_every_dataset_and_window_gets_a_table(tmp_path):
    other = []
    for s in SEEDS:
        for k in (1, 2, 3):
            other.append(_run("cluster", s, 0.3, k=k, val=0.4, dataset="voip_fixed",
                              seq_len=24, pred_len=6, family="voip"))
    runs = _sweep(tmp_path, dataset="voip_mobile", window=(48, 12), family="voip",
                  extra=other)
    md = tables.render_ccm_sweep(runs, "voip", [(48, 12), (24, 6)])
    assert "voip_mobile, L48/O12" in md and "voip_fixed, L24/O6" in md
    assert "k = C is one cluster per channel" in md
    assert "upstream wrappers" not in md                  # an EMF-only history
    assert tables.render_ccm_sweep(runs, "emf", [(336, 96)]) == ""


def test_a_k_still_training_shows_its_n_and_is_not_ranked(tmp_path):
    extra = [_run("cluster", 1, 0.10, k=5, val=0.01)]      # one seed so far
    md = tables.render_ccm_sweep(_sweep(tmp_path, extra=extra), "emf", [(336, 96)])
    row = next(line for line in md.splitlines() if line.startswith("| DLinear"))
    assert "0.1000 (n=1)" in row
    assert "k=2, sidecar; ranked 1,2,3 of 1,2,3,5" in row


def test_per_channel_times_are_labelled_contended_per_band():
    import extract_results as er
    from emfres.runlogs import Run

    def run(channel, concurrency):
        return Run("DLinear", "ed2022", 336, 96, True, 0.5, 0.4, 60.0, 1.0, 10, None, None,
                   None, channel, None, "1", Path("x"), concurrency=concurrency)

    pc = [run(0, 8), run(1, 12), run(2, None)]
    assert er.concurrency_text(pc) == "8 or 12 trainings at a time; 1 runs logged before " \
                                      "that was stamped"
    assert "per band and contended" in er.per_band_note(pc)
    assert er.per_band_note([run(None, 1)]) == ""          # a joint tree: trained alone


def test_the_series_section_lists_its_trivial_forecasts(tmp_path):
    import extract_results as er

    rows = []
    for seed, mse in ((1, 0.6), (2, 1.1619)):
        row = {f: "" for f in frames.RUN_FIELDS}
        row.update(setup="series", family="emf", model="RF_EMF_Transformer", dataset="ptv2",
                   seq_len=336, pred_len=96, seed=str(seed), conf=1, mse=repr(mse),
                   mae="0.5", lookback_mse="1.3363", train_mean_mse="1.1619")
        rows.append(row)
    path = tmp_path / "runs.csv"
    frames.write_csv(path, rows, frames.RUN_FIELDS)
    ((group, _, lines, note),) = tables.trivial_forecasts(path, "series", "emf", er)
    assert group == "Trivial forecasts" and "RevIN" in note
    assert lines[-1] == "| PTV2 | 336 | 96 | 1.3363 | 1.1619 |"
    assert tables.trivial_forecasts(path, "perchannel", "emf", er) == []


# --------------------------------------------------------------- partial cells

def _er_run(model, seed, mse, *, k=None, channel=None, dataset="ed2022"):
    from emfres.runlogs import Run
    return Run(model, dataset, 336, 96, True, mse, mse * 0.7, 60.0, 1.0, 10, None, None,
               None, channel, k, str(seed), Path("x"))


def _trees():
    """Three seeds of plain DLinear and TSMixer, a finished DLinear sweep, and a
    TSMixer sweep that has one seed of k=1 so far -- with the best numbers."""
    import extract_results as er
    mv = [_er_run(m, s, v + 0.001 * s) for s in SEEDS
          for m, v in (("DLinear", 0.50), ("TSMixer", 0.60))]
    cl = [_er_run("DLinear", s, v + 0.001 * s, k=k) for s in SEEDS
          for k, v in ((1, 0.50), (2, 0.49))]
    cl.append(_er_run("TSMixer", 1, 0.30, k=1))
    return er, mv, cl


def test_a_cell_short_of_its_seeds_shows_n_and_is_never_ranked():
    er, mv, cl = _trees()
    mv_cells, cl_cells = er.aggregate(mv), er.aggregate(cl)
    targets = er.seed_targets(mv_cells, cl_cells)
    assert targets[("ed2022", 336, 96, True)] == 3
    key = ("TSMixer", "ed2022", 336, 96, True, 1)
    assert er.shortfall(cl_cells[key], key, targets) == " (n=1)"
    full = ("DLinear", "ed2022", 336, 96, True, 1)
    assert er.shortfall(cl_cells[full], full, targets) == ""

    md = er.render_report(er.CLUSTER_TITLE, cl, cl_cells, targets=targets)
    row = next(line for line in md.splitlines() if line.startswith("| ED2022 | 336 | 96 | 1 |"))
    # TSMixer's 0.3000 is the lowest, but one seed of three: shown, not bold.
    assert "| **0.5020** | 0.3000 (n=1) |" in row
    assert "`(n=…)`" in md
    assert "- Protocol: the logged batch protocol" in md    # which cluster numbers these are
    # A complete tree renders exactly as before: no marker, no legend.
    plain = er.render_report(er.MULTIVARIATE_TITLE, mv, mv_cells, targets=targets)
    assert "(n=" not in plain and "Protocol" not in plain


def _fixed(run, mse_fixed, *, source="sidecar"):
    """A cluster run with fixed-membership metrics (MAE at the same 0.7 ratio)."""
    from dataclasses import replace
    return replace(run, mse_fixed=mse_fixed, mae_fixed=mse_fixed * 0.7,
                   val_loss_fixed=mse_fixed, ic_fixed=90.0, jc_fixed=50.0,
                   interval_width_fixed=1.0, fixed_source=source, ic=80.0, jc=40.0,
                   interval_width=2.0)


def _section_tables(md):
    """{table heading: [rows]} of a rendered section, rows as lists of cells."""
    out, cur = {}, None
    for line in md.splitlines():
        if line.startswith("### "):
            cur = line.split(" · ", 1)[1]
            out[cur] = []
        elif line.startswith("## "):
            cur = None
        elif cur and line.startswith("|") and not line.startswith("|:"):
            out[cur].append([c.strip() for c in line.strip("|").split("|")])
    return out


def test_a_cluster_section_leads_with_the_fixed_protocol_the_batch_one_last():
    import extract_results as er
    # k=2's fixed forecast is worse than its batch one, TSMixer's the other way.
    cl = [_fixed(_er_run("DLinear", s, v, k=k), fx) for s in SEEDS
          for k, v, fx in ((1, 0.50, 0.50), (2, 0.49, 0.51))]
    cl += [_fixed(_er_run("TSMixer", s, v, k=k), fx) for s in SEEDS
           for k, v, fx in ((1, 0.60, 0.60), (2, 0.52, 0.47))]
    md = er.render_setup(er.CLUSTER_TITLE, cl, er.aggregate(cl))
    assert "- Protocol: the fixed train-mean membership (`sl_test_fixed/*`)" in md
    t = _section_tables(md)
    # Primary: the fixed numbers, ranked on them.
    assert t["MSE (`sl_test_fixed/loss`)"][2][4:] == ["<u>0.5100</u>", "**0.4700**"]
    assert t["MAE (`sl_test_fixed/mae`)"][2][4:] == ["<u>0.3570</u>", "**0.3290**"]
    assert t["Marginal coverage % (`sl_test_fixed/ic`)"][1][4:] == ["90.00", "90.00"]
    # The last group: the same runs' logged batch numbers, labelled.
    last = md.split("\n## ")[-1]
    assert last.startswith("5 · Batch protocol (sensitivity)")
    assert t["MSE (`sl_test/loss`)"][2][4:] == ["**0.4900**", "<u>0.5200</u>"]
    assert t["Marginal coverage % (`sl_test/ic`)"][1][4:] == ["80.00", "80.00"]
    # At k = 1 the two protocols print the same row.
    assert t["MSE (`sl_test/loss`)"][1] == t["MSE (`sl_test_fixed/loss`)"][1]
    # Cost and size are the runs' own, not a protocol's.
    assert "Total training time (`parameters/running_time`)" in t


def test_a_cluster_run_without_fixed_metrics_blanks_only_its_fixed_cells(capsys):
    import extract_results as er
    from dataclasses import replace
    cl = [_fixed(_er_run("DLinear", s, 0.5, k=k), 0.6) for s in SEEDS for k in (1, 2)]
    # seed 3, k=2: fixed_source none, and the slowest run of its cell
    cl[-1] = replace(cl[-1], mse_fixed=None, mae_fixed=None, running_time=120.0)
    cells = er.aggregate(cl)
    md = er.render_setup(er.CLUSTER_TITLE, cl, cells, targets=er.seed_targets(cells))
    t = _section_tables(md)
    # All or nothing, as cells.csv: '-' (not logged), never a 2-seed mean
    # passed off as a cell still training.
    assert t["MSE (`sl_test_fixed/loss`)"][2][4] == er.MISSING
    assert t["MSE (`sl_test_fixed/loss`)"][1][4] == "**0.6000**"
    assert t["MSE (`sl_test/loss`)"][2][4] == "**0.5000**"
    # Time has no protocol: the cost tables keep every run, unmarked.
    assert t["Total training time (`parameters/running_time`)"][2][4] == "**1m 20s**"
    assert "(n=" not in md
    assert "1 runs of 'Cluster (k-Grouped Channels)' have no fixed" in capsys.readouterr().err
    primary, batch = er.protocol_cells(cl, cells)
    key = ("DLinear", "ed2022", 336, 96, True, 2)
    assert (primary[key].mse, primary[key].running_time) == (None, 80.0)
    assert batch is cells

    # Frames written before the fixed columns: the batch protocol, labelled so.
    bare = [_er_run("DLinear", s, 0.5, k=k) for s in SEEDS for k in (1, 2)]
    md = er.render_setup(er.CLUSTER_TITLE, bare, er.aggregate(bare))
    assert "- Protocol: the logged batch protocol" in md and "### 1.1 · MSE (`sl_test/loss`)" in md
    assert er.BATCH_GROUP not in md
    assert "no run of 'Cluster (k-Grouped Channels)' has fixed" in capsys.readouterr().err
    # Any other tree is untouched.
    mv = [_er_run("DLinear", s, 0.5) for s in SEEDS]
    assert er.render_setup(er.MULTIVARIATE_TITLE, mv, er.aggregate(mv)) == \
        er.render_report(er.MULTIVARIATE_TITLE, mv, er.aggregate(mv))


def test_a_per_channel_seed_missing_a_band_is_marked_partial():
    import extract_results as er
    runs = [_er_run("DLinear", s, 0.5, channel=ch) for s in SEEDS for ch in (0, 1)
            if (s, ch) != (3, 1)]
    cells = er.aggregate(runs)
    key = ("DLinear", "ed2022", 336, 96, True, None)
    assert (cells[key].n_seeds, cells[key].n_complete_seeds) == (3, 2)
    assert er.shortfall(cells[key], key, er.seed_targets(cells)) == " (n=2 + partial)"


def _ccm_trees():
    """Plain DLinear and TSMixer at three seeds; a finished DLinear sweep whose
    fixed protocol validates best at k=2 and tests best at k=3, and whose
    batch protocol validates best at k=3; a TSMixer sweep with one seed of
    k=1 so far, with the best numbers."""
    from dataclasses import replace
    import extract_results as er
    mv = [_er_run(m, s, v + 0.001 * s) for s in SEEDS
          for m, v in (("DLinear", 0.50), ("TSMixer", 0.60))]
    cl = []
    for s in SEEDS:
        for k, fixed, fval, batch, bval in ((1, 0.50, 0.5, 0.50, 0.5), (2, 0.49, 0.3, 0.495, 0.4),
                                            (3, 0.45, 0.4, 0.48, 0.2)):
            run = _fixed(_er_run("DLinear", s, batch + 0.001 * s, k=k), fixed + 0.001 * s)
            cl.append(replace(run, val_loss=bval, val_loss_fixed=fval))
    cl.append(replace(_fixed(_er_run("TSMixer", 1, 0.30, k=1), 0.30), val_loss=0.1,
                      val_loss_fixed=0.1))
    return er, mv, cl


def _ccm_md(er, mv, cl, **kw):
    primary, batch = er.protocol_cells(cl, er.aggregate(cl))
    return er.render_ccm_vs_base(er.CCM_TITLE, er.canonical_cells(primary),
                                 er.canonical_cells(er.aggregate(mv)), cl,
                                 batch=er.canonical_cells(batch), **kw)


def test_ccm_vs_base_takes_the_k_validation_chose_never_the_test_one():
    er, mv, cl = _ccm_trees()
    md = _ccm_md(er, mv, cl)
    t = _section_tables(md)
    fixed = next(v for h, v in t.items() if h.startswith("Long-horizon"))
    batch = next(v for h, v in t.items() if h.startswith("Batch protocol (sensitivity)"))
    # Dataset, Metric, DLinear, + CCM, TSMixer, + CCM, IMP
    mse = fixed[1]
    assert mse[3] == "<u>**0.4920 (k=2)**</u>"              # validation's k, not k=3's 0.4520
    assert mse[5] == "0.3000 (k=1; partial: ranked 1 of 1,2,3, 1/3 seeds)"
    assert "**" not in mse[4] and "<u>" not in mse[5]
    assert mse[6] == "+1.99 (1 of 2)"                        # DLinear's pair alone
    assert batch[1][3] == "<u>**0.4820 (k=3)**</u>"          # the batch protocol's own k
    config = t["Experiment configuration"]
    rows = {r[0].split(" (")[0].split(",")[0]: r[1:] for r in config}
    assert rows["k chosen on validation"] == ["2", "1 (partial)"]
    assert rows["k chosen on validation under the batch protocol"] == ["3", "1 (partial)"]
    assert rows["Oracle k"] == ["3", "1 (partial)"]          # reported, not used
    assert "- Protocol: the fixed train-mean membership (`sl_test_fixed/*`)" in md
    # Passing the same choices changes nothing: make.py tables passes the
    # paradigm comparison's own.
    choices = er.validation_choices(cl, "fixed", (336, 96), 3)
    assert _ccm_md(er, mv, cl, choices=choices) == md


def test_ccm_vs_base_leaves_a_pair_that_prints_the_same_unbolded():
    import extract_results as er
    from dataclasses import replace
    mv = [_er_run("DLinear", s, 0.5020) for s in SEEDS]
    # k=2 validates best, and its fixed MSE rounds to the base's 0.5020.
    cl = [replace(_fixed(_er_run("DLinear", s, 0.49, k=k), fx), val_loss=v, val_loss_fixed=v)
          for s in SEEDS for k, fx, v in ((1, 0.5100, 0.5), (2, 0.50203, 0.4))]
    t = _section_tables(_ccm_md(er, mv, cl))
    fixed = next(v for h, v in t.items() if h.startswith("Long-horizon"))
    assert fixed[1][2:4] == ["<u>0.5020</u>", "0.5020 (k=2)"]      # a tie: neither bold
    assert fixed[1][4] == "-0.01"                                 # IMP keeps the true gap
    assert fixed[2][2:4] == ["<u>0.3514</u>", "0.3514 (k=2)"]      # MAE, the same tie


def test_the_cross_tables_take_the_paradigm_comparisons_k(tmp_path):
    import extract_results as er
    from emfres import paradigms as PG

    runs_csv = _sweep(tmp_path, extra=[_run("cluster", 1, 0.10, k=5, val=0.01)])
    board = tables.validation_choices(runs_csv, "emf", er)
    cluster = tables.runs_from_csv(runs_csv, tmp_path, "emf")["cluster"]
    for protocol in PG.PROTOCOLS:
        helper = er.validation_choices(cluster, protocol, (336, 96), 3)
        assert {c: (ch.k, ch.ranked, ch.partial) for c, ch in board[protocol].items()} == \
            {c: (ch.k, ch.ranked, ch.partial) for c, ch in helper.items()}
        assert board[protocol][("DLinear (2023)", "ed2022")].k == 2    # validates best


def test_the_index_says_which_columns_of_the_cluster_twin_lead(tmp_path):
    runs = [_run("multivariate", s, 0.4) for s in SEEDS]
    runs += [_run("cluster", s, 0.4, k=2, val=0.3) for s in SEEDS]
    frames.write_csv(tmp_path / "cells.csv", frames.build_cells(runs), frames.CELL_FIELDS)
    twins = {p.stem: p for p in tables.cells_csv_twins(tmp_path / "cells.csv",
                                                       tmp_path / "csv")}
    assert "`*_fixed` columns" in tables.twin_note(twins["cluster"])
    assert "batch protocol" in tables.twin_note(twins["cluster"])
    assert tables.twin_note(twins["multivariate"]) == ""
    # A twin of frames written before the fixed columns: nothing to gloss.
    old = tmp_path / "old" / "cluster.csv"
    old.parent.mkdir()
    old.write_text("setup,model,mse,mae\ncluster,DLinear,0.5,0.4\n")
    assert tables.twin_note(old) == ""


def test_the_ultimate_table_does_not_rank_a_cell_still_training():
    er, mv, cl = _trees()
    pipes = er.build_pipelines({er.MULTIVARIATE_TITLE: er.aggregate(mv),
                                er.CLUSTER_TITLE: er.aggregate(cl)})
    md = er.render_ultimate(er.ULTIMATE_TITLE, pipes)
    row = next(line for line in md.splitlines() if "| Cluster (k=1) |" in line)
    assert "| **0.5020** | 0.3000 (n=1) |" in row


def test_the_ultimate_tables_cluster_rows_are_fixed_the_batch_ones_after_the_time():
    er, mv, cl = _ccm_trees()
    cells = {er.MULTIVARIATE_TITLE: er.aggregate(mv)}
    cells[er.CLUSTER_TITLE], batch = er.protocol_cells(cl, er.aggregate(cl))
    md = er.render_ultimate(er.ULTIMATE_TITLE, er.build_pipelines(cells),
                            batch=er.cluster_pipelines(batch))
    headings = [line for line in md.splitlines() if line.startswith("### ")]
    # 1.3 stays the time table (.personal/RESULTS.md includes pipelines.md#1.3).
    assert [h.split(" · ", 1)[0] for h in headings] == ["### 1.1", "### 1.2", "### 1.3",
                                                         "### 1.4", "### 1.5"]
    assert "Total training time" in headings[2] and "Batch protocol" in headings[3]
    t = _section_tables(md)
    fixed = t["MSE (`sl_test/loss`; cluster rows `sl_test_fixed/loss`)"]
    row = next(r for r in fixed if r[1] == "Cluster (k=3)")
    assert row[2] == "**0.4520**"                                # DLinear's fixed k=3
    assert next(r for r in fixed if r[1] == "Multivariate")[2] == "**0.5020**"
    batch_rows = t["Batch protocol (sensitivity): MSE of the cluster rows (`sl_test/loss`)"]
    assert [r[1] for r in batch_rows[1:]] == ["Cluster (k=1)", "Cluster (k=2)",
                                              "Cluster (k=3)"]      # [0] is the header
    assert next(r for r in batch_rows if r[1] == "Cluster (k=3)")[2] == "**0.4820**"
    assert "- Cluster protocol: the fixed train-mean membership" in md


def test_the_mv_vs_univariate_scoreboard_counts_a_short_pair_as_pending():
    import extract_results as er
    mv = [_er_run(m, s, 0.5) for s in SEEDS for m in ("DLinear", "TSMixer")]
    pc = [_er_run("DLinear", s, 0.4, channel=0) for s in SEEDS]
    pc.append(_er_run("TSMixer", 1, 0.3, channel=0))
    md = er.render_comparison(er.COMPARISON_TITLE, er.MULTIVARIATE_TITLE, er.aggregate(mv),
                              [(er.PER_CHANNEL_TITLE, er.aggregate(pc))])
    assert "| ED2022 | 336 | 96 | Transformer" not in md
    row = next(line for line in md.splitlines() if line.startswith("| ED2022 | 336 | 96 | TSMixer"))
    assert "0.3000 (n=1)" in row and "**" not in row
    board = next(line for line in md.splitlines() if "| MSE |" in line and "Per-channel" in line)
    assert board.endswith("| 0 | 1 | 0 | 1 |")          # DLinear won, TSMixer pending


def test_make_tables_warns_when_bands_csv_is_missing(tmp_path, monkeypatch, capsys):
    from emfres import cli

    rows = []
    for s in SEEDS:
        for setup, v in (("multivariate", 0.40), ("perchannel", 0.30)):
            row = _run(setup, s, v, dataset="voip_mobile", seq_len=48, pred_len=12,
                       family="voip")
            row["channel"] = "0" if setup == "perchannel" else ""
            rows.append(row)
    frames.write_csv(tmp_path / "runs.csv", rows, frames.RUN_FIELDS)
    monkeypatch.setattr(cli, "FRAMES", tmp_path)
    assert cli.main(["tables", "--family", "voip", "--check"]) == 0
    assert "bands.csv not found" in capsys.readouterr().err

    frames.write_csv(tmp_path / "bands.csv", [], frames.BAND_FIELDS)
    assert cli.main(["tables", "--family", "voip", "--check"]) == 0
    assert "bands.csv not found" not in capsys.readouterr().err
