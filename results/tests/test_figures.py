"""``make.py figures`` renders every figure from partial frames.

The frames the figures read change under a running sweep: a model's cluster
sweep has one run, a per-channel seed misses a band, a dataset the benchmark
never published turns up. Every figure must still render, from exactly the
frames it is given, with no hidden dataset in a path and no plotted mean that
disagrees with cells.csv.
"""

import pytest

from emfres import figures, frames

from test_paradigms import SEEDS, _band, _run


def _timed(row, seconds=60):
    row["running_time"] = str(seconds)
    return row


def _partial_frames(tmp_path):
    rows, bands, timing = [], [], []
    for ds in ("ed2022", "uh2022"):
        for s, v in zip(SEEDS, (0.40, 0.41, 0.42)):
            for model, off in (("DLinear", 0.0), ("LSTM", 0.05), ("ARIMA", 0.2)):
                rows.append(_timed(_run("multivariate", s, v + off, model=model, dataset=ds,
                                        lookback=1.0)))
                bands += [_band("multivariate", s, ch, v + off + 0.1 * ch, model=model,
                                dataset=ds) for ch in (0, 1)]
                for ch in (0, 1):
                    rows.append(_run("perchannel", s, v + off + 0.01, model=model, dataset=ds,
                                     channel=ch, lookback=1.0))
                    bands.append(_band("perchannel", s, ch, v + off + 0.1 * ch + 0.01,
                                       model=model, dataset=ds))
            # The Transformer: one seed trains, two forecast the lookback mean.
            rows.append(_timed(_run("multivariate", s, 0.4 if s == SEEDS[0] else 0.6,
                                    model="RF_EMF_Transformer", dataset=ds,
                                    mse=0.5 if s == SEEDS[0] else 1.0, lookback=1.0)))
            for k, dv, val in ((1, 0.001, 0.5), (2, -0.01, 0.3), (3, -0.02, 0.4)):
                rows.append(_timed(_run("cluster", s, v + dv, k=k, val_loss=val, dataset=ds,
                                        lookback=1.0), 90))
                bands += [_band("cluster", s, ch, v + dv + 0.1 * ch, dataset=ds, k=k,
                                mse_fixed=v + dv + 0.1 * ch) for ch in (0, 1)]
            timing.append({"family": "emf", "model": "DLinear", "dataset": ds, "seq_len": 336,
                           "pred_len": 96, "seed": str(s), "n_bands": 2, "epochs": 20,
                           "solo_bands": 2, "solo_per_epoch": 30, "solo_total": 600,
                           "solo_lo": 580, "solo_hi": 620})
        # LSTM's sweep has just begun.
        rows.append(_timed(_run("cluster", SEEDS[0], 0.46, k=1, val_loss=0.5, model="LSTM",
                                dataset=ds, source="log", lookback=1.0)))
    # A campaign the benchmark never published.
    rows += [_run("multivariate", s, 0.4, dataset="ed2024") for s in SEEDS]
    # The single-series grid.
    rows += [_run("series", s, 0.3, dataset="det") for s in SEEDS]
    # VoIP at both windows, with its carry-forward reference.
    for seq_len, pred_len in ((48, 12), (24, 6)):
        for ds in ("voip_mobile", "voip_fixed"):
            for s in SEEDS:
                kw = dict(dataset=ds, family="voip", seq_len=seq_len, pred_len=pred_len,
                          lookback=1.0)
                rows.append(_run("multivariate", s, 0.30, **kw))
                rows.append(_run("multivariate", s, 0.28, model="Persistence", **kw))
                rows += [_run("perchannel", s, 0.29, channel=ch, **kw) for ch in (0, 1)]
                rows += [_run("cluster", s, 0.30 - 0.001 * k, k=k, val_loss=0.5 - 0.01 * k, **kw)
                         for k in (1, 2)]
    frames.write_csv(tmp_path / "runs.csv", rows, frames.RUN_FIELDS)
    frames.write_csv(tmp_path / "cells.csv", frames.build_cells(rows), frames.CELL_FIELDS)
    frames.write_csv(tmp_path / "bands.csv", bands, frames.BAND_FIELDS)
    frames.write_csv(tmp_path / "timing.csv", timing, frames.TIMING_FIELDS)
    return tmp_path


def test_every_figure_renders_from_partial_frames(tmp_path):
    frames_dir = _partial_frames(tmp_path / "frames")
    out = tmp_path / "figures"
    written, problems = figures.render(frames_dir, out, check=True)
    assert problems == []
    names = {str(p.relative_to(out)) for p in written}
    for expected in (
        "paradigm-compare/emf/dots_l336_o96_mae.png",
        "paradigm-compare/emf/gap-forest_l336_o96_mse.png",
        "paradigm-compare/emf/k-gain_l336_o96_mae.png",
        "paradigm-compare/emf/cost_l336_o96_mae.png",
        "paradigm-compare/emf/calibration_l336_o96.png",
        "paradigm-compare/emf/seed-skill_l336_o96.png",
        "paradigm-compare/voip/dots_l24-48_o6-12_mae.png",
        "paradigm-compare/voip/gap-forest_l24_o6_mae.png",
        "paradigm-compare/scoreboard_lall_oall.png",
        "per-band/ed2022/pc-vs-mv_l336_o96_mse.png",
        "per-band/ed2022/best-k-vs-k1_l336_o96_mse.png",
        "series/dots_l336_o96_mae.png",
    ):
        assert expected in names
    assert not any("2024" in n for n in names)
    assert all(p.stat().st_size > 0 for p in written)


def test_one_family_and_one_figure(tmp_path):
    frames_dir = _partial_frames(tmp_path / "frames")
    written, _ = figures.render(frames_dir, tmp_path / "figures", families=("voip",),
                                only=["P4"], check=False)
    assert [p.name for p in written] == ["dots_l24-48_o6-12_mae.png",
                                         "dots_l24-48_o6-12_mse.png"]


def test_an_unknown_figure_id_is_refused(tmp_path):
    with pytest.raises(ValueError):
        figures.render(_partial_frames(tmp_path), tmp_path / "out", only=["P12"])


def test_a_per_channel_bracket_is_drawn_only_where_it_shows():
    import matplotlib.pyplot as plt

    from emfres import fig_cost, figkit

    style = figkit.setup("paper")
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_xscale("log")
    ax.set_xlim(0.5, 30)
    # A +-3% bracket on an axis spanning 60x hides under its mark: not drawn,
    # and the note says how wide it is instead.
    drawn, note = fig_cost._band_whiskers(fig, [(ax, 9.7, 10.3, 0.0, "k", 1.0, 10.0)], style)
    assert not drawn and "−3% to +3%" in note
    # A bracket from half to twice the estimate shows, and is drawn.
    drawn, note = fig_cost._band_whiskers(fig, [(ax, 5.0, 20.0, 0.0, "k", 1.0, 10.0)], style)
    assert drawn and note is None
    plt.close(fig)


def test_a_label_goes_to_the_first_spot_that_covers_no_mark():
    import matplotlib.pyplot as plt

    from emfres import figkit

    style = figkit.setup("paper")
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rule = ax.axvline(0.5)
    ax.plot([0.6], [0.95], marker="o", markersize=style.marker, linestyle="None")
    spots = [((0.5, top), ("data", "axes fraction"), (2 * side, 2 * (1 - 2 * top)),
              "left" if side > 0 else "right", "top" if top else "bottom")
             for top in (1, 0) for side in (1, -1)]
    # A mark right of the rule at the top: the label takes the top left.
    t = figkit.place_clear(ax, "threshold", spots, ignore=(rule,))
    assert t.get_horizontalalignment() == "right" and t.get_verticalalignment() == "top"
    assert [x.get_text() for x in ax.texts] == ["threshold"]
    # The rule it labels is ignored; any other rule is avoided like a mark.
    ax.axvline(0.3)
    t.remove()
    t = figkit.place_clear(ax, "threshold", spots, ignore=(rule,))
    assert t.get_verticalalignment() == "bottom" and t.get_horizontalalignment() == "left"
    plt.close(fig)


def _lane_panel(tmp_path, rows):
    """One ED2022 DLinear dot panel drawn from ``rows``, its labels not yet settled."""
    import matplotlib.pyplot as plt

    from emfres import fig_paradigms, figdata, figkit

    style = figkit.setup("paper")
    frames.write_csv(tmp_path / "runs.csv", rows, frames.RUN_FIELDS)
    corpus = figdata.load(tmp_path / "runs.csv", "emf", (336, 96))
    # One model row at the height a row gets in P1 (lanes ~5 points apart).
    fig = plt.figure(figsize=(2.4, 0.8))
    ax = fig.add_axes((0.1, 0.3, 0.85, 0.4))
    xs, _, labels = fig_paradigms.dot_panel(ax, corpus, "ed2022", ["DLinear"], "mae", style)
    fig_paradigms._model_ticks(ax, ["DLinear"], set(), style)
    ax.set_xlim(*figkit.pad_range(xs, 0.05))
    fig.canvas.draw()
    return fig, ax, labels, style


def _meets(ax, text, lines) -> bool:
    """Whether the text's opaque box meets any marker of ``lines``."""
    from emfres import figkit

    r = ax.figure.canvas.get_renderer()
    e = text.get_window_extent(r)
    pad = 0.1 * text.get_fontsize() * ax.figure.dpi / 72
    others = [line for line in ax.lines if line not in lines]
    return any(e.x0 - pad < b[2] and b[0] < e.x1 + pad and e.y0 - pad < b[3] and b[1] < e.y1 + pad
               for b in figkit._obstacles(ax, r, ignore=others))


def test_a_k_label_steps_clear_of_the_per_channel_mean_beside_its_diamond(tmp_path):
    import matplotlib.pyplot as plt

    from emfres import fig_paradigms

    rows = []
    for s in SEEDS:
        rows.append(_run("multivariate", s, 0.40))
        rows += [_run("perchannel", s, 0.3675, channel=ch) for ch in (0, 1)]
        rows.append(_run("cluster", s, 0.40, k=1, val_loss=0.5))
        rows.append(_run("cluster", s, 0.37, k=2, val_loss=0.3))
    fig, ax, labels, style = _lane_panel(tmp_path, rows)
    means = [line for line in ax.lines if line.get_marker() in ("o", "D")]
    (label,) = labels
    # Its first spot, on the diamond's outer side, is under the per-channel
    # mean just above it ...
    assert _meets(ax, label.text, means)
    fig_paradigms.settle(ax, labels, style)
    # ... and once settled no label box meets a mean, a diamond or a k = 1 dot.
    assert label.text.get_text() == "k=2"
    assert not any(_meets(ax, lab.text, means) for lab in labels)
    plt.close(fig)


def test_a_chosen_k_of_one_is_labelled_beside_its_diamond_not_past_an_outlier(tmp_path):
    import matplotlib.pyplot as plt

    from emfres import fig_paradigms

    rows = []
    for s, v in zip(SEEDS, (0.380, 0.381, 0.440)):
        rows.append(_run("multivariate", s, 0.40))
        rows += [_run("perchannel", s, 0.45, channel=ch) for ch in (0, 1)]
        rows.append(_run("cluster", s, v, k=1, val_loss=0.3))
        rows.append(_run("cluster", s, 0.39, k=2, val_loss=0.5))
    fig, ax, labels, style = _lane_panel(tmp_path, rows)
    fig_paradigms.settle(ax, labels, style)
    (label,) = labels
    assert label.text.get_text() == "k=1"
    r = fig.canvas.get_renderer()
    e = label.text.get_window_extent(r)
    diamond = ax.transData.transform((sum((0.380, 0.381, 0.440)) / 3, 0))[0]
    near = min(abs(e.x0 - diamond), abs(e.x1 - diamond)) * 72 / fig.dpi
    # Within two marker widths of the diamond, not past the 0.44 seed.
    assert near <= 2 * style.marker
    assert not _meets(ax, label.text, list(ax.lines))
    plt.close(fig)


def test_every_pending_cell_a_gap_forest_counts_is_drawn(tmp_path, monkeypatch):
    import re

    import matplotlib.pyplot as plt

    from emfres import fig_paradigms, figdata, figkit
    from emfres.figkit import INK_MUTED

    style = figkit.setup("paper")
    # LSTM's sweep has one run; PatchTST is planned but has no cluster run.
    rows = [_run(setup, s, 0.4 + 0.01 * i, model=model, lookback=1.0, **kw)
            for i, s in enumerate(SEEDS) for model in ("DLinear", "LSTM", "PatchTST")
            for setup, kw in (("multivariate", {}), ("perchannel", {"channel": 0}))]
    rows += [_run("cluster", s, 0.39, k=k, val_loss=0.5 - 0.1 * k, lookback=1.0)
             for s in SEEDS for k in (1, 2)]
    rows.append(_run("cluster", SEEDS[0], 0.44, k=1, val_loss=0.5, model="LSTM",
                     lookback=1.0))
    frames.write_csv(tmp_path / "runs.csv", rows, frames.RUN_FIELDS)
    corpus = figdata.load(tmp_path / "runs.csv", "emf", (336, 96))
    kept = []
    monkeypatch.setattr(fig_paradigms, "write", lambda fig, *a, **kw: kept.append(fig) or [])
    fig_paradigms.fig_forest(corpus, "mae", style, tmp_path, ("png",))
    (fig,) = kept
    for ax in fig.axes:
        sub = [t.get_text() for t in ax.texts if "/" in t.get_text()][0]
        counted = int(re.search(r"(\d+) pending", sub).group(1)) if "pending" in sub else 0
        markers = sum(1 for line in ax.lines if line.get_marker() == "o"
                      and line.get_markeredgecolor() == INK_MUTED)
        words = sum(1 for t in ax.texts if t.get_text() == "pending")
        assert markers + words == counted
    # Best k vs k=1 counts both: LSTM's gap and PatchTST's placeholder.
    assert [t.get_text() for t in fig.axes[1].texts].count("pending") == 1
    plt.close(fig)


def test_a_cluster_lane_with_no_run_yet_reads_pending(tmp_path):
    import matplotlib.pyplot as plt

    rows = [_run(setup, s, 0.4, lookback=1.0, **kw) for s in SEEDS
            for setup, kw in (("multivariate", {}), ("perchannel", {"channel": 0}))]
    fig, ax, labels, style = _lane_panel(tmp_path, rows)
    # DLinear has a cluster pipeline that has not run here: "pending", not a
    # blank lane (ARIMA's, which has none, reads "n/a").
    assert [lab.text.get_text() for lab in labels] == ["pending"]
    plt.close(fig)


def test_the_legacy_figures_read_a_cluster_cells_fixed_protocol(tmp_path):
    from pathlib import Path

    from emfres.records import TRUST_COLUMNS, records_from_cells

    results = Path(__file__).resolve().parents[1]
    base = {f: "" for f in frames.CELL_FIELDS}
    cell = dict(base, setup="cluster", family="emf", model="DLinear", dataset="ed2022",
                seq_len=336, pred_len=96, k="2", conf=1, mse="0.50", mae="0.40",
                running_time="60", ic="80", mse_fixed="0.45", mae_fixed="0.35",
                ic_fixed="90")
    mv = dict(base, setup="multivariate", family="emf", model="DLinear", dataset="ed2022",
              seq_len=336, pred_len=96, conf=1, mse="0.55", mae="0.42", running_time="50",
              ic="85")
    path = tmp_path / "cells.csv"
    frames.write_csv(path, [cell, mv], frames.CELL_FIELDS)
    got = {(r.section.split()[0], r.metric): r.value
           for r in records_from_cells(path, results)
           + records_from_cells(path, results, metrics=TRUST_COLUMNS)}
    # As the Cluster section prints them: the fixed protocol; time has none.
    assert got[("Cluster", "MSE")] == 0.45 and got[("Cluster", "MAE")] == 0.35
    assert got[("Cluster", "Coverage")] == 90.0 and got[("Cluster", "Training time")] == 60.0
    assert got[("Multivariate", "MSE")] == 0.55

    # Frames older than the fixed columns: the batch protocol, as the section then prints.
    frames.write_csv(path, [dict(cell, mse_fixed="", mae_fixed="", ic_fixed=""), mv],
                     frames.CELL_FIELDS)
    old = {(r.section.split()[0], r.metric): r.value for r in records_from_cells(path, results)}
    assert old[("Cluster", "MSE")] == 0.50


def test_p7_counts_and_notes_a_cost_another_users_gpu_process_may_have_inflated(
        tmp_path, monkeypatch):
    import csv

    from emfres import fig_cost, figdata as FD

    frames_dir = _partial_frames(tmp_path / "frames")
    runs_csv = frames_dir / "runs.csv"
    with open(runs_csv) as f:
        rows = list(csv.DictReader(f))
    # Two seeds of DLinear's k=2 on ED2022 overlapped another user's process.
    for r in rows:
        if (r["setup"], r["model"], r["dataset"], r["k"]) == ("cluster", "DLinear",
                                                              "ed2022", "2") \
                and r["seed"] != str(SEEDS[0]):
            r["foreign_overlap"], r["contended"] = "0.5", "foreign"
    frames.write_csv(runs_csv, rows, frames.RUN_FIELDS)
    corpus = FD.load(runs_csv, "emf", (336, 96))
    costs = FD.costs(corpus, runs_csv, frames_dir / "timing.csv")
    assert corpus.board.choices[("ed2022", "DLinear")].k == 2
    from emfres import paradigms as PG
    best = costs[(("ed2022", "DLinear"), PG.BEST)]
    assert best.foreign == 2 and best.n == 3
    assert costs[(("ed2022", "DLinear"), PG.MV)].foreign == 0
    assert all(c.foreign == 0 for (cell, _), c in costs.items() if cell[0] == "uh2022")

    notes = []
    monkeypatch.setattr(fig_cost, "footnote", lambda fig, text, style, y=0.0: notes.append(text))
    from emfres import figkit
    written = fig_cost.fig_cost(corpus, runs_csv, frames_dir / "timing.csv", "mae",
                                figkit.setup("paper"), tmp_path / "figs", ["png"])
    assert written and fig_cost.CONTENDED_NOTE in notes
