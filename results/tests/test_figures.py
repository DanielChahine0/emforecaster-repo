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
