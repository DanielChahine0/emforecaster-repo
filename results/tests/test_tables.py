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
