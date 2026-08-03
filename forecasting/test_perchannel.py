import json
import os

from forecasting import grids, perchannel


def _write_log(tmp_path, model_id, dataset, seq_len, pred_len, channel, payload,
               seed=perchannel.SEED):
    path = perchannel.channel_log_path(model_id, dataset, seq_len, pred_len, channel,
                                       seed=seed, logs_dir=str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def test_dataset_num_channels_matches_loader_filter():
    # The bands the loader keeps: everything but Timestamp, EMF_* and PTV.
    assert grids.dataset_num_channels("ED2022") == 27
    assert grids.dataset_num_channels("UH2022") == 27
    assert grids.dataset_num_channels("EMF_2023_university_5_with_PTV") == 29


def test_channel_run_dirs_are_distinct_per_band():
    # build_run_dir has no channel level; without the ch<i> tag every band of a
    # cell would share one directory and overwrite the previous band.
    dirs = {perchannel.channel_run_dir("DLinear", "ED2022", 192, 96, ch)
            for ch in range(27)}
    assert len(dirs) == 27
    assert perchannel.channel_run_dir("DLinear", "ED2022", 192, 96, 5).endswith("ch5")


def test_per_band_runs_log_under_their_own_tree():
    # Per-band runs go under "univariate (per channel)", never the plain
    # "univariate" tree that the single-series package owns.
    run_dir = perchannel.channel_run_dir("DLinear", "ED2022", 192, 96, 0)
    assert run_dir.startswith(os.path.join("logs", "univariate (per channel)"))
    assert perchannel.LOG_SETUP == "univariate (per channel)"
    # A leading path component exactly "univariate" would be the wrong tree.
    assert os.path.join("logs", "univariate", "DLinear") not in run_dir


def test_ablation_pins_band_and_channel_count():
    ablation = perchannel.channel_ablation("ED2022", 336, 96, 7, 0.1)
    # single_channel is only honoured in univariate mode, and univariate mode
    # skips the num_channels auto-inference, so both must be set explicitly.
    assert ablation["global.data.univariate"] is True
    assert ablation["global.data.single_channel"] is True
    assert ablation["global.data.target_channel"] == 7
    assert ablation["global.data.num_channels"] == 1
    assert ablation["global.exp.id"] == "ch7"
    assert ablation["global.data.dataset"] == "ED2022"
    assert ablation["global.data.seq_len"] == 336
    assert ablation["global.data.pred_len"] == 96


def test_worker_budget_does_not_oversubscribe_cores():
    # hardware.py pins num_workers=4 for CUDA, which is right solo but would put
    # 8*4=32 loader processes on a 24-core box. Total processes (one training per
    # job plus its loaders) must stay within the core count.
    for jobs in (1, 4, 8, 12, 24):
        workers, threads = perchannel.worker_budget(jobs, cpu_count=24)
        assert jobs + jobs * workers <= 24, f"{jobs} jobs oversubscribes"
        assert 0 <= workers <= 4
        assert threads >= 1
    assert perchannel.worker_budget(8, cpu_count=24)[0] == 2
    assert perchannel.worker_budget(12, cpu_count=24)[0] == 1
    # A tiny pool keeps the solo default rather than inventing more workers.
    assert perchannel.worker_budget(4, cpu_count=24)[0] == 4


def test_ablation_carries_worker_override():
    plain = perchannel.channel_ablation("ED2022", 192, 96, 0, 0.1)
    assert "global.data.num_workers" not in plain
    pooled = perchannel.channel_ablation("ED2022", 192, 96, 0, 0.1, workers=2)
    assert pooled["global.data.num_workers"] == 2


def test_ablation_applies_memory_profile_for_heavy_cells():
    # Same condition grid.py's joint EMForecaster cells hit at seq_len >= 336:
    # without this, `jobs` concurrent band trainings each ask for a full-batch
    # cell's worth of activations on one shared GPU.
    ablation = perchannel.channel_ablation("ED2022", 336, 96, 0, 0.1,
                                           model_id="EMForecaster")
    assert ablation["global.sl.batch_size"] == 128
    assert ablation["global.sl.accum_steps"] == 4


def test_ablation_skips_memory_profile_without_model_id():
    plain = perchannel.channel_ablation("ED2022", 336, 96, 0, 0.1)
    assert "global.sl.batch_size" not in plain
    # A light cell (e.g. DLinear) at the same window needs no override either.
    light = perchannel.channel_ablation("ED2022", 336, 96, 0, 0.1, model_id="DLinear")
    assert "global.sl.batch_size" not in light


def test_channel_done_requires_the_conformal_marker(tmp_path):
    args = ("DLinear", "ED2022", 192, 96, 3)
    assert perchannel.channel_done(*args, logs_dir=str(tmp_path)) is False
    _write_log(tmp_path, *args, {"sl_test/mse": 0.5})
    assert perchannel.channel_done(*args, logs_dir=str(tmp_path)) is False
    _write_log(tmp_path, *args, {perchannel.MARKER: 0.4, "sl_test/mse": 0.5})
    assert perchannel.channel_done(*args, logs_dir=str(tmp_path)) is True


def test_aggregate_cell_averages_bands(tmp_path):
    for channel in range(3):
        _write_log(tmp_path, "DLinear", "ED2022", 192, 96, channel, {
            perchannel.MARKER: 1.0,
            "sl_test/mse": float(channel),   # 0, 1, 2 -> mean 1.0
            "sl_test/mae": 2.0,
        })
    summary = perchannel.aggregate_cell("DLinear", "ED2022", 192, 96, 3,
                                        logs_dir=str(tmp_path))
    assert summary["complete"] is True
    assert summary["channels_complete"] == 3
    assert summary["mse_mean"] == 1.0
    assert summary["mae_mean"] == 2.0
    assert summary["mse_per_channel"] == [0.0, 1.0, 2.0]


def test_aggregate_cell_reports_missing_bands(tmp_path):
    # Only band 0 finished; the cell must be flagged partial rather than reported
    # as a clean average over the one band that happened to land.
    _write_log(tmp_path, "DLinear", "ED2022", 192, 96, 0,
               {perchannel.MARKER: 1.0, "sl_test/mse": 0.25})
    summary = perchannel.aggregate_cell("DLinear", "ED2022", 192, 96, 3,
                                        logs_dir=str(tmp_path))
    assert summary["complete"] is False
    assert summary["missing_channels"] == [1, 2]
    assert summary["channels_complete"] == 1


def test_aggregate_cell_falls_back_to_loss(tmp_path):
    # A log without sl_test/mse still has sl_test/loss, the same number under MSE.
    _write_log(tmp_path, "DLinear", "ED2022", 192, 96, 0,
               {perchannel.MARKER: 1.0, "sl_test/loss": 0.75})
    summary = perchannel.aggregate_cell("DLinear", "ED2022", 192, 96, 1,
                                        logs_dir=str(tmp_path))
    assert summary["mse_mean"] == 0.75


def test_aggregate_cell_handles_per_epoch_lists(tmp_path):
    # epoch_logger writes some metrics as a per-epoch list; the value is the last.
    _write_log(tmp_path, "DLinear", "ED2022", 192, 96, 0,
               {perchannel.MARKER: 1.0, "sl_test/mse": [0.9, 0.5]})
    summary = perchannel.aggregate_cell("DLinear", "ED2022", 192, 96, 1,
                                        logs_dir=str(tmp_path))
    assert summary["mse_mean"] == 0.5


def test_aggregate_cell_writes_aggregate_json(tmp_path):
    _write_log(tmp_path, "DLinear", "ED2022", 192, 96, 0,
               {perchannel.MARKER: 1.0, "sl_test/mse": 0.25})
    perchannel.aggregate_cell("DLinear", "ED2022", 192, 96, 1, logs_dir=str(tmp_path))
    target = os.path.join(
        perchannel.cell_dir("DLinear", "ED2022", 192, 96, logs_dir=str(tmp_path)),
        "aggregate.json",
    )
    with open(target) as f:
        written = json.load(f)
    assert written["mse_mean"] == 0.25
    assert written["num_channels"] == 1


def test_dry_run_plans_every_band_and_trains_nothing(tmp_path, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("dry_run must not train")

    monkeypatch.setattr(perchannel, "run_channel", explode)
    failed = perchannel.run_per_channel_grid(jobs=2, dry_run=True,
                                             logs_dir=str(tmp_path))
    assert failed == []


def test_window_filter_runs_only_the_selected_window(tmp_path, monkeypatch):
    # One window -> len(models) x 3 datasets x 1 window cells, and the bands of
    # just those cells: len(models) * (27 + 27 + 29) runs. Derived from the grid
    # so adding a model does not turn this into a spurious failure.
    num_models = len(grids.GRIDS["univariate"]["models"])
    windows = [(336, 96)]
    calls = []

    def fake_run_channel(job, dataset, seq_len, pred_len, channel, alpha,
                         seed=perchannel.SEED, workers=None, threads=None,
                         model_id=None):
        calls.append((seq_len, pred_len, dataset, channel))
        return True, ""

    monkeypatch.setattr(perchannel, "run_channel", fake_run_channel)
    perchannel.run_per_channel_grid(jobs=4, logs_dir=str(tmp_path), windows=windows)
    assert len(calls) == num_models * (27 + 27 + 29)
    # Every run used the requested window and nothing else.
    assert {(l, o) for l, o, _, _ in calls} == {(336, 96)}


def test_seeds_multiply_band_runs_and_aggregate_per_seed(tmp_path, monkeypatch):
    # A seed multiplies band runs, not cells: every band of every cell is trained
    # once per seed, and each seed's bands land under their own <seed>/ leaf.
    num_models = len(grids.GRIDS["univariate"]["models"])
    windows = [(336, 96)]
    seeds = [perchannel.SEED, perchannel.SEED + 1]
    calls = []

    def fake_run_channel(job, dataset, seq_len, pred_len, channel, alpha,
                         seed=perchannel.SEED, workers=None, threads=None,
                         model_id=None):
        calls.append((dataset, channel, seed))
        return True, ""

    monkeypatch.setattr(perchannel, "run_channel", fake_run_channel)
    perchannel.run_per_channel_grid(jobs=4, logs_dir=str(tmp_path), windows=windows,
                                    seeds=seeds)
    bands_per_seed = num_models * (27 + 27 + 29)
    assert len(calls) == bands_per_seed * len(seeds)
    for seed in seeds:
        assert sum(1 for c in calls if c[2] == seed) == bands_per_seed


def test_extra_seeds_get_their_own_aggregate_file(tmp_path):
    # Every seed's bands share one cell dir, so a single aggregate.json would let
    # the last seed aggregated overwrite the others.
    assert perchannel.aggregate_name(perchannel.SEED) == "aggregate.json"
    other = perchannel.SEED + 1
    assert perchannel.aggregate_name(other) == f"aggregate_seed{other}.json"

    for seed, mse in ((perchannel.SEED, 0.25), (other, 0.5)):
        _write_log(tmp_path, "DLinear", "ED2022", 192, 96, 0,
                   {perchannel.MARKER: 1.0, "sl_test/mse": mse}, seed=seed)
        perchannel.aggregate_cell("DLinear", "ED2022", 192, 96, 1, seed=seed,
                                  logs_dir=str(tmp_path))

    cell = perchannel.cell_dir("DLinear", "ED2022", 192, 96, logs_dir=str(tmp_path))
    for seed, mse in ((perchannel.SEED, 0.25), (other, 0.5)):
        with open(os.path.join(cell, perchannel.aggregate_name(seed))) as f:
            written = json.load(f)
        assert written["seed"] == seed
        assert written["mse_mean"] == mse


def test_run_grid_skips_completed_bands_and_records_failures(tmp_path, monkeypatch):
    # Every cell of the grid: (models x windows) runs on each of two 27-band
    # datasets and on the 29-band one.
    per_dataset = (len(grids.GRIDS["univariate"]["models"])
                   * len(grids.GRIDS["univariate"]["windows"]))
    expected = per_dataset * 27 + per_dataset * 27 + per_dataset * 29
    calls = []

    def fake_run_channel(job, dataset, seq_len, pred_len, channel, alpha,
                         seed=perchannel.SEED, workers=None, threads=None,
                         model_id=None):
        calls.append((job, dataset, seq_len, pred_len, channel))
        return False, "boom"

    monkeypatch.setattr(perchannel, "run_channel", fake_run_channel)
    failed = perchannel.run_per_channel_grid(jobs=4, logs_dir=str(tmp_path))
    # Every band ran and every band failed, but the sweep completed regardless.
    assert len(calls) == expected
    assert len(failed) == expected

    # A band with the marker is skipped on the next pass.
    _write_log(tmp_path, "DLinear", "ED2022", 192, 96, 0, {perchannel.MARKER: 1.0})
    calls.clear()
    perchannel.run_per_channel_grid(jobs=4, logs_dir=str(tmp_path))
    assert len(calls) == expected - 1
