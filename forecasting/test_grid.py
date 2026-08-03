import json
import os
import types

from forecasting import grid, grids


def _fake_load_config(path):
    """A config with just the attributes run_grid overwrites, so no torch needed."""
    exp = types.SimpleNamespace(model_id="DLinear", mps=False, gpu_id=0, compile=False,
                                neptune=True, id="x", ablation_id=1, seed_list=[],
                                seed=0, time="")
    data = types.SimpleNamespace(dataset="", seq_len=0, pred_len=0, num_workers=0,
                                 pin_memory=False)
    conf = types.SimpleNamespace(conf=False, alpha=0.0)
    sl = types.SimpleNamespace(batch_size=0)
    hybrid = types.SimpleNamespace(enabled=False, k_clusters=0)
    return types.SimpleNamespace(exp=exp, data=data, conf=conf, sl=sl, hybrid=hybrid)


def _fake_experiment(on_run):
    """Experiment stand-in calling on_run(args) instead of training."""
    class FakeExperiment:
        def __init__(self, args):
            self.args = args

        def run(self):
            return on_run(self.args)

    return FakeExperiment


def test_cell_done_false_when_no_log(tmp_path):
    done = grid.cell_done("multivariate", "DLinear", "ED2022", 192, 96,
                          logs_dir=str(tmp_path))
    assert done is False


def test_cell_done_true_only_with_marker(tmp_path):
    run_dir = grid.build_run_dir("multivariate", "DLinear", "ED2022", 192, 96,
                                 conf_on=True, logs_dir=str(tmp_path))
    seed_dir = os.path.join(run_dir, str(grid.SEED))
    os.makedirs(seed_dir)
    log = os.path.join(seed_dir, "log.json")
    with open(log, "w") as f:
        json.dump({"some_metric": 1}, f)
    assert grid.cell_done("multivariate", "DLinear", "ED2022", 192, 96,
                          logs_dir=str(tmp_path)) is False
    with open(log, "w") as f:
        json.dump({"sl_test/interval_width_mean": 0.42}, f)
    assert grid.cell_done("multivariate", "DLinear", "ED2022", 192, 96,
                          logs_dir=str(tmp_path)) is True


def test_grid_model_ids_match_job_configs():
    # The grid's on-disk model_id must equal the job config's exp.model_id, or
    # the resume path won't match what exp.py writes.
    for setup in ("multivariate", "univariate", "univariate_series"):
        load_config, _ = grid.setup_modules(setup)
        for label, model_id, job in grids.GRIDS[setup]["models"]:
            args = load_config(grid.args_path(setup, job))
            assert args.exp.model_id == model_id, f"{setup} {job}: {args.exp.model_id} != {model_id}"


# multivariate is MV_MODELS (6 models x 3 datasets x 4 windows). The per-band
# univariate grid runs those same six plus PatchTST, which was dropped from the
# joint sweep on 2026-07-24 but still runs one band at a time, so it is one model
# wider. univariate_series has its own 5-model list. Spelled out rather than
# derived from len(models), so a model silently dropped from a grid fails here.
EXPECTED_CELLS = {"multivariate": 72, "univariate": 84, "univariate_series": 60}


def test_grid_cell_counts():
    for setup, expected in EXPECTED_CELLS.items():
        g = grids.GRIDS[setup]
        n = len(g["models"]) * len(g["datasets"]) * len(g["windows"])
        assert n == expected, f"{setup} has {n} cells, expected {expected}"


def test_grid_does_not_use_the_batch_axis_lstm():
    # models/rf_emf_baselines.py::LSTM (model_id RF_EMF_LSTM) runs its recurrence
    # over the DataLoader batch axis and squeezes a (B, 1, O) forecast against a
    # (B, 1, O) target. It stays reachable because it produced the published
    # single-series column, so guard against it drifting back into a sweep.
    for setup in ("multivariate", "univariate"):
        ids = {model_id for _, model_id, _ in grids.GRIDS[setup]["models"]}
        assert "RF_EMF_LSTM" not in ids, f"{setup} must use 'LSTM', not RF_EMF_LSTM"
        assert "LSTM" in ids


def test_grids_use_alpha_point_one():
    for setup in ("multivariate", "univariate", "univariate_series"):
        assert grids.GRIDS[setup]["alpha"] == 0.1


def test_model_tuples_are_label_id_job():
    mv = dict((label, (model_id, job)) for label, model_id, job in grids.GRIDS["multivariate"]["models"])
    assert mv["Transformer"] == ("RF_EMF_Transformer", "rf_emf_mv/transformer")
    series = dict((label, (model_id, job)) for label, model_id, job in grids.GRIDS["univariate_series"]["models"])
    assert series["Transformer"] == ("RF_EMF_Transformer", "rf_emf/baselines/transformer")


def test_univariate_grid_mirrors_multivariate_cells():
    # The per-band grid is only interpretable if its cells line up with the joint
    # grid's, so every multivariate model/dataset/window must have a per-band
    # counterpart. PatchTST is the one deliberate extra: it left the joint sweep
    # on 2026-07-24 but still runs per band, so it has no multivariate cell
    # beside it and every other per-band model does.
    uni, mv = grids.GRIDS["univariate"], grids.GRIDS["multivariate"]
    assert [m for m in uni["models"] if m in mv["models"]] == mv["models"]
    assert [m for m in uni["models"] if m not in mv["models"]] == [grids.PATCHTST_MODEL]
    assert uni["datasets"] == mv["datasets"]
    assert uni["windows"] == mv["windows"]
    assert uni["per_channel"] is True
    assert not mv.get("per_channel")


def test_univariate_series_keeps_single_series_datasets():
    assert grids.GRIDS["univariate_series"]["datasets"] == [
        "rf_emf_det", "rf_emf_ptv", "rf_emf_ptv2"
    ]


def test_setup_package_and_log_setup_mapping():
    # univariate_series runs the original forecasting.univariate package, and the
    # per-band grid reuses forecasting.multivariate; both log under univariate/.
    assert grid.SETUP_PACKAGE["univariate_series"] == "univariate"
    assert grid.SETUP_PACKAGE["univariate"] == "multivariate"
    assert grid.LOG_SETUP["univariate_series"] == "univariate"
    assert grid.LOG_SETUP["multivariate"] == "multivariate"
    # The per-band grid logs to its own tree, not the single-series "univariate".
    assert grid.LOG_SETUP["univariate"] == "univariate (per channel)"
    assert grid.args_path("univariate_series", "rf_emf/dlinear").endswith(
        os.path.join("forecasting", "univariate", "jobs", "exp", "rf_emf", "dlinear", "args.yaml")
    )


def test_parse_window_spec_accepts_both_forms():
    assert grids.parse_window_spec("336x96") == [(336, 96)]
    assert grids.parse_window_spec("L336_O96") == [(336, 96)]
    assert grids.parse_window_spec("336_96") == [(336, 96)]
    assert grids.parse_window_spec("336x96, 192x336") == [(336, 96), (192, 336)]


def test_parse_window_spec_rejects_garbage():
    for bad in ("", "336", "axb", "336x", "L336"):
        try:
            grids.parse_window_spec(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should be rejected")


def test_select_windows_filters_and_keeps_grid_order():
    windows = grids.GRIDS["univariate"]["windows"]
    # None -> every window, unchanged.
    assert grids.select_windows(None, windows) == list(windows)
    assert grids.select_windows("336x96", windows) == [(336, 96)]
    # Result follows grid order regardless of how the flag was typed.
    assert grids.select_windows("336x336,192x96", windows) == [(192, 96), (336, 336)]


def test_select_windows_rejects_window_not_in_grid():
    windows = grids.GRIDS["univariate"]["windows"]
    try:
        grids.select_windows("500x500", windows)
    except ValueError as exc:
        assert "not in grid windows" in str(exc)
    else:
        raise AssertionError("a window outside the grid should be rejected")


def test_run_grid_window_filter_limits_cells(tmp_path):
    # One window instead of four -> 18 cells (6 models x 3 datasets x 1 window).
    failed = grid.run_grid("multivariate", dry_run=True, logs_dir=str(tmp_path),
                           windows=[(336, 96)])
    assert failed == []


def test_select_seeds_default_is_the_canonical_seed_only():
    # No --seeds must reproduce the pre-flag behaviour exactly, or every existing
    # run on disk stops matching the resume check.
    assert grids.select_seeds(None) == [grid.SEED]


def test_select_seeds_counts_up_from_the_canonical_seed():
    # The canonical seed has to come first so a multi-seed sweep resumes the runs
    # already on disk instead of training a fresh set beside them.
    assert grids.select_seeds(3) == [grid.SEED, grid.SEED + 1, grid.SEED + 2]
    assert grids.select_seeds("3") == grids.select_seeds(3)


def test_select_seeds_takes_an_explicit_list_and_dedupes():
    assert grids.select_seeds("69422,69423") == [69422, 69423]
    assert grids.select_seeds("69421,69421") == [69421]


def test_select_seeds_rejects_bad_values():
    for spec, expected in ((0, "at least 1"), (-2, "at least 1"), ("x", "bad --seeds")):
        try:
            grids.select_seeds(spec)
        except ValueError as exc:
            assert expected in str(exc), f"{spec!r}: {exc}"
        else:
            raise AssertionError(f"--seeds {spec!r} should be rejected")


def test_run_grid_trains_every_cell_once_per_seed(tmp_path, monkeypatch):
    runs = []

    def fake_run(args):
        runs.append((args.data.dataset, args.data.seq_len, args.data.pred_len,
                     args.exp.seed))

    monkeypatch.setattr(grid, "setup_modules",
                        lambda setup: (_fake_load_config, _fake_experiment(fake_run)))
    windows = [(336, 96)]
    seeds = [grid.SEED, grid.SEED + 1]
    grid.run_grid("multivariate", dry_run=False, logs_dir=str(tmp_path),
                  windows=windows, seeds=seeds)

    cells = len(grids.GRIDS["multivariate"]["models"]) * 3 * len(windows)
    assert len(runs) == cells * len(seeds)
    # Each seed covered the whole grid, and seed_list followed exp.seed rather
    # than staying on the config's default.
    for seed in seeds:
        assert sum(1 for r in runs if r[3] == seed) == cells


def test_run_grid_runs_only_the_seeds_not_already_complete(tmp_path, monkeypatch):
    # A cell complete at the canonical seed must not retrain when a second seed
    # is added -- resume is per (cell, seed), not per cell.
    run_dir = grid.build_run_dir("multivariate", "DLinear", "ED2022", 336, 96,
                                 conf_on=True, logs_dir=str(tmp_path))
    seed_dir = os.path.join(run_dir, str(grid.SEED))
    os.makedirs(seed_dir)
    with open(os.path.join(seed_dir, "log.json"), "w") as f:
        json.dump({grid.MARKER: 0.42}, f)

    runs = []
    monkeypatch.setattr(
        grid, "setup_modules",
        lambda setup: (_fake_load_config,
                       _fake_experiment(lambda args: runs.append(args.exp.seed))))
    windows = [(336, 96)]
    seeds = [grid.SEED, grid.SEED + 1]
    grid.run_grid("multivariate", dry_run=False, logs_dir=str(tmp_path),
                  windows=windows, seeds=seeds)

    cells = len(grids.GRIDS["multivariate"]["models"]) * 3 * len(windows)
    # The one complete (cell, seed) was skipped; the same cell's second seed and
    # every other cell still ran.
    assert runs.count(grid.SEED) == cells - 1
    assert runs.count(grid.SEED + 1) == cells


def test_run_grid_refuses_per_channel_setup():
    # Running the per-band grid through the joint driver would silently train the
    # wrong thing, so it must raise instead.
    try:
        grid.run_grid("univariate", dry_run=True)
    except ValueError as exc:
        assert "per-channel" in str(exc)
    else:
        raise AssertionError("run_grid should refuse a per_channel grid")


def test_run_grid_dry_run_lists_all_cells(tmp_path):
    # dry_run must not train; it should visit every cell and report them.
    failed = grid.run_grid("multivariate", dry_run=True, logs_dir=str(tmp_path))
    assert failed == []


def test_run_grid_records_failures_and_continues(tmp_path, monkeypatch):
    def explode(args):
        raise RuntimeError("boom")

    monkeypatch.setattr(grid, "setup_modules",
                        lambda setup: (_fake_load_config, _fake_experiment(explode)))
    failed = grid.run_grid("multivariate", dry_run=False, logs_dir=str(tmp_path))
    # Every cell raised, but the grid kept going and recorded them all as failed.
    assert len(failed) == EXPECTED_CELLS["multivariate"]
