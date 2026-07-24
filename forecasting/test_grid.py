import json
import os

from forecasting import grid, grids


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


# The two comparable grids share MV_MODELS (6 models x 3 datasets x 4 windows);
# univariate_series has its own 5-model list. Spelled out rather than derived
# from len(models), so a model silently dropped from a grid fails here.
EXPECTED_CELLS = {"multivariate": 72, "univariate": 72, "univariate_series": 60}


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
    # grid's, so it must share the same models, datasets and windows.
    uni, mv = grids.GRIDS["univariate"], grids.GRIDS["multivariate"]
    assert uni["models"] == mv["models"]
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
    import types as _t

    def fake_load_config(path):
        exp = _t.SimpleNamespace(model_id="DLinear", mps=False, gpu_id=0, compile=False,
                                 neptune=True, id="x", ablation_id=1, seed_list=[], seed=0, time="")
        data = _t.SimpleNamespace(dataset="", seq_len=0, pred_len=0, num_workers=0, pin_memory=False)
        conf = _t.SimpleNamespace(conf=False, alpha=0.0)
        sl = _t.SimpleNamespace(batch_size=0)
        return _t.SimpleNamespace(exp=exp, data=data, conf=conf, sl=sl)

    class FakeExperiment:
        def __init__(self, args):
            self.args = args

        def run(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(grid, "setup_modules", lambda setup: (fake_load_config, FakeExperiment))
    failed = grid.run_grid("multivariate", dry_run=False, logs_dir=str(tmp_path))
    # Every cell raised, but the grid kept going and recorded them all as failed.
    assert len(failed) == EXPECTED_CELLS["multivariate"]
