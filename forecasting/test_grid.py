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
    for setup in ("multivariate", "univariate"):
        load_config, _ = grid.setup_modules(setup)
        for label, model_id, job in grids.GRIDS[setup]["models"]:
            args = load_config(grid.args_path(setup, job))
            assert args.exp.model_id == model_id, f"{setup} {job}: {args.exp.model_id} != {model_id}"


def test_both_grids_have_60_cells():
    for setup in ("multivariate", "univariate"):
        g = grids.GRIDS[setup]
        n = len(g["models"]) * len(g["datasets"]) * len(g["windows"])
        assert n == 60, f"{setup} has {n} cells, expected 60"


def test_grids_use_alpha_point_one():
    assert grids.GRIDS["multivariate"]["alpha"] == 0.1
    assert grids.GRIDS["univariate"]["alpha"] == 0.1


def test_model_tuples_are_label_id_job():
    mv = dict((label, (model_id, job)) for label, model_id, job in grids.GRIDS["multivariate"]["models"])
    assert mv["Transformer"] == ("RF_EMF_Transformer", "rf_emf_mv/transformer")
    uni = dict((label, (model_id, job)) for label, model_id, job in grids.GRIDS["univariate"]["models"])
    assert uni["Transformer"] == ("RF_EMF_Transformer", "rf_emf/baselines/transformer")


def test_run_grid_dry_run_lists_all_cells(tmp_path):
    # dry_run must not train; it should visit all 60 cells and report them.
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
    # Every cell raised, but the grid kept going and recorded all 60 as failed.
    assert len(failed) == 60
