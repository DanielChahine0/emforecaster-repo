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
