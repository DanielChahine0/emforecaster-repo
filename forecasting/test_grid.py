from forecasting import grids


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
