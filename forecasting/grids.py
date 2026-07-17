"""Canonical experiment grids for the conformal-prediction point-forecasting sweep.

Each model entry is (display_label, on-disk model_id, job_name). The on-disk
model_id can differ from the label (the Transformer's is RF_EMF_Transformer) and
drives the run directory, so it must match the model's args.yaml exactly.
"""

WINDOWS = [(192, 96), (192, 336), (336, 96), (336, 336)]

MV_MODELS = [
    ("DLinear", "DLinear", "rf_emf_mv/dlinear"),
    ("PatchTST", "PatchTST", "rf_emf_mv/patchtst"),
    ("TSMixer", "TSMixer", "rf_emf_mv/tsmixer"),
    ("Transformer", "RF_EMF_Transformer", "rf_emf_mv/transformer"),
    ("EMForecaster", "EMForecaster", "rf_emf_mv/emforecaster"),
]
MV_DATASETS = ["ED2022", "UH2022", "EMF_2023_university_5_with_PTV"]

UNIVARIATE_MODELS = [
    ("DLinear", "DLinear", "rf_emf/dlinear"),
    ("PatchTST", "PatchTST", "rf_emf/patchtst"),
    ("TSMixer", "TSMixer", "rf_emf/tsmixer"),
    ("Transformer", "RF_EMF_Transformer", "rf_emf/baselines/transformer"),
    ("EMForecaster", "EMForecaster", "rf_emf/emforecaster"),
]
UNIVARIATE_DATASETS = ["rf_emf_det", "rf_emf_ptv", "rf_emf_ptv2"]

GRIDS = {
    "multivariate": {
        "models": MV_MODELS, "datasets": MV_DATASETS, "windows": WINDOWS, "alpha": 0.1,
    },
    "univariate": {
        "models": UNIVARIATE_MODELS, "datasets": UNIVARIATE_DATASETS, "windows": WINDOWS, "alpha": 0.1,
    },
}
