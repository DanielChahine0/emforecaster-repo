import os
from forecasting.run_path import build_run_dir, dataset_slug, iter_run_dirs


def test_dataset_slug_known():
    assert dataset_slug("ED2022") == "ed2022"
    assert dataset_slug("EMF_2023_university_5_with_PTV") == "emf2023"
    assert dataset_slug("rf_emf_det") == "det"


def test_dataset_slug_fallback_lowercases_unknown():
    assert dataset_slug("SomethingNew") == "somethingnew"


def test_build_run_dir_multivariate_cp():
    assert build_run_dir("multivariate", "DLinear", "ED2022", 192, 336, True) == \
        os.path.join("logs", "multivariate", "DLinear", "ed2022", "L192_O336_cp")


def test_build_run_dir_univariate_pf():
    assert build_run_dir("univariate", "RF_EMF_LSTM", "rf_emf_det", 192, 96, False) == \
        os.path.join("logs", "univariate", "RF_EMF_LSTM", "det", "L192_O96_pf")


def test_build_run_dir_ablation_level_only_when_not_one():
    base = os.path.join("logs", "multivariate", "DLinear", "ed2022", "L192_O96_cp")
    assert build_run_dir("multivariate", "DLinear", "ED2022", 192, 96, True) == base
    assert build_run_dir("multivariate", "DLinear", "ED2022", 192, 96, True, ablation_id=3) == \
        os.path.join(base, "ablation_3")


def test_build_run_dir_tag_level_only_when_set():
    base = os.path.join("logs", "multivariate", "PatchTST", "uh2022", "L336_O336_cp")
    assert build_run_dir("multivariate", "PatchTST", "UH2022", 336, 336, True, tag="") == base
    assert build_run_dir("multivariate", "PatchTST", "UH2022", 336, 336, True, tag="smoke") == \
        os.path.join(base, "smoke")


def test_iter_run_dirs_finds_parents_of_seed_logs(tmp_path):
    logs = tmp_path / "logs"
    runs = {
        os.path.join("multivariate", "DLinear", "ed2022", "L192_O96_cp"): ["69421", "69422"],
        os.path.join("univariate", "RF_EMF_LSTM", "det", "L192_O96_pf"): ["69421"],
    }
    for run, seeds in runs.items():
        for s in seeds:
            d = logs / run / s
            d.mkdir(parents=True)
            (d / "log.json").write_text("{}")
    found = {os.path.relpath(p, str(logs)) for p in iter_run_dirs(str(logs))}
    assert found == set(runs)
