import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "results"))
from emfres.fig_curves import find_run_dirs


def _make_run(logs, rel, seeds):
    for s in seeds:
        d = os.path.join(logs, rel, s)
        os.makedirs(d)
        with open(os.path.join(d, "log.json"), "w") as f:
            f.write("{}")


def test_find_run_dirs_filters_by_model_component(tmp_path):
    logs = str(tmp_path / "logs")
    _make_run(logs, "multivariate/DLinear/ed2022/L192_O96_cp", ["69421", "69422"])
    _make_run(logs, "multivariate/PatchTST/ed2022/L192_O96_cp", ["69421"])
    found = find_run_dirs(logs, "DLinear")
    assert [os.path.relpath(d, logs) for d in found] == \
        [os.path.normpath("multivariate/DLinear/ed2022/L192_O96_cp")]


def test_find_run_dirs_matches_rf_emf_prefixed_baselines(tmp_path):
    logs = str(tmp_path / "logs")
    _make_run(logs, "univariate/RF_EMF_Transformer/det/L192_O96_pf", ["69421"])
    found = find_run_dirs(logs, "Transformer")
    assert [os.path.relpath(d, logs) for d in found] == \
        [os.path.normpath("univariate/RF_EMF_Transformer/det/L192_O96_pf")]


def test_find_run_dirs_honours_exclude(tmp_path):
    logs = str(tmp_path / "logs")
    _make_run(logs, "multivariate/DLinear/ed2022/L192_O96_cp/cptest", ["69421"])
    assert find_run_dirs(logs, "DLinear", exclude=["cptest"]) == []
