"""Tests for the CCM re-score tool (forecasting/ccm_rescore.py).

Most run on one VoIP DLinear k=3 cell trained end to end on the CPU for two
epochs, the way test_cluster trains its end-to-end cells. It trained with the
fixed protocol, so its log already holds sl_test_fixed/*: a forced re-score of
its checkpoint must reproduce both protocols' test keys bit for bit, which is
the claim that the rebuild is the run. A copy with the fixed protocol stripped
from its log and checkpoint stands in for the runs that predate it. The last
test re-scores three real checkpoints under logs/cluster, when they are
present.
"""

import hashlib
import json
import os
import shutil

import numpy as np
import pytest
import torch
import yaml

from forecasting import ccm_rescore
from forecasting.ccm_rescore import FIXED_GATED, GATED, SIDECAR, TOL, rescore_run
# The end-to-end cell recipe test_cluster's fixed-protocol tests train on.
from forecasting.test_cluster import _run_voip_cell

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFORMAL = ("sl_test/ic", "sl_test/jc", "sl_test/interval_width_mean",
             "sl_test/interval_width_std")


@pytest.fixture(scope="module")
def cpu_run(tmp_path_factory):
    """The run dir of a k=3 VoIP cell trained on the CPU."""
    root = str(tmp_path_factory.mktemp("cpu_run"))
    _, _, exp = _run_voip_cell(3, root)
    return os.path.join(root, exp.log_dir)


@pytest.fixture(scope="module")
def cpu_run_k1(tmp_path_factory):
    """The run dir of the same cell at k=1, and its splits' window counts."""
    root = str(tmp_path_factory.mktemp("cpu_run_k1"))
    _, _, exp = _run_voip_cell(1, root)
    windows = {split: len(getattr(exp, f"{split}_loader").dataset)
               for split in ("train", "val", "test")}
    return os.path.join(root, exp.log_dir), windows


@pytest.fixture(autouse=True)
def one_thread():
    # The cell trained on one thread, so re-score it on one for bitwise
    # equality; these models are also far slower on a shared many-core box's
    # default thread count.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _copy(run_dir, tmp_path, name="run"):
    return shutil.copytree(run_dir, os.path.join(tmp_path, name))


def _read(path):
    with open(path) as f:
        return json.load(f)


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _write(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def _legacy(run_dir, tmp_path, name="legacy"):
    """A copy of the run as a run that predates the fixed protocol is.

    Those logged no *_fixed/* keys and saved no fixed_prob in the checkpoint.
    """
    run = _copy(run_dir, tmp_path, name)
    log_path = os.path.join(run, "log.json")
    _write(log_path, {key: value for key, value in _read(log_path).items()
                      if "_fixed/" not in key})
    checkpoint_path = os.path.join(run, "supervised.pth")
    state = torch.load(checkpoint_path)
    del state["fixed_prob"]
    torch.save(state, checkpoint_path)
    return run


def _edit_yaml(run_dir, group, field, value):
    path = os.path.join(run_dir, "args.yaml")
    with open(path) as f:
        config = yaml.safe_load(f)
    config[group][field] = value
    with open(path, "w") as f:
        yaml.safe_dump(config, f)


# ---------------------------------------------------------------------------
# Finding runs and placing sidecars
# ---------------------------------------------------------------------------


def test_find_runs_takes_run_dirs_trees_and_globs(tmp_path):
    runs = {}
    for leaf in ("a/1", "a/2", "b/1", "_archive/x/1"):
        runs[leaf] = str(tmp_path / leaf)
        os.makedirs(runs[leaf])
        (tmp_path / leaf / "log.json").write_text("{}")
    find = ccm_rescore.find_runs

    # A tree search leaves out the "_"-prefixed archive, as logs/_archive.
    assert find([str(tmp_path)]) == [runs["a/1"], runs["a/2"], runs["b/1"]]
    assert find([str(tmp_path / "a" / "*")]) == [runs["a/1"], runs["a/2"]]
    # Named explicitly, the archive is searched; a run named twice runs once.
    assert find([runs["b/1"], str(tmp_path / "_archive"), runs["b/1"]]) == [
        runs["b/1"], runs["_archive/x/1"]]
    with pytest.raises(FileNotFoundError, match="matches nothing"):
        find([str(tmp_path / "c" / "*")])


def test_sidecar_goes_in_the_run_or_under_out_at_its_repo_path(tmp_path):
    rel = os.path.join("logs", "cluster", "DLinear", "ed2022", "L336_O96_cp", "k3",
                       "69421")
    assert ccm_rescore.sidecar_path(rel) == os.path.join(rel, SIDECAR)
    out = str(tmp_path)
    assert ccm_rescore.sidecar_path(os.path.join(_REPO_ROOT, rel), out) == os.path.join(
        out, rel, SIDECAR)
    # Outside the repo, the absolute path is mirrored instead.
    elsewhere = os.path.join(os.sep, "elsewhere", "run")
    assert ccm_rescore.sidecar_path(elsewhere, out) == os.path.join(
        out, "elsewhere", "run", SIDECAR)


# ---------------------------------------------------------------------------
# Re-scoring a run trained on the CPU
# ---------------------------------------------------------------------------


def test_rescore_reproduces_the_run_bit_for_bit(cpu_run, tmp_path):
    log_path = os.path.join(cpu_run, "log.json")
    log, log_sha = _read(log_path), _sha256(log_path)
    # It logs the fixed protocol itself, so it is re-scored only when forced.
    assert rescore_run(cpu_run, "fixed", out=str(tmp_path))["status"] == "native"
    assert not os.listdir(tmp_path)
    result = rescore_run(cpu_run, "fixed", out=str(tmp_path), force=True)
    assert result["status"] == "ok"

    # The batch protocol's gated test keys, and the fixed protocol's whole test
    # block, are the run's own to the bit.
    assert result["batch_check"]["passed"]
    assert result["batch_check"]["gated"] == {key: 0.0 for key in GATED}
    assert result["fixed_check"]["gated"] == {key: 0.0 for key in FIXED_GATED}

    sidecar = _read(result["sidecar"])
    fixed_keys = [key for key in log if "_fixed/" in key]
    assert [key for key in sidecar if "/" in key] == fixed_keys
    for key in fixed_keys:
        if key.startswith("sl_test_fixed/"):
            assert sidecar[key] == log[key], key
        else:
            # Validation is shuffled: the same windows, summed in another order.
            np.testing.assert_allclose(sidecar[key], log[key], rtol=1e-6)
    checkpoint_path = os.path.join(cpu_run, "supervised.pth")
    checkpoint = torch.load(checkpoint_path)
    assert torch.equal(torch.tensor(sidecar["fixed_prob"]), checkpoint["fixed_prob"])
    assert result["fixed_check"]["fixed_prob_max_abs_dev"] == 0.0

    stamp = sidecar["rescore"]
    assert stamp["checkpoint_sha256"] == _sha256(checkpoint_path)
    for name in ("batch_check", "fixed_check", "val_check"):
        assert stamp[name] == result[name], name
    assert stamp["git_commit"] and stamp["device"] == "cpu"
    # The run itself is untouched.
    assert _sha256(log_path) == log_sha
    assert not os.path.exists(os.path.join(cpu_run, SIDECAR))


def test_batch_mode_writes_nothing_and_does_not_gate_conformal_keys(cpu_run):
    result = rescore_run(cpu_run, "batch")
    assert result["status"] == "checked"
    assert "sidecar" not in result
    assert not os.path.exists(os.path.join(cpu_run, SIDECAR))
    recorded = result["batch_check"]["recorded"]
    assert set(CONFORMAL) <= set(recorded)
    # Why they are recorded only: at k=3 the batch protocol's calibration
    # depends on how the shuffled validation loader batched the windows, so
    # even this same-machine re-score of a CPU run lands elsewhere.
    assert any(recorded[key] > 0 for key in CONFORMAL)


def test_an_existing_sidecar_is_left_alone_unless_forced(cpu_run, tmp_path):
    run = _legacy(cpu_run, tmp_path)
    result = rescore_run(run, "fixed")
    assert result["status"] == "ok" and "fixed_check" not in result
    path = result["sidecar"]
    assert path == os.path.join(run, SIDECAR)
    _write(path, {})

    skipped = rescore_run(run, "fixed")
    assert skipped["status"] == "skipped" and "batch_check" not in skipped
    assert _read(path) == {}
    assert rescore_run(run, "fixed", force=True)["status"] == "ok"
    assert "sl_test_fixed/mse" in _read(path)


def test_the_channel_count_comes_from_the_data_not_args_yaml(cpu_run, tmp_path):
    # Finding C7: args.yaml was dumped before the loaders inferred the channel
    # count, so the VoIP runs on disk record the job config's 27 for their 6
    # channels. Set explicitly, so this holds once new runs record 6.
    run = _copy(cpu_run, tmp_path)
    _edit_yaml(run, "data", "num_channels", 27)
    assert _read(os.path.join(run, "log.json"))["parameters/data/num_channels"] == 6
    assert rescore_run(run, "batch")["status"] == "checked"


def test_a_config_that_disagrees_with_its_log_is_refused(cpu_run, tmp_path):
    run = _copy(cpu_run, tmp_path)
    _edit_yaml(run, "sl", "batch_size", 32)
    with pytest.raises(ValueError, match="parameters/sl/batch_size"):
        rescore_run(run, "batch")


def test_a_failed_self_check_writes_no_sidecar_and_retires_an_old_one(
        cpu_run, tmp_path):
    run = _legacy(cpu_run, tmp_path)
    log_path, path = os.path.join(run, "log.json"), os.path.join(run, SIDECAR)
    log = _read(log_path)
    log["sl_test/mse"] = [log["sl_test/mse"][0] * (1 + 10 * TOL)]
    _write(log_path, log)

    result = rescore_run(run, "fixed")
    assert result["status"] == "fail"
    assert result["batch_check"]["worst_key"] == "sl_test/mse"
    assert not os.path.exists(path)

    # A sidecar from older code must not survive a forced re-score that fails.
    _write(path, {"from": "older code"})
    result = rescore_run(run, "fixed", force=True)
    assert result["status"] == "fail"
    assert not os.path.exists(path)
    assert result["stale"] == f"{path}.stale"
    assert _read(result["stale"]) == {"from": "older code"}


def test_a_native_run_is_gated_on_its_own_fixed_keys(cpu_run, tmp_path):
    run = _copy(cpu_run, tmp_path)
    log_path = os.path.join(run, "log.json")
    log = _read(log_path)
    log["sl_test_fixed/mse"] = [log["sl_test_fixed/mse"][0] * (1 + 10 * TOL)]
    _write(log_path, log)

    result = rescore_run(run, "fixed", force=True)
    assert result["status"] == "fail"
    assert result["batch_check"]["passed"]
    assert result["fixed_check"]["worst_key"] == "sl_test_fixed/mse"
    assert not os.path.exists(os.path.join(run, SIDECAR))


def test_the_validation_check_confirms_the_splits_at_k1(cpu_run_k1):
    run, windows = cpu_run_k1
    check = rescore_run(run, "batch")["val_check"]
    assert check["windows"] == windows
    # At k=1 a forecast ignores its batch-mates, so the unshuffled validation
    # loss is the checkpoint's logged one up to the summation order.
    assert check["logged_best_val_loss"] == min(
        _read(os.path.join(run, "log.json"))["sl_val/loss"])
    assert check["rel_dev"] < 1e-5


def test_cli_rescores_a_tree_in_a_pool_and_ignores_non_cluster_runs(
        cpu_run, tmp_path, monkeypatch):
    tree = os.path.join(tmp_path, "tree")
    legacy = _legacy(cpu_run, tree, os.path.join("cluster", "69421"))
    native = _copy(cpu_run, tree, os.path.join("cluster", "69422"))
    os.makedirs(os.path.join(tree, "multivariate", "69421"))
    _write(os.path.join(tree, "multivariate", "69421", "log.json"),
           {"parameters/cluster/enabled": False})
    out, report = os.path.join(tmp_path, "out"), os.path.join(tmp_path, "report.json")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # main() hides the GPUs

    # --jobs 2 takes the process-pool path.
    assert ccm_rescore.main([tree, "--mode", "fixed", "--out", out, "--jobs", "2",
                             "--threads", "1", "--report", report]) == 0
    entries = {entry["run"]: entry["status"] for entry in _read(report)}
    assert entries == {legacy: "ok", native: "native"}
    assert os.path.isfile(ccm_rescore.sidecar_path(legacy, out))
    assert not os.path.exists(ccm_rescore.sidecar_path(native, out))
    for run in (legacy, native):
        assert not os.path.exists(os.path.join(run, SIDECAR))


def test_cli_writes_into_runs_only_from_a_clean_tree(cpu_run, tmp_path, monkeypatch):
    run = _legacy(cpu_run, tmp_path)
    monkeypatch.setattr(ccm_rescore.provenance, "git_dirty", lambda: True)
    with pytest.raises(SystemExit) as exc:
        ccm_rescore.main([run, "--mode", "fixed"])
    assert exc.value.code == 2
    assert not os.path.exists(os.path.join(run, SIDECAR))


# ---------------------------------------------------------------------------
# Real checkpoints, trained on the GPU
# ---------------------------------------------------------------------------

# EMF and VoIP, the three models, k = 2, 3 and C, both VoIP windows. Each sits
# well inside the tolerance (at most 3.7e-6 on this box), so another CPU's
# kernels do not tip it over.
REAL_RUNS = (
    "logs/cluster/DLinear/ed2022/L336_O96_cp/k3/69421",
    "logs/cluster/PatchTST/voip_fixed/L24_O6_cp/k6/69422",
    "logs/cluster/TSMixer/voip_fixed/L48_O12_cp/k2/69421",
)


@pytest.mark.parametrize("rel", REAL_RUNS)
def test_real_checkpoint_passes_the_self_check(rel, tmp_path):
    run = os.path.join(_REPO_ROOT, rel)
    if not os.path.isfile(os.path.join(run, "supervised.pth")):
        pytest.skip(f"{rel} is not in this checkout")
    log = _read(os.path.join(run, "log.json"))

    result = rescore_run(run, "fixed", out=str(tmp_path))
    assert result["status"] == "ok", result["batch_check"]
    assert result["batch_check"]["max_rel_dev"] <= TOL
    sidecar = _read(result["sidecar"])
    test_keys = {key for key in log if key.startswith("sl_test/")}
    assert {key for key in sidecar if key.startswith("sl_test_fixed/")} == {
        key.replace("sl_test/", "sl_test_fixed/") for key in test_keys}
    assert "sl_val_fixed/loss" in sidecar
    assert np.shape(sidecar["fixed_prob"]) == (
        log["parameters/data/num_channels"], log["parameters/cluster/k_clusters"])
    assert sidecar["rescore"]["val_check"]["logged_best_val_loss"] == min(
        log["sl_val/loss"])
    assert not os.path.exists(os.path.join(run, SIDECAR))
