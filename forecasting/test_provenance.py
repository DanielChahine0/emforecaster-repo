import subprocess
import types

import pytest

from forecasting import provenance
# The exp group of each package's config, logged as parameters/exp/*.
from forecasting.multivariate.config.config import Experiment as MultivariateExp
from forecasting.univariate.config.config import Experiment as UnivariateExp


def test_run_provenance_has_every_contract_field():
    p = provenance.run_provenance(None)
    assert set(p) == {"git_commit", "git_dirty", "hostname", "gpu_name",
                      "torch_version", "concurrency", "cuda_mps",
                      "cuda_mps_pipe_dir"}
    assert p["concurrency"] == 1
    assert isinstance(p["git_dirty"], bool)
    assert isinstance(p["cuda_mps"], bool)
    assert p["git_commit"] == "unknown" or len(p["git_commit"]) == 40


@pytest.mark.parametrize("exp_config", [MultivariateExp, UnivariateExp])
def test_provenance_overwrites_no_config_field_but_concurrency(exp_config):
    # init_logger writes these after the config under the same parameters/exp/
    # prefix, so a shared name would replace the config's value in log.json:
    # "mps" would put the CUDA MPS stamp over the Apple-silicon switch.
    # concurrency is the one deliberate overlap, the same value either way.
    shared = set(provenance.run_provenance(None)) & set(exp_config.model_fields)
    assert shared == {"concurrency"}


def test_run_provenance_reads_concurrency_from_the_config():
    args = types.SimpleNamespace(exp=types.SimpleNamespace(concurrency=8))
    assert provenance.run_provenance(args)["concurrency"] == 8


def test_git_commit_matches_the_checked_out_head():
    head = subprocess.run(["git", "-C", provenance._REPO_ROOT, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert provenance.git_commit() == head


def test_git_dirty_ignores_logs_and_results():
    # The status pathspec is limited to code paths, so a rebuilt results/ or a
    # new run under logs/ never marks a training run as dirty.
    assert set(provenance.CODE_PATHS) == {"forecasting", "main.py", "pyproject.toml",
                                          "TimeSeriesCCM-main"}


def test_git_never_takes_the_index_lock(monkeypatch):
    # A per-run `git status` killed by its timeout must not be able to orphan
    # .git/index.lock, so every call runs with optional locks off.
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(kwargs.get("env", {}).get("GIT_OPTIONAL_LOCKS"))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    provenance.git_commit()
    provenance.git_dirty()
    assert seen and all(v == "0" for v in seen)


@pytest.fixture
def cuda(monkeypatch):
    """Pretend CUDA is usable, so only the MPS daemon decides the stamp."""
    monkeypatch.setattr(provenance.torch.cuda, "is_available", lambda: True)


def _serve(pipe_dir):
    """Leave what a running MPS control daemon leaves in its pipe directory."""
    pipe_dir.mkdir(exist_ok=True)
    (pipe_dir / "control").touch()


def test_cuda_mps_needs_a_daemon_in_the_pipe_directory(cuda, monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path))
    assert provenance.cuda_mps() is False
    _serve(tmp_path)
    assert provenance.cuda_mps() is True
    assert provenance.run_provenance(None)["cuda_mps"] is True


def test_cuda_mps_reads_the_default_pipe_directory_when_none_is_set(
        cuda, monkeypatch, tmp_path):
    # A daemon started without CUDA_MPS_PIPE_DIRECTORY listens in
    # /tmp/nvidia-mps, and a client given none looks for it there.
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    monkeypatch.setattr(provenance, "MPS_DEFAULT_PIPE_DIRECTORY", str(tmp_path))
    assert provenance.cuda_mps() is False
    _serve(tmp_path)
    assert provenance.cuda_mps() is True


def test_the_pipe_directory_is_stamped_beside_the_mps_flag(monkeypatch, tmp_path):
    # With or without CUDA, so a run can be matched to its launcher's daemon.
    monkeypatch.setattr(provenance.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path))
    assert provenance.run_provenance(None)["cuda_mps_pipe_dir"] == str(tmp_path)
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY")
    assert (provenance.run_provenance(None)["cuda_mps_pipe_dir"]
            == provenance.MPS_DEFAULT_PIPE_DIRECTORY == "/tmp/nvidia-mps")


def test_cuda_mps_ignores_a_daemon_on_another_pipe_directory(
        cuda, monkeypatch, tmp_path):
    # The private-pipe launcher on a shared box: a run given its own directory
    # does not attach to a daemon serving the default one.
    monkeypatch.setattr(provenance, "MPS_DEFAULT_PIPE_DIRECTORY",
                        str(tmp_path / "default"))
    _serve(tmp_path / "default")
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path / "private"))
    assert provenance.cuda_mps() is False


def test_cuda_mps_is_false_without_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance.torch.cuda, "is_available", lambda: False)
    _serve(tmp_path)
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path))
    assert provenance.cuda_mps() is False


def test_cuda_mps_never_raises(monkeypatch):
    def broken():
        raise RuntimeError("no driver")

    monkeypatch.setattr(provenance.torch.cuda, "is_available", broken)
    assert provenance.cuda_mps() is False
