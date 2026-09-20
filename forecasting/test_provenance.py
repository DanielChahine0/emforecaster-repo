import subprocess
import types

from forecasting import provenance


def test_run_provenance_has_every_contract_field():
    p = provenance.run_provenance(None)
    assert set(p) == {"git_commit", "git_dirty", "hostname", "gpu_name",
                      "torch_version", "concurrency"}
    assert p["concurrency"] == 1
    assert isinstance(p["git_dirty"], bool)
    assert p["git_commit"] == "unknown" or len(p["git_commit"]) == 40


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
