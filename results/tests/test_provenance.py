"""Provenance: which training code a run actually used.

The extractor judges a run by the training code it executed, not by the commit
id stamped on it, and resolves runs stamped on a dirty tree only through an
evidenced rule. Getting either wrong would silently merge two code versions into
one seed mean, or split one version across several, so both are pinned here.
"""

import json
import subprocess
from pathlib import Path

import pytest

from emfres import frames
from emfres.runlogs import attest_commit, infer_commit, load_runs

REPO = Path(__file__).resolve().parents[2]


def _in_history(*commits):
    return all(
        subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{c}^{{commit}}"],
                       capture_output=True).returncode == 0
        for c in commits)


# --------------------------------------------------------------- fingerprint

def test_results_only_commits_share_one_code_version():
    # a4e580e9 touched logs/, 46c85211 and e451d99c touched results/ only:
    # a sweep stamped across them ran one code version.
    commits = ("a4e580e9", "46c85211", "e451d99c")
    if not _in_history(*commits):
        pytest.skip("commits not in this checkout's history")
    versions = {frames.code_fingerprint(REPO, c) for c in commits}
    assert len(versions) == 1
    assert "" not in versions


def test_a_training_code_change_is_a_new_code_version():
    # Phase 0 and Phase 1 changed forecasting/ between these two.
    if not _in_history("401ba0a2", "a4e580e9"):
        pytest.skip("commits not in this checkout's history")
    assert (frames.code_fingerprint(REPO, "401ba0a2")
            != frames.code_fingerprint(REPO, "a4e580e9"))


def test_an_unknown_commit_has_no_code_version():
    assert frames.code_fingerprint(REPO, None) == ""
    assert frames.code_fingerprint(REPO, "0" * 40) == ""


def test_tests_and_docs_do_not_count_as_training_code():
    assert frames._inert("forecasting/test_voip.py")
    assert frames._inert("forecasting/multivariate/README.md")
    assert not frames._inert("forecasting/grid.py")
    # Only a test_ prefix marks a test file, not the substring.
    assert not frames._inert("forecasting/multivariate/utils/testing_helpers.py")


# --------------------------------------------------------------- rules

INFERRED = [
    {"setup": "perchannel", "from": "2026-08-01", "to": "2026-08-03", "commit": "aaa"},
    {"setup": "multivariate", "models": ["PatchTST"],
     "from": "2026-08-03 16:11", "to": "2026-08-03 16:32", "commit": "bbb"},
]


def test_inferred_rules_match_setup_model_and_a_half_open_window():
    assert infer_commit(INFERRED, "perchannel", "DLinear", "2026-08-02_10-00-00") == "aaa"
    assert infer_commit(INFERRED, "perchannel", "DLinear", "2026-08-03_00-00-00") is None
    assert infer_commit(INFERRED, "multivariate", "PatchTST", "2026-08-03_16-12-50") == "bbb"
    assert infer_commit(INFERRED, "multivariate", "DLinear", "2026-08-03_16-12-50") is None
    assert infer_commit(INFERRED, "perchannel", "DLinear", "not-a-time") is None
    assert infer_commit([], "perchannel", "DLinear", "2026-08-02_10-00-00") is None


STAMP = "6f6d60d3c2502d84d2ed795e169261b09d0db7d0"
ATTESTED = [{"stamped": "6f6d60d3", "from": "2026-09-20 10:34",
             "to": "2026-09-20 15:10", "code": "ccc"}]


def test_attestation_needs_the_stamp_prefix_and_the_window():
    assert attest_commit(ATTESTED, STAMP, "2026-09-20_12-00-00") == "ccc"
    assert attest_commit(ATTESTED, STAMP, "2026-09-20_10-33-59") is None
    assert attest_commit(ATTESTED, "deadbeef" + STAMP[8:], "2026-09-20_12-00-00") is None
    # A rule without a stamp prefix vouches for nothing.
    blank = [{"stamped": "", "from": "2000-01-01", "to": "2100-01-01", "code": "x"}]
    assert attest_commit(blank, STAMP, "2026-09-20_12-00-00") is None


# --------------------------------------------------------------- resolution

def _write_run(root, seed, **exp):
    run = root / "DLinear" / "ed2022" / "L336_O96_cp" / str(seed)
    run.mkdir(parents=True)
    log = {
        "parameters/exp/model_id": "DLinear",
        "parameters/data/dataset": "ED2022",
        "parameters/data/seq_len": 336,
        "parameters/data/pred_len": 96,
        "parameters/conf/conf": True,
        "sl_test/loss": [0.5],
        "sl_test/mae": [0.4],
    }
    log.update({f"parameters/exp/{k}": v for k, v in exp.items()})
    (run / "log.json").write_text(json.dumps(log))


def test_each_way_a_run_can_be_stamped_resolves_to_the_right_code(tmp_path):
    _write_run(tmp_path, 1, time="2026-09-21_02-00-00", git_commit="c" * 40, git_dirty=False)
    _write_run(tmp_path, 2, time="2026-09-20_12-00-00", git_commit=STAMP, git_dirty=True)
    _write_run(tmp_path, 3, time="2026-09-20_16-00-00", git_commit=STAMP, git_dirty=True)
    _write_run(tmp_path, 4, time="2026-08-02_10-00-00")
    _write_run(tmp_path, 5, time="2026-06-01_10-00-00")

    runs = {int(r.seed): r for r in load_runs(tmp_path, [], setup="perchannel",
                                                inferred=INFERRED, attested=ATTESTED)}
    got = {s: (r.commit_source, r.code_commit) for s, r in runs.items()}
    assert got == {
        1: ("log", "c" * 40),          # clean stamp: the stamp is the code
        2: ("attested", "ccc"),        # dirty stamp inside an evidenced window
        3: ("dirty", None),            # dirty stamp nobody vouched for: unknown
        4: ("inferred", "aaa"),        # unstamped, dated by a rule
        5: ("", None),                 # unstamped, no rule: unknown
    }
    # The stamp is kept as recorded even when the code resolves elsewhere.
    assert runs[2].git_commit == STAMP
