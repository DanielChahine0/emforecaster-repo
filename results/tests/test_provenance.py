"""Provenance: which training code a run actually used.

The extractor judges a run by the training code it executed, not by the commit
id stamped on it, and resolves runs stamped on a dirty tree only through an
evidenced rule. Getting either wrong would silently merge two code versions into
one seed mean, or split one version across several, so both are pinned here.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import pytest

from emfres import cli, frames
from emfres.runlogs import (attest_commit, equivalence_table, equivalent_code,
                            infer_commit, load_runs, valid_equivalence)

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


def test_a_test_fixture_is_inert_only_beside_a_test_module():
    tests = frozenset({"forecasting"})
    assert frames._inert("forecasting/test_golden.json", tests)
    assert frames._inert("forecasting/test_windows.npz", tests)
    # No test module in that folder: a loader could be reading it.
    assert not frames._inert("forecasting/data/test_split.json", tests)
    assert not frames._inert("forecasting/test_golden.json")
    # A data file is a fixture by its test_ prefix, and only a data one is.
    assert not frames._inert("forecasting/golden.json", tests)
    assert not frames._inert("forecasting/test_config.yaml", tests)


def _git(root, *argv):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    *argv], check=True, capture_output=True)


def _commit(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c")
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_regenerating_a_fixture_keeps_the_code_version(tmp_path):
    _git(tmp_path, "init", "-q")
    base = _commit(tmp_path, {"forecasting/model.py": "W = 1\n",
                              "forecasting/test_model.py": "def test(): pass\n",
                              "forecasting/test_model.json": '{"h": "a"}\n'})
    fixture = _commit(tmp_path, {"forecasting/test_model.json": '{"h": "b"}\n'})
    stray = _commit(tmp_path, {"forecasting/data/test_split.json": "[1]\n"})
    code = _commit(tmp_path, {"forecasting/model.py": "W = 2\n"})
    fp = {c: frames.code_fingerprint(tmp_path, c) for c in (base, fixture, stray, code)}
    assert fp[base] and fp[base] == fp[fixture]
    assert fp[stray] != fp[fixture]
    assert fp[code] != fp[stray]


def test_the_golden_hash_commits_kept_the_code_version():
    # 149ec6295 and 51af0718e added and regrew forecasting/test_golden.py and its
    # fixture test_golden.json on top of f84e58dae, and touched nothing else.
    commits = ("f84e58dae", "149ec6295", "51af0718e")
    if not _in_history(*commits):
        pytest.skip("commits not in this checkout's history")
    assert len({frames.code_fingerprint(REPO, c) for c in commits}) == 1


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
        # A finished run's last key; without it the run is skipped as unfinished.
        "sl_test/interval_width_mean": [1.5],
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


# --------------------------------------------------------------- equivalence

EQUIV = [
    {"from": "bbbbbbbbbbbb", "to": "aaaaaaaaaaaa", "evidence": ["golden hashes"]},
    {"from": "cccccccccccc", "to": "bbbbbbbbbbbb", "evidence": "gate"},
]


def test_an_equivalence_maps_a_version_and_chains():
    assert equivalent_code(EQUIV, "bbbbbbbbbbbb") == "aaaaaaaaaaaa"
    assert equivalent_code(EQUIV, "cccccccccccc") == "aaaaaaaaaaaa"
    # A version no rule names is its own; an unknown one stays unknown.
    assert equivalent_code(EQUIV, "aaaaaaaaaaaa") == "aaaaaaaaaaaa"
    assert equivalent_code(EQUIV, "dddddddddddd") == "dddddddddddd"
    assert equivalent_code(EQUIV, "") == ""
    assert equivalent_code([], "bbbbbbbbbbbb") == "bbbbbbbbbbbb"
    # Exact match only: a prefix is not the version.
    assert equivalent_code(EQUIV, "bbbbbb") == "bbbbbb"


def test_a_cycle_is_refused_not_followed():
    # a -> b with b -> a would count a's runs as b and b's as a: no canonical
    # version, so neither rule applies and both versions stay distinct.
    loop = [{"from": "a", "to": "b", "evidence": ["x"]},
            {"from": "b", "to": "a", "evidence": ["x"]},
            {"from": "c", "to": "a", "evidence": ["x"]}]
    assert equivalent_code(loop, "a") == "a"
    assert equivalent_code(loop, "b") == "b"
    # A rule that only leads into the cycle keeps its own mapping.
    assert equivalent_code(loop, "c") == "a"
    table, problems = equivalence_table(loop)
    assert table == {"c": "a"}
    assert problems == ["rules form a cycle: a -> b -> a"]
    # Mapping a version to itself is a cycle of one.
    assert equivalence_table([{"from": "a", "to": "a", "evidence": ["x"]}]) == (
        {}, ["rules form a cycle: a -> a"])


def test_a_version_mapped_to_two_others_is_refused():
    fork = [{"from": "b", "to": "a", "evidence": ["x"]},
            {"from": "b", "to": "c", "evidence": ["x"]},
            {"from": "d", "to": "a", "evidence": ["x"]},
            {"from": "d", "to": "a", "evidence": ["the same rule twice"]}]
    assert equivalent_code(fork, "b") == "b"
    assert equivalent_code(fork, "d") == "a"
    table, problems = equivalence_table(fork)
    assert table == {"d": "a"}
    assert problems == ["b is mapped to 2 versions (a, c)"]


def test_an_equivalence_without_evidence_vouches_for_nothing():
    for rule in ({"from": "b", "to": "a"}, {"from": "b", "to": "a", "evidence": []},
                 {"from": "b", "to": "a", "evidence": ["  "]},
                 {"from": "", "to": "a", "evidence": ["x"]},
                 {"from": "b", "to": None, "evidence": ["x"]}, "b -> a"):
        assert not valid_equivalence(rule)
        assert equivalent_code([rule], "b") == "b"
    assert valid_equivalence({"from": "b", "to": "a", "evidence": "one line"})


def test_the_declared_equivalences_carry_evidence():
    rules = cli._provenance_rules("equivalent_code")
    assert rules, "specs/provenance.yaml declares no equivalent_code rule"
    assert all(valid_equivalence(r) for r in rules)
    assert equivalent_code(rules, "bb6be0a379d9") == "21e1374ddc2c"


# A tiny repo whose two commits differ in training code, so they fingerprint
# as two versions, with a run stamped on each (_git and _commit are above).

def _log(**extra):
    log = {
        "parameters/exp/model_id": "DLinear",
        "parameters/data/dataset": "TOY",
        "parameters/data/seq_len": 336,
        "parameters/data/pred_len": 96,
        "parameters/conf/conf": True,
        "sl_train/loss": [1.0, 0.9],
        "sl_val/loss": [0.8, 0.7],
        "sl_test/loss": [0.5],
        "sl_test/mae": [0.4],
        "sl_test/interval_width_mean": [1.5],
    }
    log.update(extra)
    return log


def _write(root, rel, log):
    run = root / rel
    run.mkdir(parents=True, exist_ok=True)
    (run / "log.json").write_text(json.dumps(log))
    return run


def _extract(tmp_path, monkeypatch, capsys, rules):
    specs = tmp_path / "specs"
    specs.mkdir(exist_ok=True)
    (specs / "provenance.yaml").write_text(json.dumps({"equivalent_code": rules}))
    monkeypatch.setattr(cli, "REPO", tmp_path)
    monkeypatch.setattr(cli, "SPECS", specs)
    monkeypatch.setattr(cli, "FRAMES", tmp_path / "frames")
    args = argparse.Namespace(logs=tmp_path / "logs", exclude=[], no_spec=True,
                              dry_run=False, epochs=False)
    assert cli.cmd_extract(args) == 0
    err = capsys.readouterr().err
    manifest = json.loads((tmp_path / "frames" / "MANIFEST.json").read_text())
    with open(tmp_path / "frames" / "runs.csv") as f:
        runs = list(csv.DictReader(f))
    return err, manifest, runs


def test_declared_versions_are_listed_but_not_warned_about(tmp_path, monkeypatch, capsys):
    _git(tmp_path, "init", "-q")
    old = _commit(tmp_path, {"forecasting/model.py": "W = 1\n"})
    new = _commit(tmp_path, {"forecasting/model.py": "W = 1  # same numerics\n"})
    fp_old, fp_new = (frames.code_fingerprint(tmp_path, c) for c in (old, new))
    assert fp_old and fp_new and fp_old != fp_new
    for seed, commit in ((1, old), (2, new)):
        _write(tmp_path, f"logs/multivariate/DLinear/toy/L336_O96_cp/{seed}",
               _log(**{"parameters/exp/git_commit": commit,
                       "parameters/exp/git_dirty": False}))

    err, manifest, runs = _extract(tmp_path, monkeypatch, capsys, [])
    assert "WARNING: the published runs span 2 training-code versions" in err
    assert {r["code_equivalent"] for r in runs} == {fp_old, fp_new}

    rule = {"from": fp_new, "to": fp_old, "evidence": ["a comment-only diff"]}
    err, manifest, runs = _extract(tmp_path, monkeypatch, capsys, [rule])
    assert "training-code versions" not in err
    assert f"(equivalent to {fp_old})" in err
    # runs.csv keeps the raw fingerprint beside the version it counts as.
    assert {(r["code_version"], r["code_equivalent"]) for r in runs} == {
        (fp_old, fp_old), (fp_new, fp_old)}
    # The manifest still lists both, and says which counts as which.
    assert set(manifest["code_versions"]) == {fp_old, fp_new}
    assert manifest["code_versions"][fp_new]["equivalent_to"] == fp_old
    assert "equivalent_to" not in manifest["code_versions"][fp_old]
    assert manifest["code_equivalents"] == {
        fp_old: {"runs": 2, "versions": sorted([fp_old, fp_new])}}
    # And the one cell reads as one version under the rule, two without it.
    with open(tmp_path / "frames" / "cells.csv") as f:
        (cell,) = list(csv.DictReader(f))
    assert cell["code_versions"] == ";".join(sorted([fp_old, fp_new]))
    assert cell["code_equivalents"] == fp_old


def test_a_rule_without_evidence_is_warned_about_and_ignored(tmp_path, monkeypatch, capsys):
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "provenance.yaml").write_text(json.dumps({"equivalent_code": [
        {"from": "b", "to": "a"}, {"from": "c", "to": "a", "evidence": ["x"]}]}))
    monkeypatch.setattr(cli, "SPECS", specs)
    assert cli.spec_equivalent() == [{"from": "c", "to": "a", "evidence": ["x"]}]
    assert "needs from, to and evidence" in capsys.readouterr().err


def test_a_cycle_or_fork_in_the_spec_is_warned_about_and_ignored(tmp_path, monkeypatch,
                                                                 capsys):
    specs = tmp_path / "specs"
    specs.mkdir()
    keep = {"from": "e", "to": "a", "evidence": ["x"]}
    (specs / "provenance.yaml").write_text(json.dumps({"equivalent_code": [
        {"from": "a", "to": "b", "evidence": ["x"]},
        {"from": "b", "to": "a", "evidence": ["x"]},
        {"from": "c", "to": "d", "evidence": ["x"]},
        {"from": "c", "to": "e", "evidence": ["x"]}, keep]}))
    monkeypatch.setattr(cli, "SPECS", specs)
    assert cli.spec_equivalent() == [keep]
    err = capsys.readouterr().err
    assert "WARNING: specs/provenance.yaml equivalent_code: rules form a cycle: a -> b -> a" in err
    assert "c is mapped to 2 versions (d, e)" in err
