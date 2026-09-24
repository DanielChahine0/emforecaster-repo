"""What a run records about itself, in every paradigm.

One VoIP TSMixer cell per paradigm -- multivariate, a per-channel band and a
k=2 cluster cell -- trained end to end on the CPU, the way test_cluster trains
its end-to-end cells. TSMixer because its channel-mixing layers are sized by
num_channels, so a config that records the wrong channel count cannot rebuild
the model. The VoIP files carry 6 channels against the job configs' 27, which
is what the records are checked against. Two more multivariate cells take the
other ways a run loads its splits: TSMixer with conformal prediction off, whose
test() loads the test split alone, and Persistence, which trains nothing. One
more logs to a stand-in Neptune run, for the copy of args.yaml that Neptune
keeps.
"""

import os

import matplotlib.pyplot as plt
import pytest
import torch
import yaml

from forecasting import grid, grids, perchannel, provenance
from forecasting.multivariate.exp import exp as exp_module
from forecasting.multivariate.utils.logger import convert_pydantic_types
from forecasting.multivariate.utils.models import get_model
# The end-to-end cell recipe test_cluster's fixed-protocol tests train on.
from forecasting.test_cluster import _run_voip_cell

JOB = "rf_emf_mv/tsmixer"
PERSISTENCE = "rf_emf_mv/persistence"
DATASET = "VOIP_MOBILE"
SEQ_LEN, PRED_LEN = 48, 12
BAND = 0
SPLITS = ("train", "val", "test")


def _band_settings():
    """perchannel.channel_ablation for one band, as run_grid-style overrides."""
    ablation = perchannel.channel_ablation(
        DATASET, SEQ_LEN, PRED_LEN, BAND, grids.GRIDS["voip_perchannel"]["alpha"],
        overrides=grids.GRIDS["voip_perchannel"]["overrides"])
    return {dotted.removeprefix("global."): value
            for dotted, value in ablation.items()}


# name: (job, k, settings, the channel count the model is built for)
CASES = {
    "mv": (JOB, None, None, 6),
    "band": (JOB, None, _band_settings(), 1),
    "k2": (JOB, 2, None, 6),
    "mv_conf_off": (JOB, None, {"conf.conf": False}, 6),
    "persistence": (PERSISTENCE, None, None, 6),
}
# Persistence fits nothing, so it leaves no checkpoint to rebuild.
CHECKPOINTED = [name for name, case in CASES.items() if case[0] != PERSISTENCE]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """name -> (log, run dir, Experiment) for each case."""
    out = {}
    for name, (job, k, settings, _) in CASES.items():
        root = str(tmp_path_factory.mktemp(name))
        settings = {**(settings or {}), "sl.epochs": 1}
        log, _, exp = _run_voip_cell(k, root, job=job, settings=settings)
        out[name] = (log, os.path.join(root, exp.log_dir), exp)
    return out


def _load_run_config(run_dir):
    load_config, _ = grid.setup_modules("voip")
    return load_config(os.path.join(run_dir, "args.yaml"))


def test_the_job_config_still_says_27_channels():
    # What makes the checks below mean something: the channel count the runs
    # record comes from the data, not from the config they started from.
    load_config, _ = grid.setup_modules("voip")
    assert load_config(grid.args_path("voip", JOB)).data.num_channels == 27


def test_the_extra_cells_load_their_splits_the_other_ways(runs):
    # Conformal off: test() replaced the train and val loaders with the test
    # split's, so their counts are the ones supervised_train logged.
    log, _, exp = runs["mv_conf_off"]
    assert not hasattr(exp, "train_loader") and not hasattr(exp, "val_loader")
    assert grid.MARKER not in log
    # Persistence: no epoch loop and no checkpoint, but the same test().
    _, run_dir, exp = runs["persistence"]
    assert exp.args.exp.no_train
    assert not os.path.exists(os.path.join(run_dir, "supervised.pth"))


@pytest.mark.parametrize("name", CASES)
def test_args_yaml_records_the_channel_count_the_model_trained_on(runs, name):
    log, run_dir, _ = runs[name]
    channels = CASES[name][3]
    with open(os.path.join(run_dir, "args.yaml")) as f:
        assert yaml.safe_load(f)["data"]["num_channels"] == channels
    assert log["parameters/data/num_channels"] == channels


@pytest.mark.parametrize("name", CHECKPOINTED)
def test_args_yaml_rebuilds_the_model_its_checkpoint_holds(runs, name):
    # Finding C7: an args.yaml dumped before the loaders built a 27-channel
    # model that its own checkpoint does not fit.
    _, run_dir, _ = runs[name]
    state = torch.load(os.path.join(run_dir, "supervised.pth"))
    args = _load_run_config(run_dir)
    get_model(args).load_state_dict(state)

    args.data.num_channels = 27
    with pytest.raises(RuntimeError, match="size mismatch"):
        get_model(args).load_state_dict(state)


@pytest.mark.parametrize("name", CASES)
def test_args_yaml_agrees_with_the_logged_config(runs, name):
    # Field for field, the channel count included: nothing the loaders set is
    # left out of either record.
    log, run_dir, _ = runs[name]
    args = _load_run_config(run_dir)
    for group, section in args:
        for field, value in section:
            key = f"parameters/{group}/{field}"
            if key in log:
                assert convert_pydantic_types(value) == log[key], key


class _NeptuneRun:
    """A stand-in neptune.init_run() run: keeps each field's uploaded files."""

    def __init__(self):
        self.uploads = {}

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        return _NeptuneField(self, key)

    def stop(self):
        pass


class _NeptuneField:
    def __init__(self, run, key):
        self.run, self.key = run, key

    def upload(self, path):
        with open(path, "rb") as f:
            self.run.uploads.setdefault(self.key, []).append(f.read())

    def append(self, value):
        pass


def test_neptune_keeps_the_config_the_model_was_built_from(tmp_path, monkeypatch):
    # init_logger uploads args.yaml before the loaders run; the upload that
    # replaces it last must be the re-dumped copy. No grid run logs to Neptune.
    neptune_run = _NeptuneRun()
    monkeypatch.setattr(exp_module.neptune, "init_run", lambda **_: neptune_run)
    load_config, Experiment = grid.setup_modules("voip")
    args = load_config(grid.args_path("voip", JOB))
    args.data.dataset = DATASET
    args.data.seq_len, args.data.pred_len = SEQ_LEN, PRED_LEN
    args.exp.neptune = True
    args.exp.seed = grid.SEED
    args.exp.compile = False
    grid.apply_overrides(args, grids.GRIDS["voip"]["overrides"])
    args.sl.epochs = 1
    args.data.num_workers = 0
    args.data.pin_memory = False

    monkeypatch.chdir(tmp_path)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        Experiment(args).run()
    finally:
        torch.set_num_threads(threads)
        plt.close("all")
    configs = [yaml.safe_load(upload)
               for upload in neptune_run.uploads["config"]]
    assert [config["data"]["num_channels"] for config in configs] == [27, 6]


@pytest.mark.parametrize("name", CASES)
def test_every_run_logs_its_window_counts_before_its_test_metrics(runs, name):
    # So before the grid's resume marker, in the runs that log it.
    log, _, exp = runs[name]
    keys = list(log)
    first_test = min(keys.index(key) for key in keys if key.startswith("sl_test"))
    for split in SPLITS:
        key = f"parameters/data/n_{split}_windows"
        assert log[key] > 0
        assert keys.index(key) < first_test
        # Against the loaders the run still holds; the conformal-off run's
        # train and val counts are held to the other runs' below.
        loader = getattr(exp, f"{split}_loader", None)
        if loader is not None:
            assert log[key] == len(loader.dataset)


def test_every_paradigm_logs_the_same_windows(runs):
    # A band is windowed on its own column, but over the same rows and the
    # same split as the joint and cluster cells (finding PE-9).
    counts = {name: [log[f"parameters/data/n_{split}_windows"] for split in SPLITS]
              for name, (log, _, _) in runs.items()}
    assert all(count == counts["mv"] for count in counts.values()), counts


@pytest.mark.parametrize("name", CASES)
def test_every_run_stamps_whether_it_ran_under_cuda_mps(runs, name):
    # Beside, not over, the config's Apple-silicon switch parameters/exp/mps.
    log, _, exp = runs[name]
    assert log["parameters/exp/cuda_mps"] is provenance.cuda_mps()
    assert log["parameters/exp/cuda_mps_pipe_dir"] == provenance.mps_pipe_directory()
    assert log["parameters/exp/mps"] is exp.args.exp.mps
