"""Another user's GPU processes: the windows, the runs they overlap, the flags.

A timing flag that is wrong either way misleads: a run marked foreign that had
the card to itself makes a fair time look suspect, and one left unmarked
makes an inflated time look like a cost. These pin how the sweep's GPU log is
read into windows, how a run's training window is held against them, and what
the frames then carry, on small logs whose answer is known.
"""

import json
from datetime import datetime

import pytest

from emfres import cli, contention as C
from emfres.runlogs import MARKER, load_runs


def T(hms: str) -> datetime:
    return datetime.strptime(f"2026-09-24 {hms}", C.TIME_FMT)


GPU_LOG = """\
=== 2026-09-24 10:00:00 before q1 per-channel EMForecaster
  FOREIGN 11, /other/venv/bin/python, 672 MiB
--- 2026-09-24 10:00:00 [q1 per-channel EMForecaster] compute apps changed
  FOREIGN 11, /other/venv/bin/python, 672 MiB
  own     21, /repo/venv/bin/python, 1034 MiB
--- 2026-09-24 10:03:00 [q1 per-channel EMForecaster] compute apps changed
  own     21, /repo/venv/bin/python, 1034 MiB
--- 2026-09-24 10:10:00 [q1 per-channel EMForecaster] compute apps changed
  own     21, /repo/venv/bin/python, 1034 MiB
  FOREIGN 12, /other/venv/bin/python, 20932 MiB
--- 2026-09-24 10:12:00 [q1 per-channel EMForecaster] compute apps changed
  own     21, /repo/venv/bin/python, 1034 MiB
  FOREIGN 13, /other/venv/bin/python, 900 MiB
--- 2026-09-24 10:15:00 [q1 per-channel EMForecaster] compute apps changed
  own     21, /repo/venv/bin/python, 1034 MiB
"""

SWEEP_LOG = """\
=== start 2026-09-24 09:59:58 at commit abc, fingerprint bb6be0a379d9
=== q1 per-channel EMForecaster  2026-09-24 10:00:00
=== DONE 2026-09-24 10:20:00  complete new runs per block: q1 1
"""


def test_a_state_holds_until_the_next_entry_and_adjacent_windows_merge():
    c = C.derive(GPU_LOG, SWEEP_LOG)
    assert c.monitored == [(T("10:00:00"), T("10:20:00"))]      # to the DONE line
    assert not c.open and c.observed_until == T("10:20:00")
    first, second = c.windows
    assert (first.start, first.end, first.peak_mib, first.processes) == \
        (T("10:00:00"), T("10:03:00"), 672, 1)
    # 10:10-10:12 and 10:12-10:15 are one window of two processes.
    assert (second.start, second.end, second.peak_mib, second.processes) == \
        (T("10:10:00"), T("10:15:00"), 20932, 2)


def test_a_run_overlaps_by_the_fraction_of_its_training_window():
    c = C.derive(GPU_LOG, SWEEP_LOG)
    assert c.overlap(T("10:01:00"), 240) == pytest.approx(0.5)  # 10:01-10:05, 2 of 4 min
    assert c.overlap(T("10:04:00"), 300) == 0.0                   # between the windows
    assert c.overlap(T("10:16:00"), 60) == 0.0
    assert c.overlap(T("09:00:00"), 600) is None                  # before any record
    assert c.overlap(None, 60) is None and c.overlap(T("10:04:00"), None) is None


def test_a_sweep_still_running_is_monitored_to_until_and_says_so():
    running = SWEEP_LOG.splitlines()[0] + "\n"
    c = C.derive(GPU_LOG, running, until=T("10:30:00"))
    assert c.open and c.monitored == [(T("10:00:00"), T("10:30:00"))]
    # Without an until it ends at its last entry, never later.
    assert C.derive(GPU_LOG, running).monitored == [(T("10:00:00"), T("10:15:00"))]


def test_a_killed_sweep_ends_at_its_last_entry_and_a_relaunch_starts_a_span():
    gpu = GPU_LOG + ("=== 2026-09-24 11:00:00 before q2 cluster\n"
                     "  FOREIGN 14, /other/venv/bin/python, 500 MiB\n")
    sweep = (SWEEP_LOG.splitlines()[0] + "\n"            # killed: no STOP, no DONE
             "=== start 2026-09-24 10:59:59 at commit abc, fingerprint bb6be0a379d9\n"
             "=== STOP 2026-09-24 11:05:00: q2 training exited 1\n")
    c = C.derive(gpu, sweep)
    assert c.monitored == [(T("10:00:00"), T("10:15:00")), (T("11:00:00"), T("11:05:00"))]
    assert c.windows[-1].start == T("11:00:00") and c.windows[-1].end == T("11:05:00")
    assert c.overlap(T("10:40:00"), 60) is None                  # between the launches
    assert not c.open


def test_an_unreadable_entry_leaves_its_interval_unmonitored():
    gpu = GPU_LOG.replace("--- 2026-09-24 10:03:00 [q1 per-channel EMForecaster] "
                          "compute apps changed\n",
                          "--- 2026-09-24 10:03:00 [q1 per-channel EMForecaster] "
                          "compute apps changed\n  FOREIGN No devices were found,\n")
    c = C.derive(gpu, SWEEP_LOG)
    assert c.monitored[0] == (T("10:00:00"), T("10:03:00"))
    assert c.overlap(T("10:05:00"), 60) is None


TAIL_GPU_LOG = """\
=== 2026-09-24 12:00:00 [tail-1] before q7 per-channel August bands
  own-mps 31, nvidia-cuda-mps-server, 30 MiB
--- 2026-09-24 12:01:00 [tail-1 q7] compute apps changed
  own-mps 31, nvidia-cuda-mps-server, 30 MiB
  FOREIGN 41, /other/venv/bin/python, 800 MiB (user someone)
--- 2026-09-24 12:02:00 [tail-1 q7] compute apps changed
  (nvidia-smi: Unable to determine the device handle)
--- 2026-09-24 12:04:00 [tail-1 q7] compute apps changed
  own-mps 31, nvidia-cuda-mps-server, 30 MiB
"""

TAIL_SWEEP_LOG = """\
=== launch 2026-09-24 11:59:00 [tail-1]
=== start 2026-09-24 11:59:30 at commit abc, fingerprint bb6be0a379d9, JOBS=12
=== DONE 2026-09-24 12:10:00  399 runs; 1 foreign GPU process(es) seen while training
"""


def test_the_tail_reruns_log_reads_the_same_way_and_joins_the_sweeps():
    tail = C.derive(TAIL_GPU_LOG, TAIL_SWEEP_LOG, source=".personal/tail_gpu_contention.log")
    # own-mps is ours; a FOREIGN entry names its user; nvidia-smi's error is a gap.
    assert tail.monitored == [(T("12:00:00"), T("12:02:00")), (T("12:04:00"), T("12:10:00"))]
    (w,) = tail.windows
    assert (w.start, w.end, w.peak_mib, w.processes) == (T("12:01:00"), T("12:02:00"), 800, 1)
    both = C.combine([C.derive(GPU_LOG, SWEEP_LOG, source="a"), tail])
    assert both.source == "a; .personal/tail_gpu_contention.log"
    assert len(both.windows) == 3 and both.observed_until == T("12:10:00")
    assert both.overlap(T("12:01:30"), 60) == pytest.approx(0.5)
    assert both.overlap(T("10:01:00"), 240) == pytest.approx(0.5)
    assert C.combine([]).windows == []


def test_the_flag_names_every_source_of_contention():
    assert C.flag(1, 0.0) == "none" and C.flag(1, None) == "none"
    assert C.flag(8, None) == "concurrent"
    assert C.flag(1, 0.25) == "foreign"
    assert C.flag(8, 0.25) == "concurrent;foreign"
    assert C.flag(None, None) == "?"
    assert C.tokens(["foreign", "none", "concurrent;foreign", ""]) == "none;concurrent;foreign"


def test_the_spec_round_trips_and_carries_its_derivation():
    c = C.derive(GPU_LOG, SWEEP_LOG, source=".personal/ccm_gpu_contention.log")
    text = C.dump(c, [{"setup": "cluster", "model": "LSTM", "runs": 3, "foreign": 1}])
    assert text.startswith("# Another user's processes on the GPU")
    back = C.loads(text)
    assert (back.monitored, back.windows, back.source, back.observed_until, back.open) == \
        (c.monitored, c.windows, c.source, c.observed_until, c.open)
    import yaml
    assert yaml.safe_load(text)["cells"] == [{"setup": "cluster", "model": "LSTM",
                                              "runs": 3, "foreign": 1}]
    assert "monitored: []" in C.dump(C.Contention([], []))


def test_the_committed_spec_loads_and_every_window_is_monitored():
    c = cli.spec_contention()
    assert c is not None and c.windows and c.source == ".personal/ccm_gpu_contention.log"
    for w in c.windows:
        assert w.start < w.end
        assert c.monitored_seconds(w.start, w.end) == (w.end - w.start).total_seconds()


def _log(started, seconds, concurrency, **extra):
    log = {"parameters/exp/model_id": "LSTM", "parameters/data/dataset": "ED2022",
           "parameters/data/seq_len": 336, "parameters/data/pred_len": 96,
           "parameters/conf/conf": True, "sl_train/loss": [1.0, 0.9],
           "sl_val/loss": [0.8, 0.7], "sl_test/loss": [0.5], "sl_test/mae": [0.4],
           "parameters/exp/time": started.strftime("%Y-%m-%d_%H-%M-%S"),
           "parameters/running_time": f"{seconds}s",
           "parameters/exp/concurrency": concurrency, MARKER: [1.5]}
    log.update(extra)
    return log


def test_a_loaded_run_knows_its_training_window_and_its_flag(tmp_path):
    c = C.derive(GPU_LOG, SWEEP_LOG)
    run = tmp_path / "LSTM" / "ed2022" / "L336_O96_cp" / "1"
    run.mkdir(parents=True)
    (run / "log.json").write_text(json.dumps(_log(T("10:01:00"), 240, 8)))
    (r,) = load_runs(tmp_path, [], contention=c)
    assert r.started == T("10:01:00") and r.running_time == 240
    assert r.foreign_overlap == pytest.approx(0.5) and r.contended == "concurrent;foreign"
    # No windows: no overlap, and the flag from the concurrency stamp alone.
    (r,) = load_runs(tmp_path, [])
    assert r.foreign_overlap is None and r.contended == "concurrent"
