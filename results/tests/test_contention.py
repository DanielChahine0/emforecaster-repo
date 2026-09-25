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

from emfres import cli, contention as C, frames
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
    assert c is not None and c.windows
    assert c.source.split("; ")[0] == ".personal/ccm_gpu_contention.log"
    for w in list(c.windows) + list(c.shared):
        assert w.start < w.end
        assert c.monitored_seconds(w.start, w.end) == (w.end - w.start).total_seconds()


# The ARIMA tail (.personal/run_tail_arima.sh): what an own process runs after
# it, a line of card totals under each entry, a 15 s poll; it ran beside the
# CCM sweep, whose log (GPU_LOG) marks its bands `own` as well.
ARIMA_GPU_LOG = """\
=== 2026-09-24 10:05:00 [arima-1] before the ARIMA bands
  own     21, /repo/venv/bin/python, 1034 MiB (--setup cluster)
  gpu: used, free, util = 1034 MiB, 31000 MiB, 98 %
--- 2026-09-24 10:05:15 [arima-1] compute apps changed
  own     21, /repo/venv/bin/python, 1034 MiB (--setup cluster)
  own     51, /repo/venv/bin/python, 598 MiB (rf_emf_mv/arima)
  FOREIGN 12, /other/venv/bin/python, 20932 MiB (user someone)
  gpu: used, free, util = 22564 MiB, 9500 MiB, 99 %
=== 2026-09-24 10:11:00 [arima-1] ARIMA bands done (driver exit 0)
  own     21, /repo/venv/bin/python, 1034 MiB (--setup cluster)
  FOREIGN 51, [No data], 598 MiB (user ?)
  gpu: used, free, util = 1034 MiB, 31000 MiB, 98 %
"""

ARIMA_SWEEP_LOG = """\
=== launch 2026-09-24 10:04:50 [arima-1] pid 1 (process group 1), JOBS=12 NICE=10
=== start 2026-09-24 10:04:50 at commit abc, fingerprint bb6be0a379d9, JOBS=12
=== ARIMA per-channel bands, 83 to train, 12 concurrent, without MPS, nice 10  2026-09-24 10:05:00
=== DONE 2026-09-24 10:11:05  83 band(s)
"""


def test_the_arima_tails_log_reads_whole_with_its_notes_and_totals():
    snaps = C.parse_gpu_log(ARIMA_GPU_LOG)
    assert [s.unknown for s in snaps] == [False, False, False]
    assert [len(s.own) for s in snaps] == [1, 2, 1]
    assert snaps[1].foreign == [(12, "/other/venv/bin/python", 20932)]
    # Band 51 exited while nvidia-smi listed it: not another user's process.
    assert snaps[2].foreign == [] and snaps[2].own == [(21, "/repo/venv/bin/python", 1034)]
    tail = C.derive(ARIMA_GPU_LOG, ARIMA_SWEEP_LOG, source="arima")
    assert tail.monitored == [(T("10:05:00"), T("10:11:05"))]      # to its DONE line
    assert tail.launches == [(T("10:05:00"), T("10:11:05"))]


def test_two_launches_at_once_share_the_card_and_count_as_contended():
    # The CCM sweep's log with no foreign entry: its own view of 10:05-10:11
    # shows only `own` processes, the ARIMA bands among them.
    quiet = "\n".join(l for l in GPU_LOG.splitlines() if "FOREIGN" not in l) + "\n"
    ccm = C.derive(quiet, SWEEP_LOG, source="ccm")
    assert ccm.windows == [] and ccm.overlap(T("10:06:00"), 120) == 0.0
    both = C.combine([ccm, C.derive(ARIMA_GPU_LOG, ARIMA_SWEEP_LOG, source="arima")])
    (s,) = both.shared
    assert (s.start, s.end, s.launches) == (T("10:05:00"), T("10:11:05"), "arima + ccm")
    assert both.overlap(T("10:06:00"), 120) == 1.0                 # inside it
    assert both.overlap(T("10:10:00"), 240) == pytest.approx(65 / 240)
    assert both.overlap(T("10:12:00"), 60) == 0.0                  # after it
    assert C.flag(1, both.overlap(T("10:06:00"), 120)) == "foreign"
    # Foreign and shared windows over the same seconds count them once: of
    # 10:04-10:14, 10:05-10:14 is contended (shared 10:05-10:11:05, the other
    # user's pid 12 from 10:05:15 in the ARIMA log, 12 and 13 from 10:10 in
    # the sweep's), 540 of 600 s, where a sum would pass 1.
    full = C.combine([C.derive(GPU_LOG, SWEEP_LOG, source="ccm"),
                      C.derive(ARIMA_GPU_LOG, ARIMA_SWEEP_LOG, source="arima")])
    assert full.foreign_seconds(T("10:04:00"), T("10:14:00")) == 540
    assert full.overlap(T("10:04:00"), 600) == pytest.approx(0.9)
    # The other user's pid 12 is in both logs: one process, not two.
    w = [w for w in full.windows if w.start <= T("10:10:00") < w.end]
    assert len(w) == 1 and w[0].processes == 2                     # pids 12 and 13
    # One launcher's launches never overlap: no shared window from one log.
    assert C.combine([ccm]).shared == []


def test_a_derive_to_an_earlier_until_leaves_a_later_launch_out():
    running = SWEEP_LOG.splitlines()[0] + "\n"
    parts = [C.derive(GPU_LOG, running, until=T("10:04:00"), source="ccm"),
             C.derive(ARIMA_GPU_LOG, ARIMA_SWEEP_LOG, until=T("10:04:00"), source="arima")]
    early = C.combine(parts)
    assert early.source == "ccm" and early.shared == [] and early.open
    assert early.monitored == [(T("10:00:00"), T("10:04:00"))]
    assert C.dump(early) == C.dump(C.derive(GPU_LOG, running, until=T("10:04:00"),
                                            source="ccm"))
    # A launch that ended after until was still running at it: the open last span.
    mid = C.derive(ARIMA_GPU_LOG, ARIMA_SWEEP_LOG, until=T("10:08:00"))
    assert mid.open and mid.monitored == [(T("10:05:00"), T("10:08:00"))]
    # A killed launch whose relaunch came after until was running at it too.
    gpu = GPU_LOG + "=== 2026-09-24 11:00:00 before q2 cluster\n"
    sweep = (running + "=== start 2026-09-24 10:59:59 at commit abc, fingerprint x\n")
    assert C.derive(gpu, sweep, until=T("10:30:00")).monitored == \
        [(T("10:00:00"), T("10:30:00"))]


def test_a_shared_window_round_trips_through_the_spec():
    both = C.combine([C.derive(GPU_LOG, SWEEP_LOG, source="ccm"),
                      C.derive(ARIMA_GPU_LOG, ARIMA_SWEEP_LOG, source="arima")])
    text = C.dump(both)
    assert "shared:\n  - {from: '2026-09-24 10:05:00', to: '2026-09-24 10:11:05', " \
           "launches: 'arima + ccm'}" in text
    back = C.loads(text)
    assert back.shared == both.shared and back.windows == both.windows
    assert back.overlap(T("10:04:00"), 600) == both.overlap(T("10:04:00"), 600)
    assert "shared: []" in C.dump(C.Contention([], []))
    # A spec written before `shared` existed loads with none.
    assert C.loads(text.replace("shared:", "unused:")).shared == []


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


def test_the_frames_carry_the_overlap_and_the_flag(tmp_path):
    c = C.derive(GPU_LOG, SWEEP_LOG)
    for seed, (start, secs, conc) in {"1": ("10:01:00", 240, 1), "2": ("10:04:00", 300, 1),
                                      "3": ("10:11:00", 60, 8)}.items():
        run = tmp_path / "multivariate" / "LSTM" / "ed2022" / "L336_O96_cp" / seed
        run.mkdir(parents=True)
        (run / "log.json").write_text(json.dumps(_log(T(start), secs, conc)))
    loaded = frames.load_all(tmp_path, [], contention=c)
    rows = {r["seed"]: r for r in frames.build_runs(tmp_path, tmp_path, [], loaded=loaded)}
    assert float(rows["1"]["foreign_overlap"]) == pytest.approx(0.5)
    assert (rows["1"]["contended"], rows["2"]["contended"], rows["3"]["contended"]) == \
        ("foreign", "none", "concurrent;foreign")
    assert rows["2"]["foreign_overlap"] == "0.0"
    (cell,) = frames.build_cells(list(rows.values()))
    assert cell["n_foreign"] == 2 and cell["contended"] == "none;concurrent;foreign"
    # Without the windows a run keeps its concurrency flag and no overlap.
    plain = {r.seed: r for r in load_runs(tmp_path / "multivariate", [], setup="multivariate")}
    assert plain["1"].foreign_overlap is None and plain["1"].contended == "none"
    assert plain["3"].contended == "concurrent"


def test_timing_names_the_contention_of_a_seeds_bands_and_its_multivariate_run():
    solo = {"seq_len": 336, "pred_len": 96, "bands": [
        {"model": "DLinear", "dataset": "ed2022", "channel": 0, "epochs": 10,
         "seconds": 10.0, "twin_epochs": 10, "twin_seconds": 20.0}]}

    def row(setup, channel, contended, **kw):
        r = {f: "" for f in frames.RUN_FIELDS}
        r.update(setup=setup, family="emf", model="DLinear", dataset="ed2022", seq_len=336,
                 pred_len=96, seed="1", conf=1, channel=channel, epochs=10,
                 running_time="20.0", contended=contended, **kw)
        return r

    rows = [row("perchannel", 0, "concurrent"),
            row("perchannel", 1, "concurrent;foreign", foreign_overlap="0.3"),
            row("multivariate", "", "none")]
    (t,) = frames.build_timing(rows, solo)
    assert (t["contended"], t["mv_contended"]) == ("concurrent;foreign", "none")


def test_a_run_past_an_open_specs_end_is_not_known_rather_than_none():
    running = SWEEP_LOG.splitlines()[0] + "\n"            # the sweep is still training
    c = C.derive(GPU_LOG, running, until=T("10:12:00"))
    assert c.open and c.observed_until == T("10:12:00")
    assert not c.past_end(T("10:05:00"), 120)             # ends 10:07
    assert c.past_end(T("10:11:00"), 180) and c.past_end(T("10:13:00"), 60)
    assert C.flag(1, None, past_end=True) == "?"
    assert C.flag(1, 0.3, past_end=True) == "foreign"     # an overlap seen: a lower bound
    assert C.flag(8, 0.0, past_end=True) == "concurrent;?"
    assert C.flag(None, None, past_end=True) == "?"
    # A closed spec ends at its sweep's DONE: a later run has no GPU record, as
    # every run before the sweep, and reads none.
    closed = C.derive(GPU_LOG, SWEEP_LOG)
    assert not closed.open and not closed.past_end(T("10:19:00"), 600)


def test_extract_flags_every_run_and_warns_when_the_spec_ends_before_a_run(
        tmp_path, monkeypatch, capsys):
    specs, logs = tmp_path / "specs", tmp_path / "logs"
    specs.mkdir()
    running = SWEEP_LOG.splitlines()[0] + "\n"            # the sweep is still training
    (specs / "contention.yaml").write_text(
        C.dump(C.derive(GPU_LOG, running, until=T("10:12:00"))))
    for seed, (start, secs) in {"1": ("10:01:00", 240), "2": ("10:11:00", 180),
                                "3": ("10:05:00", 120), "4": ("10:13:00", 60)}.items():
        run = logs / "multivariate" / "LSTM" / "toy" / "L336_O96_cp" / seed
        run.mkdir(parents=True)
        (run / "log.json").write_text(json.dumps(
            _log(T(start), secs, 1, **{"parameters/data/dataset": "TOY"})))
    # The GPU log went on after the spec: one entry, at 10:15.
    gpu, sweep = tmp_path / "gpu.log", tmp_path / "sweep.log"
    gpu.write_text(GPU_LOG)
    sweep.write_text(running)
    monkeypatch.setattr(cli, "LOG_PAIRS", [(gpu, sweep)])
    monkeypatch.setattr(cli, "SPECS", specs)
    monkeypatch.setattr(cli, "FRAMES", tmp_path / "frames")
    monkeypatch.setattr(cli, "REPO", tmp_path)
    assert cli.main(["extract", "--logs", str(logs), "--no-spec"]) == 0
    err = capsys.readouterr().err
    assert "contended timings: 1 ?, 2 foreign, 1 none" in err
    assert "its last span still open" in err
    assert ("2 runs trained past specs/contention.yaml's observed_until 2026-09-24 10:12:00 "
            "while its sweep was still running") in err
    assert "the GPU logs have 1 entries after" in err
    import csv
    with open(tmp_path / "frames" / "runs.csv") as f:
        rows = {r["seed"]: r for r in csv.DictReader(f)}
    assert rows["1"]["contended"] == "foreign" and rows["3"]["contended"] == "none"
    # 10:11-10:14 against windows to 10:12 (the spec's end): a lower bound.
    assert rows["2"]["contended"] == "foreign"
    assert float(rows["2"]["foreign_overlap"]) == pytest.approx(1 / 3)
    # 10:13-10:14, all of it after the spec's end: not known, not none.
    assert rows["4"]["contended"] == "?" and rows["4"]["foreign_overlap"] == ""
    manifest = json.loads((tmp_path / "frames" / "MANIFEST.json").read_text())
    assert manifest["contention"] == {"observed_until": "2026-09-24 10:12:00", "open": True,
                                      "past_end": 2, "gpu_log_entries_after": 1}
    assert "2 runs trained past" in cli.stale_contention(tmp_path / "frames")
    assert "1 entries after it" in cli.stale_contention(tmp_path / "frames")

    # Derived again once the sweep is done: nothing past its end, nothing after it.
    (specs / "contention.yaml").write_text(C.dump(C.derive(GPU_LOG, SWEEP_LOG)))
    sweep.write_text(SWEEP_LOG)
    assert cli.main(["extract", "--logs", str(logs), "--no-spec"]) == 0
    capsys.readouterr()
    manifest = json.loads((tmp_path / "frames" / "MANIFEST.json").read_text())
    assert manifest["contention"]["past_end"] == 0
    assert manifest["contention"]["gpu_log_entries_after"] == 0
    assert cli.stale_contention(tmp_path / "frames") == ""
    # The ? resolves: 10:13-10:14 was inside the foreign window from 10:12 to 10:15,
    # which a none would have hidden.
    with open(tmp_path / "frames" / "runs.csv") as f:
        again = {r["seed"]: r for r in csv.DictReader(f)}
    assert again["4"]["contended"] == "foreign" and float(again["4"]["foreign_overlap"]) == 1.0
