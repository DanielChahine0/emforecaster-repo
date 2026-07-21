"""Per-band ("univariate") driver for the conformal-prediction grid.

The multivariate grid trains ONE model per (model, dataset, window) cell that
forecasts all 27/29 frequency bands jointly. This driver is its univariate
counterpart over the *same* cells: it trains one independent model PER BAND, then
averages the per-band test metrics back into a single number per cell, so a
univariate cell is directly comparable to the multivariate cell beside it.

How a band is selected. data.single_channel + data.target_channel pin one band,
but the loader only honours them inside data.univariate mode
(multivariate/utils/dataloading.py). That same mode skips the num_channels
auto-inference, so num_channels is pinned to 1 here -- otherwise the model would
build RevIN for 27 channels and then be handed 1. build_run_dir has no channel
level either, so each band runs under exp.id="ch<i>"; without that tag every band
of a cell would overwrite the same directory and the resume check would report the
cell finished after the first band.

Why subprocesses. Bands are fully independent models, so they run concurrently in
a bounded pool instead of 27 sequential trainings. Each worker is a fresh
`python main.py <job> --ablation {...}` process: its own CUDA context, no shared
interpreter state, and one band crashing cannot take down the sweep. A single GPU
time-slices them, so the speedup is real but sublinear -- raise --jobs until the
GPU saturates, lower it if you hit OOM.
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from rich.console import Console
from rich.table import Table

from forecasting.grid import MARKER, SEED
from forecasting.grids import GRIDS, dataset_num_channels
from forecasting.run_path import build_run_dir, PER_CHANNEL_SETUP

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Per-band runs land here because exp.init_logger switches the <setup> folder on
# data.single_channel to this same value. Both read PER_CHANNEL_SETUP so the
# resume/aggregation paths and what exp.py writes can never drift apart.
LOG_SETUP = PER_CHANNEL_SETUP

# Test metrics averaged across bands to form a cell's headline numbers.
CELL_METRICS = ("mse", "mae", "interval_width_mean", "ic", "jc")


def channel_tag(channel):
    """Run-dir tag isolating one band, since build_run_dir has no channel level."""
    return f"ch{channel}"


def worker_budget(jobs, cpu_count=None):
    """(dataloader workers, intra-op threads) for one band run in a pool of `jobs`.

    hardware.build_profile hardcodes num_workers=4 on CUDA, which is right for a
    single training but oversubscribes badly in a pool: 8 concurrent runs would
    spawn 32 loader processes on a 24-core box, and torch would size each
    process's thread pool for the whole machine on top of that. Split the cores
    across the pool instead, reserving one per run for the training process, so
    total processes stay at or under the core count.
    """
    cores = cpu_count or os.cpu_count() or 1
    jobs = max(1, jobs)
    workers = min(4, max(0, (cores - jobs) // jobs))
    threads = max(1, min(4, cores // jobs))
    return workers, threads


def cell_dir(model_id, dataset, seq_len, pred_len, logs_dir="logs"):
    """Directory holding a cell's per-band run dirs (and its aggregate.json)."""
    return build_run_dir(LOG_SETUP, model_id, dataset, seq_len, pred_len,
                         conf_on=True, logs_dir=logs_dir)


def channel_run_dir(model_id, dataset, seq_len, pred_len, channel, logs_dir="logs"):
    return build_run_dir(LOG_SETUP, model_id, dataset, seq_len, pred_len,
                         conf_on=True, tag=channel_tag(channel), logs_dir=logs_dir)


def channel_log_path(model_id, dataset, seq_len, pred_len, channel, seed=SEED,
                     logs_dir="logs"):
    run_dir = channel_run_dir(model_id, dataset, seq_len, pred_len, channel, logs_dir)
    return os.path.join(run_dir, str(seed), "log.json")


def _read_log(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def channel_done(model_id, dataset, seq_len, pred_len, channel, seed=SEED,
                 logs_dir="logs"):
    """Whether one band already trained through test + conformal recalibration.

    Same marker the joint grid resumes on, just checked per band, so an
    interrupted sweep restarts only the bands that did not finish.
    """
    log = _read_log(channel_log_path(model_id, dataset, seq_len, pred_len, channel,
                                     seed, logs_dir))
    return log is not None and MARKER in log


def channel_ablation(dataset, seq_len, pred_len, channel, alpha, seed=SEED,
                     workers=None):
    """Config overrides turning a joint multivariate job into one band's run.

    `workers` overrides the dataloader worker count that hardware.normalize_config
    picked for a solo run; main.py applies the ablation after that call, so this
    wins.
    """
    ablation = {
        "global.data.dataset": dataset,
        "global.data.seq_len": seq_len,
        "global.data.pred_len": pred_len,
        # univariate=True is what makes the loader honour single_channel at all.
        "global.data.univariate": True,
        "global.data.single_channel": True,
        "global.data.target_channel": channel,
        # In univariate mode the loader returns pre-windowed (x, y) pairs rather
        # than a (channels, time) array, which only UnivariateForecastingDataset
        # accepts; leaving the joint config's "forecasting" class here hands a
        # tuple to ForecastingDataset and dies in torch.from_numpy.
        "global.sl.dataset_class": "univariate_forecasting",
        # num_channels auto-inference is skipped in univariate mode, so pin it to
        # 1 or the model builds for 27 channels and receives 1.
        "global.data.num_channels": 1,
        "global.conf.conf": True,
        "global.conf.alpha": alpha,
        # Without a distinct tag every band overwrites the same run directory.
        "global.exp.id": channel_tag(channel),
        "global.exp.neptune": False,
        "global.exp.seed_list": [seed],
        "global.exp.seed": seed,
    }
    if workers is not None:
        ablation["global.data.num_workers"] = workers
    return ablation


def run_channel(job, dataset, seq_len, pred_len, channel, alpha, seed=SEED,
                workers=None, threads=None):
    """Train one band in its own process. Returns (ok, error_tail)."""
    ablation = channel_ablation(dataset, seq_len, pred_len, channel, alpha, seed,
                                workers=workers)
    cmd = [sys.executable, os.path.join(_REPO_ROOT, "main.py"), job,
           "--ablation", json.dumps(ablation)]
    env = dict(os.environ)
    if threads is not None:
        # Each worker would otherwise size its BLAS/OMP pools for the whole
        # machine, so `jobs` processes fight over the same cores.
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
    # Captured, not streamed: `jobs` processes writing to one terminal interleave
    # into unreadable noise. Only the tail of a failure is surfaced.
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True,
                          env=env)
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout or "").strip()[-1200:]


def _scalar(value):
    """Numeric value of a logged metric.

    epoch_logger stores some metrics as a per-epoch list and others as a bare
    scalar; the test metrics we average are written once, so the last element is
    the value in either shape.
    """
    if isinstance(value, (list, tuple)):
        value = value[-1] if value else None
    if isinstance(value, bool) or value is None:
        return None
    return float(value) if isinstance(value, (int, float)) else None


def aggregate_cell(model_id, dataset, seq_len, pred_len, num_channels, seed=SEED,
                   logs_dir="logs", write=True):
    """Average the per-band test metrics into one cell's headline numbers.

    Keeps the per-band vectors beside the means so a bad band stays visible, and
    reports which bands are missing rather than silently averaging over whichever
    ones happened to finish.
    """
    per_band = {metric: [] for metric in CELL_METRICS}
    missing = []
    for channel in range(num_channels):
        log = _read_log(channel_log_path(model_id, dataset, seq_len, pred_len,
                                         channel, seed, logs_dir))
        if log is None or MARKER not in log:
            missing.append(channel)
            continue
        for metric in CELL_METRICS:
            # sl_test/mse is the channel-averaged MSE, which for a single band is
            # that band's MSE. sl_test/loss is the same number under an MSE
            # criterion and covers a log written before the mse key existed.
            raw = log.get(f"sl_test/{metric}")
            if raw is None and metric == "mse":
                raw = log.get("sl_test/loss")
            per_band[metric].append(_scalar(raw))

    summary = {
        "model_id": model_id,
        "dataset": dataset,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "seed": seed,
        "num_channels": num_channels,
        "channels_complete": num_channels - len(missing),
        "missing_channels": missing,
        "complete": not missing,
    }
    for metric in CELL_METRICS:
        values = [v for v in per_band[metric] if v is not None]
        summary[f"{metric}_mean"] = sum(values) / len(values) if values else None
        summary[f"{metric}_per_channel"] = per_band[metric]

    if write and summary["channels_complete"]:
        target = cell_dir(model_id, dataset, seq_len, pred_len, logs_dir)
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, "aggregate.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def run_per_channel_grid(setup="univariate", jobs=8, dry_run=False, logs_dir="logs",
                         windows=None):
    """Run every band of every cell, then aggregate. Returns the failed bands.

    windows overrides the grid's window list (already validated by the caller);
    None runs all of them.
    """
    console = Console()
    grid_def = GRIDS[setup]
    alpha = grid_def["alpha"]
    windows = grid_def["windows"] if windows is None else windows
    channels_by_dataset = {d: dataset_num_channels(d) for d in grid_def["datasets"]}

    cells = [(m, d, w)
             for m in grid_def["models"]
             for d in grid_def["datasets"]
             for w in windows]

    # Flatten every (cell, band) into one work list so the pool stays saturated.
    # Draining a cell before starting the next would idle workers on that cell's
    # stragglers, once per cell, 60 times over.
    work, skipped = [], 0
    for (label, model_id, job), dataset, (seq_len, pred_len) in cells:
        for channel in range(channels_by_dataset[dataset]):
            if channel_done(model_id, dataset, seq_len, pred_len, channel,
                            logs_dir=logs_dir):
                skipped += 1
                continue
            work.append((label, model_id, job, dataset, seq_len, pred_len, channel))

    workers, threads = worker_budget(jobs)
    total_bands = sum(channels_by_dataset[d] for _, d, _ in cells)
    console.log(f"===== {setup} per-channel grid =====")
    console.log(f"  cells:    {len(cells)} ({len(grid_def['models'])} models x "
                f"{len(grid_def['datasets'])} datasets x {len(windows)} windows)")
    console.log(f"  bands:    {dict(channels_by_dataset)}")
    console.log(f"  runs:     {total_bands} total, {skipped} already complete, "
                f"{len(work)} to run")
    console.log(f"  parallel: {jobs} concurrent band trainings "
                f"({workers} dataloader workers + {threads} threads each, "
                f"{os.cpu_count()} cores)")

    if dry_run:
        for label, model_id, job, dataset, seq_len, pred_len, channel in work:
            console.log(f"[DRY ] {label:<12} {dataset}/L{seq_len}/O{pred_len}/"
                        f"{channel_tag(channel)} -> {job}")
        console.log(f"[DRY ] {len(work)} band runs planned")
        return []

    failed, done = [], 0

    def _train(item):
        label, model_id, job, dataset, seq_len, pred_len, channel = item
        ok, err = run_channel(job, dataset, seq_len, pred_len, channel, alpha,
                              workers=workers, threads=threads)
        return item, ok, err

    # Threads, not processes: each one only blocks on subprocess.run, so the GIL
    # is irrelevant and max_workers is exactly the training concurrency.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for item, ok, err in pool.map(_train, work):
            label, model_id, job, dataset, seq_len, pred_len, channel = item
            done += 1
            tag = (f"[{done:4d}/{len(work)}] {label:<12} "
                   f"{dataset}/L{seq_len}/O{pred_len}/{channel_tag(channel)}")
            if ok:
                console.log(f"[ OK ] {tag}")
            else:
                console.log(f"[FAIL] {tag}")
                console.log(f"       {err.splitlines()[-1] if err else 'no output'}")
                failed.append(tag)

    # Aggregate every cell, including ones fully satisfied from a previous run.
    console.log("")
    console.log(f"===== {setup} per-channel grid summary =====")
    table = Table(show_header=True, header_style="bold")
    for column in ("model", "dataset", "window", "bands", "MSE (mean)", "MAE (mean)"):
        table.add_column(column)

    incomplete = []
    for (label, model_id, job), dataset, (seq_len, pred_len) in cells:
        summary = aggregate_cell(model_id, dataset, seq_len, pred_len,
                                 channels_by_dataset[dataset], logs_dir=logs_dir)
        if not summary["complete"]:
            incomplete.append(
                f"{label} {dataset}/L{seq_len}/O{pred_len} "
                f"({summary['channels_complete']}/{summary['num_channels']} bands)"
            )
        mse, mae = summary["mse_mean"], summary["mae_mean"]
        table.add_row(
            label, dataset, f"L{seq_len}_O{pred_len}",
            f"{summary['channels_complete']}/{summary['num_channels']}",
            f"{mse:.6f}" if mse is not None else "-",
            f"{mae:.6f}" if mae is not None else "-",
        )
    console.print(table)

    console.log(f"  band runs failed: {len(failed)}")
    for cell in failed:
        console.log(f"    - {cell}")
    # Averaging a partial cell would quietly understate or overstate it, so say so.
    console.log(f"  incomplete cells: {len(incomplete)}")
    for cell in incomplete:
        console.log(f"    - {cell}")
    console.log(f"  per-cell aggregates written to {logs_dir}/{LOG_SETUP}/"
                "<model>/<dataset>/L<seq>_O<pred>_cp/aggregate.json")
    return failed
