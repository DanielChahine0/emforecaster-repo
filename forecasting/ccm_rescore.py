"""Re-score saved CCM cluster checkpoints under the fixed membership, on CPU.

Cluster runs trained before the fixed-membership evaluation (models/ccm.py,
audit finding C1) logged only upstream's batch protocol, whose test forecasts
depend on which windows share a batch. This scores their saved checkpoints
under the fixed train-mean membership without retraining, through the same
Experiment methods a new cluster run calls at test time, and writes the result
beside the run:

    <run dir>/eval_fixed.json   sl_val_fixed/* and sl_test_fixed/* under the
                                key names and list format log.json uses, the
                                fitted membership, and a "rescore" block: the
                                tool's commit, the checkpoint's sha256, the
                                batch-protocol self-check below and the
                                validation check.

log.json is never edited. A run is rebuilt from its own args.yaml, checked
field by field against log.json's parameters/*, and refused if the two
disagree. The one exception is num_channels in a run that logs no window
counts (parameters/data/n_*_windows): such runs come from code that dumped
args.yaml before the loaders set the channel count (finding C7), so it is
taken from the data, as the run itself did. A run that logs its window counts
is also refused if the rebuilt loaders give other counts.

Two modes:

  batch  re-evaluate the logged batch protocol and compare it with the logged
         sl_test/*: the self-check that the rebuild is the run. Writes nothing.
  fixed  the same self-check, then the fixed protocol. The sidecar is written
         only if the self-check passes, and an existing one is left alone
         unless --force; a forced re-score that fails moves the old one to
         eval_fixed.json.stale. A run that raises is left as it was.

A run trained with the fixed protocol logs sl_test_fixed/* itself, computed
on the device it trained on; --mode fixed skips it (status "native") rather
than put a second, CPU, set of fixed numbers beside them. With --force it is
re-scored, and its own fixed keys gate the sidecar the way sl_test/* does.

What the self-check gates. The test loader is sequential, so the batch
protocol's test forecasts are a function of the checkpoint and the batch size
alone: the cell's sl_test/{loss,mse,mae} must match the log to --tol (5e-5
relative by default). A run trained on the CPU reproduces them bit for bit.
The 375 cluster runs of 2026-09 trained on a GPU with TF32 matmuls, and there
the CPU re-score sits a median 1.4e-5 and at most 9.2e-5 from the log (19
runs above 5e-5, none above 1e-4; the k=1 runs, whose forecasts ignore the
batch, spread the same way, and a plain multivariate PatchTST run re-scored
alike sits 6.0e-5 off). On one band that noise reaches 1.4e-3, so
sl_test/mse_per_channel is compared and recorded, not gated. So are the
conformal keys (ic, jc, interval widths): at k>1 each validation window's
calibration residual depends on its batch-mates in the validation loader,
which shuffles from the training run's RNG stream, and a coverage count moves
in whole windows. The fixed protocol has no such dependence: every number in
the sidecar is a function of the checkpoint and the data alone.

The test keys see the train and val splits only through the scaler, so each
run also records its window counts, which only a run that logs its own is held
to (above), and, ungated, the batch protocol's validation loss against
min(sl_val/loss), the checkpoint's epoch. At k=1 the two carry the same device
noise as the test keys when the splits are the run's: over the 87 k=1 runs a
median 1.1e-5 and at most 1.7e-4 (PatchTST on VoIP; DLinear on EMF at most
1.1e-6), against a median 2.9e-3 at k>1.

    venv/bin/python -m forecasting.ccm_rescore logs/cluster --mode batch
    venv/bin/python -m forecasting.ccm_rescore logs/cluster --mode fixed --jobs 8

A PATH is a run dir (the <seed>/ leaf holding log.json), a tree to search, or
a quoted glob; a tree search skips "_"-prefixed dirs (logs/_archive) and
non-cluster runs. --out DIR writes each sidecar under DIR, at the run's path
relative to the repo, instead of into the run; without it --mode fixed needs
a clean code tree, since a sidecar names the commit that scored it. --report
FILE keeps every run's self-checks as JSON.
"""

import argparse
import contextlib
import glob
import hashlib
import json
import multiprocessing
import os
import random
import socket
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import torch
from rich.console import Console
from torch.utils.data import DataLoader

from forecasting import provenance
from forecasting.grid import MARKER
from forecasting.multivariate.config.config import load_config
from forecasting.multivariate.exp.exp import Experiment
from forecasting.multivariate.utils.logger import convert_pydantic_types
from forecasting.multivariate.utils.models import get_criterion

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIDECAR = "eval_fixed.json"
CHECKPOINT = "supervised.pth"
TOL = 5e-5

# The batch-protocol keys the self-check gates on; the rest of sl_test/* is
# recorded only (see the module docstring).
GATED = ("sl_test/loss", "sl_test/mse", "sl_test/mae")
# Their fixed-protocol twins, for a run that logged the fixed protocol itself.
FIXED_GATED = tuple(key.replace("sl_test/", "sl_test_fixed/") for key in GATED)

# args.yaml keys whose value, in a run that dumped it before the loaders
# (finding C7), is not the one the run used.
FROM_DATA = {"parameters/data/num_channels"}
# The window counts a run logs. The code that logs them also dumps args.yaml
# again after the loaders, so a run that has them is held to FROM_DATA too.
WINDOW_KEYS = tuple(f"parameters/data/n_{split}_windows"
                    for split in ("train", "val", "test"))


def find_runs(paths):
    """Run dirs under each path, in order, without duplicates.

    A path is a run dir, a tree searched for log.json (skipping "_"-prefixed
    dirs, the archive convention of logs/_archive), or a glob of either.
    """
    runs, seen = [], set()
    for path in paths:
        matches = sorted(glob.glob(path)) if glob.has_magic(path) else [path]
        if not matches:
            raise FileNotFoundError(f"{path!r} matches nothing")
        for match in matches:
            if os.path.isfile(os.path.join(match, "log.json")):
                found = [match]
            elif os.path.isdir(match):
                found = []
                for root, dirs, files in os.walk(match):
                    dirs[:] = sorted(d for d in dirs if not d.startswith("_"))
                    if "log.json" in files:
                        found.append(root)
            else:
                raise FileNotFoundError(f"{match!r} is not a run dir or a tree")
            for run in found:
                key = os.path.realpath(run)
                if key not in seen:
                    seen.add(key)
                    runs.append(run)
    return runs


def sidecar_path(run_dir, out=None):
    """Where a run's sidecar goes: in the run, or under out at its repo path."""
    if out is None:
        return os.path.join(run_dir, SIDECAR)
    run = os.path.abspath(run_dir)
    rel = os.path.relpath(run, _REPO_ROOT)
    if rel.startswith(os.pardir):
        rel = run.lstrip(os.sep)
    return os.path.join(out, rel, SIDECAR)


def read_log(run_dir):
    with open(os.path.join(run_dir, "log.json")) as f:
        return json.load(f)


def logged_config(run_dir, log):
    """The run's config from its args.yaml, refused if log.json disagrees."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        args = load_config(os.path.join(run_dir, "args.yaml"))
    exempt = set() if any(key in log for key in WINDOW_KEYS) else FROM_DATA
    mismatched = []
    for group, section in args:
        for field, value in section:
            key = f"parameters/{group}/{field}"
            if (key in log and key not in exempt
                    and convert_pydantic_types(value) != log[key]):
                mismatched.append(key)
    if mismatched:
        raise ValueError(
            f"args.yaml and log.json disagree on {', '.join(mismatched)}"
        )
    return args


def rebuild(run_dir, log):
    """The run's Experiment on the CPU: its loaders and its restored checkpoint.

    Only what Experiment.test needs is set up: nothing is logged or written,
    and num_channels is inferred by the loaders before the model is built.
    Refused if the loaders' window counts or the model's parameter count
    differ from the ones the run logged.
    """
    args = logged_config(run_dir, log)
    if not args.cluster.enabled:
        raise ValueError("not a cluster run")
    # Hardware knobs only. The loaders' order is the sampler's, drawn in this
    # process whatever the worker count, and a pool of re-scores would
    # otherwise fork four loader processes each. Eager like every grid cell.
    args.exp.mps = False
    args.exp.compile = False
    args.data.num_workers = 0
    args.data.pin_memory = False

    exp = Experiment(args)
    exp.console = Console(quiet=True)
    exp.device = torch.device("cpu")
    exp.logger = {}
    # coverage() stats this before it checks validation_eval, which a cluster
    # run refuses; nothing is ever written to it.
    exp.score_path = os.devnull
    # As Experiment.run does: the validation loader shuffles from the global RNG.
    torch.manual_seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    random.seed(args.exp.seed)
    exp.generator = torch.Generator().manual_seed(args.exp.seed)

    exp.init_dataloaders(loader_type="all", learning_type="sl")
    for key in WINDOW_KEYS:
        if key in log and exp.logger[key] != log[key]:
            raise ValueError(
                f"rebuilt loaders give {key} = {exp.logger[key]:,}, "
                f"the run logged {log[key]:,}"
            )
    exp.init_model()
    if exp.logger["parameters/sl/num_params"] != log["parameters/sl/num_params"]:
        raise ValueError(
            f"rebuilt model has {exp.logger['parameters/sl/num_params']:,} "
            f"parameters, the run logged {log['parameters/sl/num_params']:,}"
        )
    state = torch.load(os.path.join(run_dir, CHECKPOINT), map_location="cpu")
    exp.model.load_state_dict(state)
    exp.criterion = get_criterion(args, args.sl.criterion)
    return exp


def _flags(args):
    """(mae, acc, ch_acc) as Experiment.supervised_train passes them to test()."""
    return args.exp.mae, args.exp.acc, args.exp.ch_acc


def _as_logged(exp, stats, mode, coverage=False):
    """stats under the keys, and in the list format, the run's log gives them."""
    mae, acc, ch_acc = _flags(exp.args)
    # Experiment.test adjusts these between evaluating and logging.
    ch_acc = True if exp.args.data.full_channels else ch_acc
    acc = False if exp.args.data.full_channels else acc
    exp.logger = {}
    exp.log_stats(stats, "sl", mae, acc, ch_acc, mode=mode, coverage=coverage)
    return exp.logger


def batch_protocol(exp):
    """The run's sl_test/* recomputed as Experiment.test computed them."""
    args, model = exp.args, exp.model
    model_id = args.exp.model_id
    mae, acc, ch_acc = _flags(args)
    if args.conf.conf and not args.conf.validation_eval:
        exp.evaluate(model=model, model_id=model_id, loader=exp.val_loader,
                     criterion=exp.criterion, flag="sl", mae=mae, acc=acc,
                     ch_acc=ch_acc, conformal_calibrate=True)
    stats = exp.evaluate(model=model, model_id=model_id, loader=exp.test_loader,
                         criterion=exp.criterion, flag="sl", mae=mae, acc=acc,
                         ch_acc=ch_acc, coverage=args.conf.conf)
    return _as_logged(exp, stats, "test", coverage=args.conf.conf)


def fixed_protocol(exp):
    """sl_val_fixed/* and sl_test_fixed/*, as a cluster run now logs them."""
    args, model = exp.args, exp.model
    mae, acc, ch_acc = _flags(args)
    exp.fit_fixed_membership(model)
    val, test = exp.evaluate_fixed_membership(
        model, args.exp.model_id, exp.criterion, "sl", mae, acc, ch_acc
    )
    return {**_as_logged(exp, val, "val_fixed"),
            **_as_logged(exp, test, "test_fixed", coverage=args.conf.conf)}


def val_check(exp, log):
    """The splits' window counts, and the validation loss against the log's.

    The checkpoint is the epoch of min(sl_val/loss) when the run selected on
    loss. Read through an unshuffled loader, so the value does not depend on
    the RNG; at k>1 the logged one came from the training run's shuffled
    batches, so only k=1 is expected to agree.
    """
    args = exp.args
    loader = DataLoader(exp.val_loader.dataset,
                        batch_size=exp.val_loader.batch_size, shuffle=False)
    stats = exp.evaluate(model=exp.model, model_id=args.exp.model_id,
                         loader=loader, criterion=exp.criterion, flag="sl")
    logged = log.get("sl_val/loss") if args.exp.best_model_metric == "loss" else None
    best = min(logged) if logged else None
    return {
        "val_loss": stats["loss"],
        "logged_best_val_loss": best,
        "rel_dev": abs(stats["loss"] - best) / abs(best) if best else None,
        "windows": {split: len(getattr(exp, f"{split}_loader").dataset)
                    for split in ("train", "val", "test")},
    }


def deviations(new, logged):
    """Max relative deviation of each key in new from its logged value."""
    out = {}
    for key, value in new.items():
        if key not in logged:
            out[key] = None
            continue
        a = np.asarray(value, dtype=np.float64)
        b = np.asarray(logged[key], dtype=np.float64)
        if a.shape != b.shape:
            out[key] = float("inf")
        else:
            rel = np.abs(a - b) / np.maximum(np.abs(b), np.finfo(np.float64).tiny)
            out[key] = float(rel.max()) if rel.size else 0.0
    return out


def self_check(new, logged, gated, tol):
    """A deviations() summary: whether every gated key is within tol."""
    devs = deviations(new, logged)
    missing = [key for key in gated if devs.get(key) is None]
    gated_devs = {key: devs[key] for key in gated if devs.get(key) is not None}
    worst = max(gated_devs, key=gated_devs.get) if gated_devs else None
    max_dev = gated_devs[worst] if worst else None
    return {
        "tolerance": tol,
        "passed": not missing and max_dev is not None and max_dev <= tol,
        "max_rel_dev": max_dev,
        "worst_key": worst,
        "missing": missing,
        "gated": gated_devs,
        "recorded": {key: dev for key, dev in devs.items() if key not in gated},
    }


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stamp():
    """The tool's provenance, taken once per invocation."""
    return {
        "tool": "forecasting.ccm_rescore",
        "git_commit": provenance.git_commit(),
        "git_dirty": provenance.git_dirty(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "device": "cpu",
    }


def rescore_run(run_dir, mode="fixed", out=None, tol=TOL, force=False,
                tool=None):
    """Re-score one run. Returns a result dict; raises if the run is unusable.

    status is "ok" (fixed: sidecar written), "checked" (batch: self-check
    passed), "fail" (a self-check failed; nothing written), "skipped" (fixed:
    a sidecar already exists) or "native" (fixed: the log holds the fixed
    protocol itself); the last two only while force is off. rescore() adds
    "error" for a run that raised.
    """
    log = read_log(run_dir)
    k = log.get("parameters/cluster/k_clusters")
    result = {"run": run_dir, "k": k, "model": log.get("parameters/exp/model_id")}
    path = sidecar_path(run_dir, out)
    native = any(key.startswith("sl_test_fixed/") for key in log)
    if mode == "fixed" and not force:
        if native:
            return {**result, "status": "native"}
        if os.path.exists(path):
            return {**result, "status": "skipped", "sidecar": path}
    if MARKER not in log:
        # The grid's resume marker: without it the run never finished its test.
        raise ValueError(f"the run never finished its test ({MARKER} not logged)")
    if not os.path.isfile(os.path.join(run_dir, CHECKPOINT)):
        raise FileNotFoundError(f"no {CHECKPOINT} in the run")

    exp = rebuild(run_dir, log)
    stored_prob = getattr(exp.model, "fixed_prob", None)
    result["batch_check"] = self_check(batch_protocol(exp), log, GATED, tol)
    if mode == "fixed" and result["batch_check"]["passed"]:
        fixed = fixed_protocol(exp)
        if native:
            # Its own fixed keys gate the sidecar too.
            check = result["fixed_check"] = self_check(fixed, log, FIXED_GATED, tol)
            if stored_prob is not None:
                check["fixed_prob_max_abs_dev"] = float(
                    (exp.model.fixed_prob - stored_prob).abs().max())
    # Last: its loader draws a seed from the global RNG, which the protocols'
    # shuffled validation loaders read.
    result["val_check"] = val_check(exp, log)
    passed = all(result[name]["passed"] for name in ("batch_check", "fixed_check")
                 if name in result)
    if mode == "batch":
        return {**result, "status": "checked" if passed else "fail"}
    if not passed:
        if os.path.exists(path):
            # Only a forced re-score gets here with a sidecar. This code does
            # not reproduce the run, so the older sidecar must not stand as
            # if it did.
            result["stale"] = f"{path}.stale"
            os.replace(path, result["stale"])
        return {**result, "status": "fail"}

    rescore = {**(tool or stamp()),
               "created": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
               "checkpoint": CHECKPOINT,
               "checkpoint_sha256": _sha256(os.path.join(run_dir, CHECKPOINT)),
               **{name: result[name] for name in
                  ("batch_check", "fixed_check", "val_check") if name in result}}
    sidecar = {**fixed, "fixed_prob": exp.model.fixed_prob.tolist(),
               "rescore": rescore}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Written whole, then renamed: an interrupted write must not leave a
    # sidecar that the next invocation would skip as done.
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(sidecar, f, indent=2)
    os.replace(tmp, path)
    return {**result, "status": "ok", "sidecar": path}


def _init_worker(threads):
    torch.set_num_threads(threads)


def _task(run_dir, mode, out, tol, force, tool):
    # Experiment and the conformal code print freely; keep a pool's terminal
    # down to the one line per run that rescore() prints.
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            return rescore_run(run_dir, mode, out, tol, force, tool)
    except Exception as exc:
        return {"run": run_dir, "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:]}


def rescore(runs, mode="fixed", jobs=1, threads=None, out=None, tol=TOL,
            force=False, console=None):
    """Re-score runs, jobs at a time, each on threads torch threads.

    Returns the result dicts, in the order the runs finished. jobs=1 runs in
    this process; more run in spawned processes, so no worker inherits another's
    torch state.
    """
    console = console or Console()
    cores = os.cpu_count() or 1
    threads = threads or max(1, min(4, cores // max(1, jobs)))
    tool = stamp()
    results = []

    def report(res):
        results.append(res)
        tag = {"ok": "[ OK ]", "checked": "[ OK ]", "skipped": "[SKIP]",
               "native": "[SKIP]", "fail": "[FAIL]",
               "error": "[ERR ]"}[res["status"]]
        line = f"{tag} {res['run']}"
        for name, label in (("batch_check", "batch dev"), ("fixed_check", "fixed dev"),
                            ("val_check", "val dev")):
            check = res.get(name)
            if check:
                dev = check["rel_dev" if name == "val_check" else "max_rel_dev"]
                if name == "batch_check":
                    line += f"  k={res['k']}"
                line += f"  {label} {'n/a' if dev is None else f'{dev:.1e}'}"
        if res["status"] == "native":
            line += "  (logs the fixed protocol itself)"
        if res["status"] == "error":
            line += f"  ({res['error']})"
        console.log(line)

    work = [(run, mode, out, tol, force, tool) for run in runs]
    if jobs <= 1:
        before = torch.get_num_threads()
        torch.set_num_threads(threads)
        try:
            for item in work:
                report(_task(*item))
        finally:
            torch.set_num_threads(before)
    else:
        with ProcessPoolExecutor(
            max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker, initargs=(threads,),
        ) as pool:
            futures = {pool.submit(_task, *item): item[0] for item in work}
            for future in as_completed(futures):
                try:
                    report(future.result())
                except Exception as exc:  # a worker died, e.g. killed for memory
                    report({"run": futures[future], "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"})
    summarize(results, mode, tol, console)
    return results


def summarize(results, mode, tol, console):
    by_status = {}
    for res in results:
        by_status.setdefault(res["status"], []).append(res)
    checked = [res for res in results if res.get("batch_check")]
    console.log("")
    console.log(f"===== ccm re-score ({mode}) =====")
    done = "ok" if mode == "fixed" else "checked"
    console.log(f"  {'written' if mode == 'fixed' else 'passed'}: "
                f"{len(by_status.get(done, []))}")
    if mode == "fixed":
        console.log(f"  skipped: {len(by_status.get('skipped', []))} "
                    f"(sidecar exists; --force rewrites)")
        console.log(f"  native:  {len(by_status.get('native', []))} "
                    f"(log holds sl_test_fixed/*; --force re-scores)")
    console.log(f"  failed self-check: {len(by_status.get('fail', []))}")
    console.log(f"  errors:  {len(by_status.get('error', []))}"
                + (" (any sidecar left as it was)" if mode == "fixed" else ""))
    devs = [(res["batch_check"]["max_rel_dev"], res["run"]) for res in checked
            if res["batch_check"]["max_rel_dev"] is not None]
    if devs:
        dev, run = max(devs)
        console.log(f"  max batch-protocol deviation: {dev:.2e} (tol {tol:.0e}) "
                    f"at {run}")
    val_devs = [(res["val_check"]["rel_dev"], res["run"]) for res in results
                if res.get("k") == 1 and res.get("val_check")
                and res["val_check"]["rel_dev"] is not None]
    if val_devs:
        dev, run = max(val_devs)
        console.log(f"  max k=1 validation-loss deviation: {dev:.2e} at {run}")
    for res in by_status.get("fail", []):
        for name in ("batch_check", "fixed_check"):
            check = res.get(name)
            if check and not check["passed"]:
                why = (f"{check['worst_key']} off by {check['max_rel_dev']:.1e}"
                       if not check["missing"] else f"not logged: {check['missing']}")
                break
        if res.get("stale"):
            why += f"; old sidecar moved to {res['stale']}"
        console.log(f"    - {res['run']}: {why}")
    for res in by_status.get("error", []):
        console.log(f"    - {res['run']}: {res['error']}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m forecasting.ccm_rescore",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("paths", nargs="+",
                        help="run dirs, trees to search, or quoted globs")
    parser.add_argument("--mode", choices=("batch", "fixed"), required=True,
                        help="batch: self-check only; fixed: also write the sidecar")
    parser.add_argument("--jobs", type=int, default=4,
                        help="runs re-scored at once, each in its own process")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch threads per run (default: cores / jobs, at most 4)")
    parser.add_argument("--tol", type=float, default=TOL,
                        help="max relative deviation of the gated sl_test/* keys")
    parser.add_argument("--out", default=None,
                        help="write sidecars under this dir instead of into the runs")
    parser.add_argument("--force", action="store_true",
                        help="rewrite existing sidecars, and re-score runs "
                             "that log the fixed protocol themselves")
    parser.add_argument("--report", default=None,
                        help="also write every run's result, self-checks "
                             "included, to this JSON file")
    opts = parser.parse_args(argv)
    if opts.mode == "batch" and (opts.out or opts.force):
        parser.error("--out and --force only apply to --mode fixed")
    if opts.mode == "fixed" and opts.out is None and provenance.git_dirty():
        # The sidecar names the commit that scored it; with uncommitted code
        # that commit is not what ran.
        parser.error("the code tree has uncommitted changes: commit them, "
                     "or write the sidecars elsewhere with --out")

    # CPU only: the card is shared, and every run is re-scored on the CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    console = Console(log_path=False)
    runs, not_cluster = [], 0
    for run in find_runs(opts.paths):
        try:
            enabled = read_log(run).get("parameters/cluster/enabled")
        except (OSError, json.JSONDecodeError):
            enabled = True  # unreadable: let the run report its own error
        if enabled:
            runs.append(run)
        else:
            not_cluster += 1
    console.log(f"{len(runs)} cluster runs"
                + (f" ({not_cluster} non-cluster runs ignored)" if not_cluster else ""))
    results = rescore(runs, opts.mode, opts.jobs, opts.threads, opts.out,
                      opts.tol, opts.force, console)
    if opts.report:
        with open(opts.report, "w") as f:
            json.dump(sorted(results, key=lambda res: res["run"]), f, indent=2)
    return int(any(res["status"] in ("fail", "error") for res in results))


if __name__ == "__main__":
    raise SystemExit(main())
