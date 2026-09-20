"""Provenance stamped into every run's log.json under parameters/exp/*.

Why this exists: on 2026-09-19 the three paradigm trees under logs/ turned out
to have been produced by three different commits (multivariate and per-channel
on 2026-08-01/03, cluster on 2026-08-25) and no run recorded which, so "same
network, different paradigm" could not be asserted from disk. A run now says
which code, which machine, and how many trainings shared the GPU while it ran.
See .personal/PLAN.md, contract C1 (commit) and C7 (concurrency).

Everything here is best-effort: a missing git binary, a tarball checkout or a
CPU-only box must never abort a training run, so failures degrade to "unknown".
"""

import os
import socket
import subprocess

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths whose uncommitted edits change what a training run computes. logs/,
# results/ and data/ are deliberately absent: they change all the time and say
# nothing about the code that produced a run.
CODE_PATHS = ("forecasting", "main.py", "pyproject.toml", "TimeSeriesCCM-main")


def _git(*argv):
    try:
        out = subprocess.run(["git", "-C", _REPO_ROOT, *argv], capture_output=True,
                             text=True, check=True, timeout=15)
        return out.stdout.strip()
    except Exception:
        return None


def git_commit():
    """Full SHA of the checked-out HEAD, or "unknown" outside a git checkout."""
    return _git("rev-parse", "HEAD") or "unknown"


def git_dirty():
    """True if any CODE_PATHS file has uncommitted changes.

    False when git is unavailable: git_commit() already reads "unknown" in that
    case, which is the stronger signal, and a bool keeps the logged field one
    type across runs.
    """
    out = _git("status", "--porcelain", "--", *CODE_PATHS)
    return bool(out.strip()) if out is not None else False


def gpu_name():
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run_provenance(args=None):
    """The fields Experiment.init_logger writes under parameters/exp/.

    `concurrency` is read from args.exp.concurrency, which the grid drivers set
    to their --jobs value; a single-job run leaves it at the config default 1.
    """
    exp = getattr(args, "exp", None)
    concurrency = getattr(exp, "concurrency", 1) if exp is not None else 1
    return {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "hostname": socket.gethostname(),
        "gpu_name": gpu_name(),
        "torch_version": torch.__version__,
        "concurrency": int(concurrency),
    }
