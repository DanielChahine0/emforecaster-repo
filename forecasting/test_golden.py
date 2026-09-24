"""Golden hashes: the existing models' numerics, pinned bit for bit.

The tripwire for changes that must not move an existing model. Extending CCM to
more models touches get_model, the model constructors, the cluster wrapper and
the training step, and every run already under logs/ has to stay reproducible
by the code that replaces them: a byte of drift in an init, a forward or a
gradient and the old and new runs are no longer the same network.

Each case builds one network through the grid's own path (the rf_emf_mv/* job
config, the grid's window and overrides -- or perchannel.channel_ablation for a
band -- then get_model) from a fixed seed and records sha256 digests of:

  keys     the state_dict key names, so a module or buffer added, removed or
           renamed fails as that rather than as changed weights.
  state    the state_dict, names and bytes: the init.
  rng      the CPU RNG state right after the build: what the constructor drew,
           which shifts every later draw (dropout masks, the CCM assigner).
  forward  an eval-mode forward on a fixed input; for CCM through
           CCMForecaster, so the assigner's membership is in it.
  train_*  one optimizer step taken the way Experiment.train takes it: the
           train-mode forward (dropout masks, the CCM cluster_emb refresh), the
           loss with CCM's beta-weighted similarity term, the gradients, the
           state after the step and the RNG state after it.

The cases are the six gradient-trained multivariate models and the CCM builds
of the same six at the cluster grids' k values, at the EMF benchmark shape (27
bands, L336/O96) and the VoIP one (6 channels, L48/O12, VOIP_OVERRIDES); the
six again as one band (C=1) at both windows, the way a per-channel run builds
them; and the six at EMF under the grids' second seed.

A failure means a change altered an existing model's numerics: fix the change,
not the fixture. The fixture is only regenerated for a deliberate numerics
change, a new torch or a change to the cases here, with the training code
committed:

    venv/bin/python -m forecasting.test_golden --regenerate

It records the commit whose code it pins and the git blob id of this module as
it ran (`git log --all --find-object=<id>` finds the commit that carries it),
so this module and its fixture can change together in one commit.

The digests are only bit-exact for the torch build and CPU kernel set that
recorded them (AVX2 and AVX512 kernels round differently), so the hash checks
skip when either differs from the fixture's. A skip is not a pass: `pytest -rs`
prints why. MKL and oneDNN also pick kernels by CPU model, which the fixture
records and a failure on another model names.
"""

import argparse
import collections
import hashlib
import json
import os
import platform
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from forecasting import grid, grids, perchannel, provenance
from forecasting.multivariate.utils.models import (compute_aux_loss, compute_loss,
                                                   forward_pass, get_criterion,
                                                   get_model, get_optim,
                                                   model_update)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(_REPO_ROOT, "forecasting", "test_golden.json")
HARNESS = os.path.abspath(__file__)

SEED = grids.SEED
# What `--seeds 2` adds. EMForecaster reseeds from args.exp.seed inside its
# constructor, so one seed alone would not show the seed reaching it.
SECOND_SEED = grids.select_seeds(2)[1]
INPUT_SEED = 0
# More than one window, so the CCM assigner's batch-mean membership is pinned
# as the pipeline computes it.
BATCH = 4
# CPU kernels split reductions by thread count (12 of the 30 original forwards
# change at 24 threads), so one thread keeps the bytes independent of the
# machine.
NUM_THREADS = 1

Shape = collections.namedtuple(
    "Shape", "name num_channels seq_len pred_len dataset overrides per_channel k_values")

# k: what the cluster grids sweep, degenerate ends included (k=1, and k=C=6 on
# VoIP). VoIP also keeps k=9, more clusters than channels, which builds and was
# pinned from the first version of the fixture; every CCM model is pinned there
# too, so all six are held to one set of k.
SHAPES = [
    Shape("emf", 27, 336, 96, "ED2022", None, False, [1, 2, 3, 5, 9]),
    Shape("voip", 6, 48, 12, "VOIP_MOBILE", grids.VOIP_OVERRIDES, False,
          [1, 2, 3, 6, 9]),
    Shape("emf-pc", 1, 336, 96, "ED2022", None, True, []),
    Shape("voip-pc", 1, 48, 12, "VOIP_MOBILE", grids.VOIP_OVERRIDES, True, []),
]
# The band only selects data; the build sees num_channels=1 whichever it is.
PC_CHANNEL = 0
MV_IDS = ["DLinear", "PatchTST", "TSMixer", "RF_EMF_Transformer", "EMForecaster",
          "LSTM"]
# The cluster grids' six models (grids.CLUSTER_MODELS). The Transformer,
# EMForecaster and LSTM joined after the first three were pinned, so they come
# after them here and in the fixture.
CCM_IDS = ["DLinear", "PatchTST", "TSMixer", "RF_EMF_Transformer", "EMForecaster",
           "LSTM"]

_JOBS = {model_id: job for _, model_id, job in grids.MV_MODELS}

# Checked in this order, so the first failure is the most structural one.
CHECKS = [
    ("keys", "state_dict keys added, removed or renamed"),
    ("num_params", "parameter count changed"),
    ("out_shape", "output shape changed"),
    ("state", "initial weights changed"),
    ("rng", "the build draws from the RNG differently"),
    ("forward", "eval forward changed"),
    ("train_forward", "train-mode forward changed (dropout masks or the CCM refresh)"),
    ("train_loss", "training loss changed (forecast MSE or CCM similarity term)"),
    ("grads", "gradients changed"),
    ("train_state", "state after one optimizer step changed"),
    ("train_rng", "the training step draws from the RNG differently"),
]


def _cases():
    cases = []
    for shape in SHAPES:
        cases += [(shape, model_id, None, SEED) for model_id in MV_IDS]
        cases += [(shape, model_id, k, SEED)
                  for model_id in CCM_IDS for k in shape.k_values]
    cases += [(SHAPES[0], model_id, None, SECOND_SEED) for model_id in MV_IDS]
    return cases


def case_id(shape, model_id, k, seed):
    return (f"{shape.name}-{model_id}" + (f"-k{k}" if k is not None else "")
            + (f"-s{seed}" if seed != SEED else ""))


CASES = {case_id(*case): case for case in _cases()}


def build_args(shape, model_id, k, seed):
    """The config run_grid, or perchannel.py for a band, would hand Experiment.

    hardware.normalize_config is left out: it only sets batch, worker and
    compile fields, none of which reach a constructor, and applying it would
    tie the fixture to the host. num_channels is set by hand for the joint
    shapes because get_loaders infers it from the data, which the build does
    not load.
    """
    load_config, _ = grid.setup_modules("multivariate")
    args = load_config(grid.args_path("multivariate", _JOBS[model_id]))
    if shape.per_channel:
        # perchannel.py passes this to main.py as --ablation, which sets each
        # global.<group>.<field> in turn (main.update_global_config).
        ablation = perchannel.channel_ablation(
            shape.dataset, shape.seq_len, shape.pred_len, PC_CHANNEL,
            grids.GRIDS["univariate"]["alpha"], seed=seed, overrides=shape.overrides)
        for key, value in ablation.items():
            _, group, field = key.split(".")
            setattr(getattr(args, group), field, value)
        return args
    args.data.dataset = shape.dataset
    args.data.seq_len = shape.seq_len
    args.data.pred_len = shape.pred_len
    args.data.num_channels = shape.num_channels
    args.exp.seed_list = [seed]
    args.exp.seed = seed
    if k is not None:
        args.cluster.enabled = True
        args.cluster.k_clusters = k
    grid.apply_overrides(args, shape.overrides)
    return args


def _update(h, name, tensor):
    t = tensor.detach().cpu().contiguous()
    h.update(f"{name}|{t.dtype}|{tuple(t.shape)}\n".encode())
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())


def key_hash(model):
    return hashlib.sha256("\n".join(model.state_dict()).encode()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        _update(h, name, tensor)
    return h.hexdigest()


def grad_hash(model):
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        if param.grad is None:
            h.update(f"{name}|none\n".encode())
        else:
            _update(h, name, param.grad)
    return h.hexdigest()


def tensor_hash(tensor):
    h = hashlib.sha256()
    _update(h, "", tensor)
    return h.hexdigest()


def rng_hash():
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def train_step(args, model, batch, seed):
    """One optimizer step as Experiment.train takes it, digested.

    The global RNG is reseeded first: in a run the step starts wherever data
    loading left it, which a build cannot reproduce, and a fixed start keeps
    these digests independent of the eval forward before it.
    """
    model_id = args.exp.model_id
    optimizer = get_optim(args, model, args.sl.optimizer)
    criterion = get_criterion(args, args.sl.criterion)
    accum_steps = max(1, args.sl.accum_steps)

    torch.manual_seed(seed)
    model.train()
    optimizer.zero_grad()
    output = forward_pass(args, model, batch, model_id, "cpu")
    loss = compute_loss(output, batch, criterion, model_id, args, "cpu")
    aux_loss = compute_aux_loss(model, batch, args, "cpu")
    step_loss = loss if aux_loss is None else loss + aux_loss
    model_update(model, step_loss / accum_steps, optimizer, model_id)

    losses = [loss] if aux_loss is None else [loss, aux_loss]
    return {
        "train_forward": tensor_hash(output),
        "train_loss": tensor_hash(torch.stack(losses)),
        "grads": grad_hash(model),
        "train_state": state_hash(model),
        "train_rng": rng_hash(),
    }


def compute(case):
    """The recorded digests for one case.

    Global RNG state and the thread count are restored afterwards, so the
    tests around this one see the process as they left it.
    """
    shape, model_id, k, seed = case
    args = build_args(*case)

    saved = torch.get_rng_state(), np.random.get_state(), random.getstate()
    threads = torch.get_num_threads()
    torch.set_num_threads(NUM_THREADS)
    try:
        # Seeded the way Experiment.run seeds before it builds the model.
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = get_model(args, torch.Generator().manual_seed(seed))
        got = {
            "rng": rng_hash(),
            "keys": key_hash(model),
            "state": state_hash(model),
            "num_params": sum(p.numel() for p in model.parameters()),
        }

        inputs = torch.Generator().manual_seed(INPUT_SEED)
        x = torch.randn(BATCH, shape.num_channels, shape.seq_len, generator=inputs)
        y = torch.randn(BATCH, shape.num_channels, shape.pred_len, generator=inputs)
        model.eval()
        with torch.no_grad():
            out = model(x)
        got["forward"] = tensor_hash(out)
        got["out_shape"] = list(out.shape)

        got.update(train_step(args, model, (x, y), seed))
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(saved[0])
        np.random.set_state(saved[1])
        random.setstate(saved[2])

    return got


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _load_fixture():
    with open(FIXTURE) as f:
        return json.load(f)


GOLDEN = _load_fixture() if os.path.exists(FIXTURE) else None


def _skip_reason():
    if GOLDEN is None:
        return None
    if torch.__version__ != GOLDEN["torch"]:
        return (f"golden hashes were recorded under torch {GOLDEN['torch']} on "
                f"{GOLDEN['cpu_model']}; this is torch {torch.__version__}, whose "
                f"kernels need not be bit-identical")
    capability = torch.backends.cpu.get_cpu_capability()
    if capability != GOLDEN["cpu_capability"]:
        return (f"golden hashes were recorded with {GOLDEN['cpu_capability']} CPU "
                f"kernels on {GOLDEN['cpu_model']}; this machine runs {capability}, "
                f"which rounds differently")
    return None


def _machine_note():
    if GOLDEN is None or cpu_model() == GOLDEN["cpu_model"]:
        return ""
    return (f" [recorded on {GOLDEN['cpu_model']}, running on {cpu_model()}: MKL "
            f"and oneDNN pick kernels by CPU model, so this may be the machine "
            f"rather than the code]")


_SKIP = _skip_reason()


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def test_fixture_exists():
    assert GOLDEN is not None, f"{FIXTURE} is missing; see the module docstring"


def test_fixture_covers_exactly_the_cases():
    # A case or check added here without a regeneration would otherwise never
    # be checked, and one dropped would leave a stale entry that reads as
    # coverage.
    assert GOLDEN is not None
    assert list(GOLDEN["cases"]) == list(CASES)
    fields = {field for field, _ in CHECKS}
    assert all(set(digests) == fields for digests in GOLDEN["cases"].values())


def test_cases_pin_every_cluster_grid_model_at_every_grid_k():
    # A model or k added to a cluster grid without a case here would train
    # under code nothing pins.
    assert set(CCM_IDS) == {model_id for _, model_id, _ in grids.CLUSTER_MODELS}
    for setup, shape in (("cluster", SHAPES[0]), ("voip_cluster", SHAPES[1])):
        assert set(grids.GRIDS[setup]["k_values"]) <= set(shape.k_values), setup


def test_fixture_records_how_it_was_computed():
    # Changing any of these changes every digest; they are recorded so a
    # mismatch can be told apart from a settings change.
    assert GOLDEN is not None
    assert (GOLDEN["seed"], GOLDEN["input_seed"], GOLDEN["batch"],
            GOLDEN["num_threads"]) == (SEED, INPUT_SEED, BATCH, NUM_THREADS)


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
@pytest.mark.parametrize("name", list(CASES))
def test_golden_hashes(name):
    assert GOLDEN is not None and name in GOLDEN["cases"]
    want = GOLDEN["cases"][name]
    got = compute(CASES[name])
    for field, problem in CHECKS:
        assert got[field] == want[field], f"{name}: {problem}{_machine_note()}"


# ---------------------------------------------------------------------------
# Regeneration
# ---------------------------------------------------------------------------


def _git(*argv):
    # GIT_OPTIONAL_LOCKS=0 keeps status read-only, as in provenance._git, whose
    # .strip() would eat the porcelain status column this parses.
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    return subprocess.run(["git", "-C", _REPO_ROOT, *argv], capture_output=True,
                          text=True, check=True, env=env).stdout


def _dirty_code_paths():
    """Changed or untracked files under the code paths that could move a digest.

    This module and its fixture are exempt, since the fixture records the
    module's blob id, and so is markdown, which nothing imports. Anything else,
    other tests included, has to be committed for the recorded commit to mean
    what it says.
    """
    exempt = {os.path.relpath(path, _REPO_ROOT) for path in (FIXTURE, HARNESS)}
    out = _git("status", "--porcelain", "--untracked-files=all", "--",
               *provenance.CODE_PATHS)
    dirty = []
    for line in out.splitlines():
        path = line[3:].split(" -> ")[-1]
        if path not in exempt and not path.endswith(".md"):
            dirty.append(path)
    return dirty


def regenerate():
    commit = provenance.git_commit()
    if commit == "unknown":
        sys.exit("not a git checkout; the fixture has to name the commit it pins")
    dirty = _dirty_code_paths()
    if dirty:
        sys.exit("refusing to regenerate on a dirty tree; the fixture must pin "
                 "committed code:\n  " + "\n  ".join(dirty))
    golden = {
        "torch": torch.__version__,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "cpu_model": cpu_model(),
        "commit": commit,
        "harness": _git("hash-object", HARNESS).strip(),
        "seed": SEED,
        "input_seed": INPUT_SEED,
        "batch": BATCH,
        "num_threads": NUM_THREADS,
        "cases": {name: compute(case) for name, case in CASES.items()},
    }
    changed = [name for name, digests in golden["cases"].items()
               if GOLDEN is None or GOLDEN["cases"].get(name) != digests]
    with open(FIXTURE, "w") as f:
        json.dump(golden, f, indent=2)
        f.write("\n")
    print(f"wrote {len(golden['cases'])} cases to {FIXTURE} "
          f"({len(changed)} new or changed) at {commit[:12]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--regenerate", action="store_true",
                        help="rewrite the fixture from the current code")
    if not parser.parse_args().regenerate:
        parser.error("nothing to do: pytest runs the check; pass --regenerate "
                     "to rewrite the fixture")
    regenerate()
