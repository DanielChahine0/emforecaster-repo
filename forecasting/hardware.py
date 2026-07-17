"""Single place that decides how to run on the current machine.

resolve_backend() picks cuda -> mps -> cpu by availability. build_profile()
turns a backend + model_id into concrete knobs (device, dataloader settings,
compile, batch size). A later task adds normalize_config() here, which will
write those knobs onto a loaded Global config so the same YAML runs on any
machine.
"""
import dataclasses
from typing import Optional

import torch

# Per-model batch sizes for Apple MPS, where 512 crosses the ~10-11 GiB
# memory-thrash cliff documented in MEMORY.md (40-130x slowdown). Anything not
# listed uses _MPS_DEFAULT_BATCH.
_MPS_BATCH = {
    "EMForecaster": 128,
    "RF_EMF_Transformer": 32,
}
_MPS_DEFAULT_BATCH = 128
_CPU_DEFAULT_BATCH = 64

# Attention-heavy models where torch.compile pays off on CUDA. The tiny
# datasets here mean other models compile slower than they run.
_COMPILE_MODELS = {"RF_EMF_Transformer", "PatchTST"}

# MPS compile eligibility is deliberately narrower than _COMPILE_MODELS above:
# per MEMORY.md, torch.compile is a net win for the MV transformer on MPS
# (~1.5x) but not worth it for the other models on this backend.
_MPS_COMPILE_MODELS = {"RF_EMF_Transformer"}


@dataclasses.dataclass
class HardwareProfile:
    backend: str            # "cuda" | "mps" | "cpu"
    device: str             # torch device string
    mps: bool               # value for args.exp.mps
    gpu_id: int             # value for args.exp.gpu_id
    num_workers: int
    pin_memory: bool
    compile: bool           # value for args.exp.compile (per model)
    batch_size: Optional[int]  # override for args.sl.batch_size, or None to keep YAML


def resolve_backend():
    """Pick the best available backend: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_profile(backend, model_id):
    """Build the concrete HardwareProfile for a given backend and model."""
    if backend == "cuda":
        return HardwareProfile(
            backend="cuda", device="cuda:0", mps=False, gpu_id=0,
            num_workers=4, pin_memory=True,
            compile=model_id in _COMPILE_MODELS,
            batch_size=None,  # keep the YAML value; 512 fits 32 GB VRAM
        )
    elif backend == "mps":
        return HardwareProfile(
            backend="mps", device="mps", mps=True, gpu_id=0,
            # Datasets are already in-RAM tensors, so MPS worker processes
            # add only IPC/RAM overhead with no load-time benefit.
            num_workers=0, pin_memory=False,
            compile=model_id in _MPS_COMPILE_MODELS,
            batch_size=_MPS_BATCH.get(model_id, _MPS_DEFAULT_BATCH),
        )
    elif backend == "cpu":
        return HardwareProfile(
            backend="cpu", device="cpu", mps=False, gpu_id=0,
            num_workers=2, pin_memory=False, compile=False,
            batch_size=_CPU_DEFAULT_BATCH,
        )
    else:
        raise ValueError(f"unknown backend: {backend!r}")
