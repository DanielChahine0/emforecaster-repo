import pytest

from forecasting import hardware


def test_resolve_backend_returns_valid_backend():
    assert hardware.resolve_backend() in {"cuda", "mps", "cpu"}


def test_cuda_profile_keeps_yaml_batch_and_enables_workers():
    p = hardware.build_profile("cuda", "EMForecaster")
    assert p.device == "cuda:0"
    assert p.mps is False and p.gpu_id == 0
    assert p.num_workers == 4 and p.pin_memory is True
    assert p.batch_size is None  # keep the YAML value on CUDA


def test_cuda_profile_compiles_attention_models_only():
    assert hardware.build_profile("cuda", "RF_EMF_Transformer").compile is True
    assert hardware.build_profile("cuda", "PatchTST").compile is True
    assert hardware.build_profile("cuda", "DLinear").compile is False


def test_mps_profile_uses_tuned_batch_sizes():
    assert hardware.build_profile("mps", "EMForecaster").batch_size == 128
    assert hardware.build_profile("mps", "RF_EMF_Transformer").batch_size == 32
    assert hardware.build_profile("mps", "DLinear").batch_size == 128  # default
    p = hardware.build_profile("mps", "RF_EMF_Transformer")
    assert p.device == "mps" and p.mps is True
    assert p.num_workers == 0 and p.pin_memory is False
    assert p.compile is True  # MEMORY.md: keep compile on for the MV transformer
    # PatchTST compiles on CUDA but must NOT compile on MPS; guards the
    # divergence between _COMPILE_MODELS and _MPS_COMPILE_MODELS.
    assert hardware.build_profile("mps", "PatchTST").compile is False


def test_build_profile_raises_on_unknown_backend():
    with pytest.raises(ValueError):
        hardware.build_profile("rocm", "EMForecaster")


def test_cpu_profile_is_conservative():
    p = hardware.build_profile("cpu", "RF_EMF_Transformer")
    assert p.device == "cpu" and p.mps is False
    assert p.pin_memory is False and p.compile is False
    assert p.batch_size == 64


import types

from forecasting import hardware as hw


def _fake_args():
    exp = types.SimpleNamespace(model_id="EMForecaster", mps=False, gpu_id=7, compile=True)
    data = types.SimpleNamespace(num_workers=99, pin_memory=True)
    sl = types.SimpleNamespace(batch_size=512)
    return types.SimpleNamespace(exp=exp, data=data, sl=sl)


def test_normalize_config_applies_mps_profile():
    args = _fake_args()
    profile = hw.normalize_config(args, backend="mps")
    assert profile.backend == "mps"
    assert args.exp.mps is True and args.exp.gpu_id == 0
    assert args.exp.compile is False  # EMForecaster on MPS
    assert args.data.num_workers == 0 and args.data.pin_memory is False
    assert args.sl.batch_size == 128


def test_normalize_config_keeps_yaml_batch_on_cuda():
    args = _fake_args()
    args.sl.batch_size = 512
    hw.normalize_config(args, backend="cuda")
    assert args.sl.batch_size == 512  # batch_size is None on cuda -> unchanged
    assert args.data.num_workers == 4 and args.data.pin_memory is True


def test_cuda_non_blocking_only_true_for_cuda():
    import torch
    assert hardware.cuda_non_blocking(torch.device("cpu")) is False
    assert hardware.cuda_non_blocking(torch.device("mps")) is False
    assert hardware.cuda_non_blocking(types.SimpleNamespace(type="cuda")) is True
    assert hardware.cuda_non_blocking(None) is False


def test_to_device_moves_tensor_and_preserves_values():
    import torch
    t = torch.tensor([1.0, 2.0])
    out = hardware.to_device(t, torch.device("cpu"))
    assert out.device.type == "cpu"
    assert torch.equal(out, t)


def test_normalize_config_on_real_pydantic_config():
    # Guard the real path: normalize_config must mutate a genuine pydantic Global
    # (validate_assignment/coercion), not just the SimpleNamespace fakes above.
    from forecasting import grid
    load_config, _ = grid.setup_modules("multivariate")
    args = load_config(grid.args_path("multivariate", "rf_emf_mv/dlinear"))
    profile = hardware.normalize_config(args)  # detects this machine's backend
    assert args.exp.mps == profile.mps
    assert args.exp.gpu_id == profile.gpu_id
    assert args.exp.compile == profile.compile
    assert args.data.num_workers == profile.num_workers
    assert args.data.pin_memory == profile.pin_memory
    if profile.batch_size is not None:
        assert args.sl.batch_size == profile.batch_size
