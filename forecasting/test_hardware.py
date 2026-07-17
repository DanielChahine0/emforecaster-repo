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


def test_cpu_profile_is_conservative():
    p = hardware.build_profile("cpu", "RF_EMF_Transformer")
    assert p.device == "cpu" and p.mps is False
    assert p.pin_memory is False and p.compile is False
    assert p.batch_size == 64
