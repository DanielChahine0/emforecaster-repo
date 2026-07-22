import numpy as np
import pytest
import torch

from forecasting.multivariate.utils import dataloading as mv_dataloading
from forecasting.univariate.utils import dataloading as uv_dataloading


# Both pipelines carry their own copy of create_windows; a fix in one that
# misses the other silently corrupts that pipeline's runs, so every test runs
# against both.
@pytest.fixture(
    params=[mv_dataloading.create_windows, uv_dataloading.create_windows],
    ids=["multivariate", "univariate"],
)
def create_windows(request):
    return request.param


def test_noncontiguous_band_view_windows_the_band(create_windows):
    # The per-band path windows train_data.T[i]: a view of one channel whose
    # storage stride is num_channels. The as_strided implementation read raw
    # storage here, so every band trained on the same channel-interleaved slice
    # of the flattened array (the bug behind the 2026-07 per-channel grid).
    data = np.arange(200.0).reshape(50, 4)  # (time, channels), C-contiguous
    band = torch.from_numpy(data.T[2])
    assert not band.is_contiguous()

    windows = create_windows(band, 8, 1, None)

    series = torch.from_numpy(data[:, 2].copy())
    assert windows.shape == (43, 8)  # (50 - 8) // 1 + 1, unchanged by the fix
    for j in range(43):
        assert torch.equal(windows[j], series[j : j + 8])


def test_band_view_and_band_copy_agree(create_windows):
    # Windowing must depend only on the tensor's logical values, never on how
    # they happen to be laid out in memory.
    data = np.arange(120.0).reshape(30, 4)
    view = torch.from_numpy(data.T[1])
    copy = torch.from_numpy(data[:, 1].copy())
    assert torch.equal(create_windows(view, 7, 3, None),
                       create_windows(copy, 7, 3, None))


def test_1d_contiguous_windows(create_windows):
    series = torch.arange(10.0)
    windows = create_windows(series, 4, 2, None)
    assert windows.shape == (4, 4)
    for j in range(4):
        assert torch.equal(windows[j], series[2 * j : 2 * j + 4])


def test_1d_nonoverlapping_stride(create_windows):
    # window_stride == -1 means back-to-back windows.
    series = torch.arange(10.0)
    windows = create_windows(series, 5, -1, None)
    assert windows.shape == (2, 5)
    assert torch.equal(windows[0], series[:5])
    assert torch.equal(windows[1], series[5:])


def test_2d_windows_shape_and_content(create_windows):
    # The datetime path windows (time, features) arrays; the contract is
    # (num_windows, window_size, num_channels).
    data = torch.arange(60.0).reshape(20, 3)
    windows = create_windows(data, 5, 2, None)
    assert windows.shape == (8, 5, 3)
    for j in range(8):
        assert torch.equal(windows[j], data[2 * j : 2 * j + 5])


def test_2d_noncontiguous_input(create_windows):
    # A column-sliced view has a row stride wider than its channel count; the
    # as_strided implementation assumed row stride == num_channels and produced
    # windows shifted into the dropped column.
    full = torch.arange(80.0).reshape(20, 4)
    data = full[:, 1:]
    assert not data.is_contiguous()

    windows = create_windows(data, 5, 2, None)

    assert windows.shape == (8, 5, 3)
    for j in range(8):
        assert torch.equal(windows[j], data[2 * j : 2 * j + 5])


def test_full_sequence_window(create_windows):
    # window_size == -1 returns the whole sequence as a single window.
    series = torch.arange(6.0)
    windows = create_windows(series, -1, 1, None)
    assert torch.equal(windows, series.unsqueeze(0))
