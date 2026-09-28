import itertools

import pytest
import torch
import torch.nn.functional as F

from vton.mobile_ops import correlation, grid_sample_gather


@pytest.mark.parametrize("align_corners,padding_mode",
                         list(itertools.product([True, False], ["zeros", "border"])))
def test_grid_sample_gather_matches_torch(align_corners, padding_mode):
    torch.manual_seed(0)
    inp = torch.randn(2, 5, 17, 13)
    # Grid covers out-of-range samples and a different output size than the input.
    grid = torch.rand(2, 11, 9, 2) * 2.6 - 1.3
    ref = F.grid_sample(inp, grid, mode="bilinear", padding_mode=padding_mode, align_corners=align_corners)
    ours = grid_sample_gather(inp, grid, padding_mode=padding_mode, align_corners=align_corners)
    assert ours.shape == ref.shape
    assert torch.allclose(ours, ref, atol=1e-5)


def test_grid_sample_gather_identity_grid():
    inp = torch.randn(1, 3, 8, 6)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 6), indexing="ij")
    grid = torch.stack([xs, ys], -1)[None]
    out = grid_sample_gather(inp, grid, padding_mode="border", align_corners=True)
    assert torch.allclose(out, inp, atol=1e-6)


def test_correlation_matches_naive_loop():
    torch.manual_seed(1)
    a = torch.randn(1, 4, 6, 5)
    b = torch.randn(1, 4, 6, 5)
    out = correlation(a, b)
    assert out.shape == (1, 49, 6, 5)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            ch = (dy + 3) * 7 + (dx + 3)
            for y in range(6):
                for x in range(5):
                    y2, x2 = y + dy, x + dx
                    expected = (a[0, :, y, x] * b[0, :, y2, x2]).mean() if 0 <= y2 < 6 and 0 <= x2 < 5 else 0.0
                    assert out[0, ch, y, x].item() == pytest.approx(float(expected), abs=1e-6)
