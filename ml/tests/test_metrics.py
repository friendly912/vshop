import numpy as np
import pytest
import torch
from scipy.signal import convolve2d

from vton import metrics


def _reference_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Independent NumPy/SciPy implementation for one grayscale image in [0, 1]."""
    win = metrics._gaussian_window().numpy()

    def f(x):
        return convolve2d(x, win[::-1, ::-1], mode="valid")

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = f(a), f(b)
    va, vb = f(a * a) - ma ** 2, f(b * b) - mb ** 2
    cov = f(a * b) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def test_ssim_identical_is_one():
    x = torch.rand(2, 3, 32, 32)
    assert torch.allclose(metrics.ssim(x, x), torch.ones(2), atol=1e-6)


def test_ssim_matches_reference_and_is_symmetric():
    rng = np.random.default_rng(0)
    a = rng.random((40, 30))
    b = np.clip(a + rng.normal(0, 0.1, a.shape), 0, 1)
    ta = torch.tensor(a)[None, None]
    tb = torch.tensor(b)[None, None]
    ours = metrics.ssim(ta, tb).item()
    assert ours == pytest.approx(_reference_ssim(a, b), abs=1e-5)
    assert metrics.ssim(tb, ta).item() == pytest.approx(ours, abs=1e-7)


def test_ssim_decreases_with_noise():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 48, 48)
    vals = [metrics.ssim(x, (x + s * torch.randn_like(x)).clamp(0, 1)).item() for s in (0.02, 0.1, 0.3)]
    assert vals[0] > vals[1] > vals[2]


def test_ssim_rejects_bad_shapes():
    with pytest.raises(ValueError):
        metrics.ssim(torch.rand(3, 8, 8), torch.rand(3, 8, 8))


def test_fid_zero_for_identical_sets():
    f = np.random.default_rng(0).normal(size=(500, 8))
    assert metrics.fid_from_features(f, f) == pytest.approx(0.0, abs=1e-6)


def test_fid_mean_shift_is_exact():
    f = np.random.default_rng(1).normal(size=(300, 6))
    # Same covariance, means differ by c in every dim -> FID = d * c^2.
    assert metrics.fid_from_features(f, f + 0.5) == pytest.approx(6 * 0.25, rel=1e-6)


def test_fid_scaled_set_is_exact():
    f = np.random.default_rng(2).normal(size=(400, 5)) + 1.0
    s1 = np.cov(f, rowvar=False)
    mu = f.mean(0)
    # f2 = 2 f: S2 = 4 S1, sqrt(S1 S2) = 2 S1 -> FID = |mu|^2 + tr(S1).
    expected = mu @ mu + np.trace(s1)
    assert metrics.fid_from_features(f, 2 * f) == pytest.approx(expected, rel=1e-6)


def test_mmd_matches_brute_force():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(5, 4)), rng.normal(1, 1, size=(5, 4))
    d = 4
    k = lambda a, b: (a @ b / d + 1) ** 3
    m = 5
    xx = sum(k(x[i], x[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    yy = sum(k(y[i], y[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    xy = sum(k(x[i], y[j]) for i in range(m) for j in range(m)) / (m * m)
    assert metrics._mmd2_unbiased(x, y, d) == pytest.approx(xx + yy - 2 * xy, rel=1e-9)


def test_kid_near_zero_for_same_distribution_and_positive_otherwise():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=(400, 16)), rng.normal(size=(400, 16))
    c = rng.normal(0.5, 1, size=(400, 16))
    same, _ = metrics.kid_from_features(a, b, subsets=20, subset_size=200)
    diff, _ = metrics.kid_from_features(a, c, subsets=20, subset_size=200)
    assert abs(same) < 0.01
    assert diff > 10 * max(abs(same), 1e-3)
