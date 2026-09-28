"""Standard VITON evaluation metrics.

Paired setting (person wearing their own garment, compared to the real photo):
    SSIM (higher is better), LPIPS (lower is better)
Unpaired setting (person with a different garment, no ground truth):
    FID, KID (lower is better), computed between the set of outputs and the set of real photos.

SSIM, FID and KID math is implemented here and unit-tested. Feature extractors need
pretrained weights (downloaded on first use) and are imported lazily:
    LPIPS          - `lpips` package, AlexNet backbone (the VITON literature standard)
    FID/KID feats  - `pytorch_fid` InceptionV3 (the reference FID weights; torchvision's
                     Inception gives numbers that are not comparable with papers)
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg


# ------------------------------------------------------------------ SSIM

def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])


def ssim(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Mean SSIM per image. a, b: [N, C, H, W] in [0, data_range].

    Gaussian window 11x11, sigma 1.5, K1=0.01, K2=0.03, 'valid' filtering, averaged over
    channels and pixels (Wang et al. 2004; matches skimage with gaussian_weights=True,
    use_sample_covariance=False).
    """
    if a.shape != b.shape or a.dim() != 4:
        raise ValueError(f"expected equal [N,C,H,W] shapes, got {tuple(a.shape)} and {tuple(b.shape)}")
    a = a.double()
    b = b.double()
    c = a.shape[1]
    w = _gaussian_window().to(a.device)[None, None].repeat(c, 1, 1, 1)

    def filt(x):
        return F.conv2d(x, w, groups=c)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a, mu_b = filt(a), filt(b)
    var_a = filt(a * a) - mu_a ** 2
    var_b = filt(b * b) - mu_b ** 2
    cov = filt(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return s.flatten(1).mean(1).float()


# ------------------------------------------------------------------ FID / KID

def fid_from_features(f1: np.ndarray, f2: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to two feature sets [N, D]."""
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False)
    s2 = np.cov(f2, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        # Singular product (few samples): regularize as pytorch-fid does.
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    covmean = covmean.real
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def kid_from_features(f1: np.ndarray, f2: np.ndarray, subsets: int = 100,
                      subset_size: int = 1000, seed: int = 0) -> tuple:
    """Kernel Inception Distance: unbiased MMD^2 with kernel (x.y/d + 1)^3, averaged over
    random subsets (Binkowski et al. 2018). Returns (mean, std). Can be slightly negative."""
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(f1), len(f2))
    d = f1.shape[1]
    vals = []
    for _ in range(subsets):
        x = f1[rng.choice(len(f1), m, replace=False)]
        y = f2[rng.choice(len(f2), m, replace=False)]
        vals.append(_mmd2_unbiased(x, y, d))
    return float(np.mean(vals)), float(np.std(vals))


def _mmd2_unbiased(x: np.ndarray, y: np.ndarray, d: int) -> float:
    kxx = (x @ x.T / d + 1) ** 3
    kyy = (y @ y.T / d + 1) ** 3
    kxy = (x @ y.T / d + 1) ** 3
    m = len(x)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                 + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                 - 2 * kxy.mean())


# ------------------------------------------------------------------ pretrained extractors

class InceptionFeatures:
    """2048-d pool features with the reference FID InceptionV3 weights (pytorch_fid)."""

    def __init__(self, device: str = "cpu"):
        from pytorch_fid.inception import InceptionV3  # lazy: downloads weights on first use
        self.model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, images01: torch.Tensor) -> np.ndarray:
        """images01: [N, 3, H, W] in [0, 1] (resized to 299 internally)."""
        out = self.model(images01.to(self.device))[0]
        return out.squeeze(-1).squeeze(-1).cpu().numpy()


class Lpips:
    """LPIPS distance with the AlexNet backbone."""

    def __init__(self, device: str = "cpu"):
        import lpips  # lazy: downloads weights on first use
        self.model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, a01: torch.Tensor, b01: torch.Tensor) -> torch.Tensor:
        """a01, b01: [N, 3, H, W] in [0, 1]. Returns [N]."""
        return self.model(a01.to(self.device) * 2 - 1, b01.to(self.device) * 2 - 1).flatten().cpu()


def to01(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1], clamped."""
    return ((x + 1) / 2).clamp(0, 1)


def summarize(values: list, name: str, digits: int = 4) -> Optional[dict]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {name: round(float(arr.mean()), digits), f"{name}_std": round(float(arr.std()), digits)}
