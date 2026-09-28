"""DM-VTON (Distilled Mobile Real-time Virtual Try-On) on CPU and for on-device export.

DM-VTON is a parser-free student model: person + garment + garment mask -> try-on, 256x192.
Code and weights are CC BY-NC-SA 4.0 (non-commercial, share-alike); they are not vendored
into this repo. Fetch them into ml/.cache:

    git clone --depth 1 https://github.com/KiseKloset/DM-VTON ml/.cache/DM-VTON
    python -m gdown --folder https://drive.google.com/drive/folders/1wfWGsR0vWC5LrA26xhj92ec_GoCKV80A -O ml/.cache/dmvton_ckpt

Two ops are swapped for mobile-friendly equivalents (vton.mobile_ops): the CUDA-only
correlation layer and F.grid_sample (-> 1-D gather).

    python -m vton.dmvton check              # patched model == reference model
    python -m vton.dmvton infer --viton ml/.cache/VITON-Clean/VITON_test --out runs/dmvton
"""

import argparse
import contextlib
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from vton import mobile_ops

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "ml" / ".cache" / "DM-VTON"
CKPT_DIR = ROOT / "ml" / ".cache" / "dmvton_ckpt"
HEIGHT, WIDTH = 256, 192

_torch_grid_sample = F.grid_sample


def _import_dmvton():
    if not CODE_DIR.is_dir():
        raise FileNotFoundError(f"DM-VTON code not found at {CODE_DIR} (see module docstring)")
    if "cupy" not in sys.modules:
        # correlation.py imports cupy and decorates with cupy.memoize at import time; the
        # CUDA kernels themselves are never called because FunctionCorrelation is replaced.
        stub = types.ModuleType("cupy")
        stub.memoize = lambda **_: (lambda fn: fn)
        sys.modules["cupy"] = stub
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    import models.warp_modules.mobile_afwm as mobile_afwm
    from pipelines import DMVTONPipeline
    mobile_afwm.FunctionCorrelation = (
        lambda tenFirst, tenSecond, intStride=1: mobile_ops.correlation(tenFirst, tenSecond, intStride))
    return DMVTONPipeline


@contextlib.contextmanager
def gather_grid_sample(enabled: bool = True):
    """Route every F.grid_sample call through the gather implementation."""
    if not enabled:
        yield
        return
    F.grid_sample = mobile_ops.grid_sample_gather
    torch.nn.functional.grid_sample = mobile_ops.grid_sample_gather
    try:
        yield
    finally:
        F.grid_sample = _torch_grid_sample
        torch.nn.functional.grid_sample = _torch_grid_sample


def _safe_load(path: Path) -> dict:
    """torch.load in weights-only mode (no arbitrary code execution), allow-listing only the
    NumPy scalar types these checkpoints store alongside the weights."""
    allowed = [
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np._core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
        type(np.dtype(np.float32)),
        type(np.dtype(np.int64)),
    ]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_pipeline(ckpt_dir: Path = CKPT_DIR) -> nn.Module:
    pipeline_cls = _import_dmvton()
    from utils.torch_utils import load_ckpt
    pipe = pipeline_cls(align_corners=True, checkpoints=None)
    load_ckpt(pipe.warp_model, _safe_load(Path(ckpt_dir) / "dmvton_pf_warp.pt"))
    load_ckpt(pipe.gen_model, _safe_load(Path(ckpt_dir) / "dmvton_pf_gen.pt"))
    return pipe.eval()


class TryOnNHWC(nn.Module):
    """The app's model contract (assets/models/README.md), with the garment mask input:
    person [1,H,W,3], garment [1,H,W,3] in [-1,1]; garment_mask [1,H,W,1] in {0,1}
    -> try-on [1,H,W,3] in [-1,1]."""

    def __init__(self, pipeline: nn.Module):
        super().__init__()
        self.pipeline = pipeline

    def forward(self, person, garment, garment_mask):
        p = person.permute(0, 3, 1, 2)
        g = garment.permute(0, 3, 1, 2)
        m = garment_mask.permute(0, 3, 1, 2)
        tryon, _ = self.pipeline(p, g, m, phase="test")
        return tryon.permute(0, 2, 3, 1)


# ------------------------------------------------------------------ I/O helpers

def load_rgb(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((WIDTH, HEIGHT), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img, np.float32) / 127.5 - 1)[None]  # [1,H,W,3]


def load_mask(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((WIDTH, HEIGHT), Image.NEAREST)
    return torch.from_numpy((np.asarray(img) > 127).astype(np.float32))[None, ..., None]


def save_rgb(t: torch.Tensor, path: Path) -> None:
    arr = ((t[0].clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
    Image.fromarray(arr).save(path)


# ------------------------------------------------------------------ commands

def check(seed: int = 0) -> float:
    """Max abs difference between the gather-patched model and torch grid_sample."""
    model = TryOnNHWC(load_pipeline())
    g = torch.Generator().manual_seed(seed)
    person = torch.rand(1, HEIGHT, WIDTH, 3, generator=g) * 2 - 1
    garment = torch.rand(1, HEIGHT, WIDTH, 3, generator=g) * 2 - 1
    mask = (torch.rand(1, HEIGHT, WIDTH, 1, generator=g) > 0.5).float()
    with torch.no_grad():
        ref = model(person, garment, mask)
        with gather_grid_sample():
            ours = model(person, garment, mask)
    return float((ours - ref).abs().max())


def infer(viton_test: Path, out_dir: Path, limit: int, use_gather: bool) -> dict:
    """Run on VITON(-Clean) test pairs (test_img / test_color / test_edge + test_pairs.txt)."""
    model = TryOnNHWC(load_pipeline())
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [line.split() for line in (viton_test / "test_pairs.txt").read_text().splitlines() if line.strip()]
    times = []
    with torch.no_grad(), gather_grid_sample(use_gather):
        for person_name, cloth_name in pairs[:limit]:
            person = load_rgb(viton_test / "test_img" / person_name)
            garment = load_rgb(viton_test / "test_color" / cloth_name)
            mask = load_mask(viton_test / "test_edge" / cloth_name)
            t0 = time.perf_counter()
            tryon = model(person, garment, mask)
            times.append((time.perf_counter() - t0) * 1000)
            save_rgb(tryon, out_dir / person_name)
    return {"images": len(times), "cpu_ms_median": round(float(np.median(times)), 1), "out": str(out_dir)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("infer")
    p.add_argument("--viton", type=Path, required=True, help="VITON_test folder")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--torch-grid-sample", action="store_true", help="use F.grid_sample (reference)")
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    if args.cmd == "check":
        print(f"max abs diff, gather vs torch grid_sample: {check():.2e}")
    else:
        print(infer(args.viton, args.out, args.limit, not args.torch_grid_sample))


if __name__ == "__main__":
    main()
