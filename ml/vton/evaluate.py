"""Score try-on outputs against real photos.

    # Paired: outputs named like the person images they reconstruct.
    python -m vton.evaluate --pred out/paired --gt data/test/image --mode paired

    # Unpaired: any outputs vs the real test photos (distribution distance).
    python -m vton.evaluate --pred out/unpaired --gt data/test/image --mode unpaired

Paired computes SSIM and LPIPS; unpaired computes FID and KID. FID is only meaningful with
thousands of images (VITON-HD test has 2,032); with fewer it's reported with a warning.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import torch
from PIL import Image

from vton import metrics
from vton.data import IMAGE_EXTS

MIN_FID_IMAGES = 2000


def list_images(directory: Path) -> List[Path]:
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load01(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    h, w = size
    img = Image.open(path).convert("RGB").resize((w, h), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def batches(paths: List[Path], size: Tuple[int, int], batch: int) -> Iterator[torch.Tensor]:
    for i in range(0, len(paths), batch):
        yield torch.stack([load01(p, size) for p in paths[i:i + batch]])


def paired(pred_dir: Path, gt_dir: Path, size, batch: int, device: str, use_lpips: bool) -> dict:
    gt = {p.stem: p for p in list_images(gt_dir)}
    pred = [p for p in list_images(pred_dir) if p.stem in gt]
    if not pred:
        raise SystemExit(f"No outputs in {pred_dir} match file names in {gt_dir}")
    lp = metrics.Lpips(device) if use_lpips else None
    ssim_vals, lpips_vals = [], []
    for i in range(0, len(pred), batch):
        chunk = pred[i:i + batch]
        a = torch.stack([load01(p, size) for p in chunk])
        b = torch.stack([load01(gt[p.stem], size) for p in chunk])
        ssim_vals += metrics.ssim(a, b).tolist()
        if lp is not None:
            lpips_vals += lp(a, b).tolist()
    out = {"mode": "paired", "images": len(pred), "unmatched_outputs": len(list_images(pred_dir)) - len(pred)}
    out.update(metrics.summarize(ssim_vals, "ssim"))
    if lpips_vals:
        out.update(metrics.summarize(lpips_vals, "lpips"))
    return out


def unpaired(pred_dir: Path, gt_dir: Path, size, batch: int, device: str) -> dict:
    pred, gt = list_images(pred_dir), list_images(gt_dir)
    if min(len(pred), len(gt)) < 2:
        raise SystemExit("Need at least 2 images in each folder for FID/KID")
    inc = metrics.InceptionFeatures(device)
    fp = np.concatenate([inc(x) for x in batches(pred, size, batch)])
    fg = np.concatenate([inc(x) for x in batches(gt, size, batch)])
    kid_mean, kid_std = metrics.kid_from_features(fp, fg)
    out = {
        "mode": "unpaired", "outputs": len(pred), "real": len(gt),
        "fid": round(metrics.fid_from_features(fp, fg), 3),
        "kid_x1000": round(kid_mean * 1000, 3), "kid_x1000_std": round(kid_std * 1000, 3),
    }
    if min(len(pred), len(gt)) < MIN_FID_IMAGES:
        out["warning"] = f"fewer than {MIN_FID_IMAGES} images: FID is biased upward; compare only like-for-like"
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True, help="folder of try-on outputs")
    ap.add_argument("--gt", type=Path, required=True, help="folder of real person photos")
    ap.add_argument("--mode", choices=["paired", "unpaired"], required=True)
    ap.add_argument("--size", type=int, nargs=2, default=[1024, 768], metavar=("H", "W"),
                    help="evaluation resolution (VITON-HD papers use 1024 768)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-lpips", action="store_true", help="skip LPIPS (no weight download)")
    ap.add_argument("--out", type=Path, help="also write the result JSON here")
    args = ap.parse_args(argv)

    size = tuple(args.size)
    if args.mode == "paired":
        result = paired(args.pred, args.gt, size, args.batch, args.device, not args.no_lpips)
    else:
        result = unpaired(args.pred, args.gt, size, args.batch, args.device)
    result["size"] = list(size)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
