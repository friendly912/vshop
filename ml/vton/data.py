"""Try-on dataset in the VITON-HD folder layout.

This layout is the project's common format. VITON-HD uses it natively; DressCode and
self-collected data should be converted to it (see docs/ROADMAP.md, phase 0 licensing).

    <root>/
      train_pairs.txt, test_pairs.txt   one "person.jpg cloth.jpg" per line
      <split>/image/        person photos                     (required)
      <split>/cloth/        garment product photos            (required)
      <split>/cloth-mask/   garment masks, png or jpg         (optional)
      <split>/parse-mp/     6-class MediaPipe parsing, png    (optional, from vton.preprocess)
      <split>/pose-mp/      33 MediaPipe landmarks, json      (optional, from vton.preprocess)

Paired mode (default) uses each person's own garment (cloth file with the same name), which
is how SSIM/LPIPS are evaluated. Unpaired mode uses the pairs file (FID/KID setting).

Check a dataset:  python -m vton.data --root <root> --split test
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

OPTIONAL_DIRS = ("cloth-mask", "parse-mp", "pose-mp")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class LayoutReport:
    split_dir: Path
    pairs: int = 0
    optional: Dict[str, bool] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.pairs > 0 and not self.missing


def read_pairs(root: Path, split: str, paired: bool) -> List[Tuple[str, str]]:
    """(person, cloth) file names. Paired mode pairs each person with its own garment."""
    split_dir = root / split
    if paired:
        people = sorted(p.name for p in (split_dir / "image").iterdir() if p.suffix.lower() in IMAGE_EXTS)
        return [(p, p) for p in people]
    pairs_file = root / f"{split}_pairs.txt"
    pairs = []
    for line in pairs_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def _find(directory: Path, name: str) -> Optional[Path]:
    """File with the same stem in any supported extension (masks are often png for jpg images)."""
    stem = Path(name).stem
    for ext in (Path(name).suffix,) + IMAGE_EXTS + (".json",):
        p = directory / (stem + ext)
        if p.exists():
            return p
    return None


def validate_layout(root: Path, split: str, paired: bool = True) -> LayoutReport:
    split_dir = Path(root) / split
    report = LayoutReport(split_dir=split_dir)
    for required in ("image", "cloth"):
        if not (split_dir / required).is_dir():
            report.missing.append(f"{split_dir / required}/ (directory)")
    if report.missing:
        return report
    if not paired and not (Path(root) / f"{split}_pairs.txt").exists():
        report.missing.append(str(Path(root) / f"{split}_pairs.txt"))
        return report

    pairs = read_pairs(Path(root), split, paired)
    report.pairs = len(pairs)
    for d in OPTIONAL_DIRS:
        report.optional[d] = (split_dir / d).is_dir()
    for person, cloth in pairs:
        if _find(split_dir / "image", person) is None:
            report.missing.append(f"image/{person}")
        if _find(split_dir / "cloth", cloth) is None:
            report.missing.append(f"cloth/{cloth}")
        if report.optional["cloth-mask"] and _find(split_dir / "cloth-mask", cloth) is None:
            report.missing.append(f"cloth-mask/{cloth}")
        if report.optional["parse-mp"] and _find(split_dir / "parse-mp", person) is None:
            report.missing.append(f"parse-mp/{person}")
        if report.optional["pose-mp"] and _find(split_dir / "pose-mp", person) is None:
            report.missing.append(f"pose-mp/{person}")
    return report


def _rgb(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    h, w = size
    img = Image.open(path).convert("RGB").resize((w, h), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _mask(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    h, w = size
    img = Image.open(path).convert("L").resize((w, h), Image.NEAREST)
    return torch.from_numpy((np.asarray(img) > 127).astype(np.float32))[None]


def _labels(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    h, w = size
    img = Image.open(path).resize((w, h), Image.NEAREST)
    return torch.from_numpy(np.asarray(img, dtype=np.int64).copy())


def _pose(path: Path) -> torch.Tensor:
    """[33, 4] (x, y, z, visibility), normalized; all zeros if no person was detected."""
    data = json.loads(path.read_text(encoding="utf-8"))
    lm = data.get("landmarks") or []
    out = torch.zeros(33, 4)
    if lm:
        out[:] = torch.tensor(lm, dtype=torch.float32)
    return out


class TryOnDataset(Dataset):
    """Returns dicts: person, cloth ([3,H,W] in [-1,1]); cloth_mask [1,H,W]; parse [H,W] int64;
    pose [33,4]; person_name, cloth_name. Optional keys appear only if their folder exists."""

    def __init__(self, root, split: str = "test", size: Tuple[int, int] = (256, 192),
                 paired: bool = True):
        self.root = Path(root)
        self.split_dir = self.root / split
        self.size = size
        report = validate_layout(self.root, split, paired)
        if not report.ok:
            shown = "\n  ".join(report.missing[:10])
            raise FileNotFoundError(f"Invalid dataset at {self.split_dir} "
                                    f"({len(report.missing)} missing):\n  {shown}")
        self.optional = report.optional
        self.pairs = read_pairs(self.root, split, paired)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        person, cloth = self.pairs[i]
        d = self.split_dir
        item = {
            "person": _rgb(_find(d / "image", person), self.size),
            "cloth": _rgb(_find(d / "cloth", cloth), self.size),
            "person_name": person,
            "cloth_name": cloth,
        }
        if self.optional["cloth-mask"]:
            item["cloth_mask"] = _mask(_find(d / "cloth-mask", cloth), self.size)
        if self.optional["parse-mp"]:
            item["parse"] = _labels(_find(d / "parse-mp", person), self.size)
        if self.optional["pose-mp"]:
            item["pose"] = _pose(_find(d / "pose-mp", person))
        return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a try-on dataset layout.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--unpaired", action="store_true", help="check the <split>_pairs.txt pairs")
    args = parser.parse_args()
    r = validate_layout(args.root, args.split, paired=not args.unpaired)
    print(f"{r.split_dir}: {r.pairs} pairs, optional: {r.optional}")
    for m in r.missing[:20]:
        print("  missing:", m)
    if len(r.missing) > 20:
        print(f"  ... and {len(r.missing) - 20} more")
    raise SystemExit(0 if r.ok else 1)


if __name__ == "__main__":
    main()
