"""Generate annotations for datasets that lack them (e.g. self-collected photos).

    python -m vton.preprocess --root data --split train            # everything missing
    python -m vton.preprocess --root data --split train --pose     # just pose

Writes next to image/ and cloth/ (see vton.data):
    pose-mp/<person>.json    33 MediaPipe landmarks [x, y, z, visibility], normalized
    parse-mp/<person>.png    6-class labels: 0 background, 1 hair, 2 body skin,
                             3 face skin, 4 clothes, 5 others
    cloth-mask/<cloth>.png   garment mask for product photos on a white background

Pose and parsing use the same MediaPipe models as the Android app, so training-time and
on-device preprocessing agree.
"""

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from vton.data import IMAGE_EXTS

ROOT = Path(__file__).resolve().parents[2]
APP_MODELS = ROOT / "android" / "app" / "src" / "main" / "assets" / "models"
CACHE = ROOT / "ml" / ".cache" / "models"
MODEL_URLS = {
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "selfie_multiclass_256x256.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}


# ------------------------------------------------------------------ cloth mask

def cloth_mask(rgb: np.ndarray, white_threshold: int = 30) -> np.ndarray:
    """Foreground mask (uint8, 0/255) of a garment photographed on a white background.

    Pixels farther than `white_threshold` from pure white (max channel distance) are
    foreground; then holes (e.g. white prints) are filled, specks removed, and only the
    largest component kept. Limitation: white garments on white backgrounds need a real
    segmentation model or manual masks.
    """
    h, w = rgb.shape[:2]
    dist = 255 - rgb.astype(np.int16).min(axis=2)
    fg = (dist > white_threshold).astype(np.uint8) * 255

    k = max(3, (min(h, w) // 100) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return np.zeros((h, w), np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg = np.where(labels == largest, 255, 0).astype(np.uint8)
    return _fill_holes(fg)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = np.pad(mask, 1, constant_values=0).copy()
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood[1:-1, 1:-1])
    return cv2.bitwise_or(mask, holes)


# ------------------------------------------------------------------ MediaPipe

def model_path(name: str) -> Path:
    """The app's copy if present, else a cached download."""
    for candidate in (APP_MODELS / name, CACHE / name):
        if candidate.exists():
            return candidate
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / name
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(MODEL_URLS[name], target)
    return target


class MediaPipeAnnotator:
    def __init__(self, pose: bool, parse: bool):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.pose = None
        self.seg = None
        if pose:
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path("pose_landmarker_lite.task"))),
                running_mode=vision.RunningMode.IMAGE, num_poses=1))
        if parse:
            self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(model_path("selfie_multiclass_256x256.tflite"))),
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True, output_confidence_masks=False))

    def _image(self, rgb: np.ndarray):
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def landmarks(self, rgb: np.ndarray) -> list:
        """[[x, y, z, visibility]] * 33, or [] if no person."""
        res = self.pose.detect(self._image(rgb))
        if not res.pose_landmarks:
            return []
        return [[lm.x, lm.y, lm.z, lm.visibility if lm.visibility is not None else 0.0]
                for lm in res.pose_landmarks[0]]

    def labels(self, rgb: np.ndarray) -> np.ndarray:
        res = self.seg.segment(self._image(rgb))
        # numpy_view() is H x W x 1 for single-channel masks.
        labels = np.array(res.category_mask.numpy_view(), dtype=np.uint8, copy=True)
        return labels.reshape(rgb.shape[0], rgb.shape[1])

    def close(self) -> None:
        for task in (self.pose, self.seg):
            if task is not None:
                task.close()


# ------------------------------------------------------------------ CLI

def _images(directory: Path):
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def run(root: Path, split: str, pose: bool, parse: bool, masks: bool, overwrite: bool,
        limit: Optional[int] = None) -> dict:
    split_dir = Path(root) / split
    counts = {"pose": 0, "parse": 0, "cloth_mask": 0, "no_person": 0}

    if pose or parse:
        ann = MediaPipeAnnotator(pose, parse)
        try:
            (split_dir / "pose-mp").mkdir(exist_ok=True)
            (split_dir / "parse-mp").mkdir(exist_ok=True)
            for p in _images(split_dir / "image")[:limit]:
                pose_out = split_dir / "pose-mp" / f"{p.stem}.json"
                parse_out = split_dir / "parse-mp" / f"{p.stem}.png"
                need_pose = pose and (overwrite or not pose_out.exists())
                need_parse = parse and (overwrite or not parse_out.exists())
                if not (need_pose or need_parse):
                    continue
                rgb = np.asarray(Image.open(p).convert("RGB"))
                if need_pose:
                    lm = ann.landmarks(rgb)
                    counts["no_person"] += 0 if lm else 1
                    pose_out.write_text(json.dumps(
                        {"image_size": [rgb.shape[0], rgb.shape[1]], "landmarks": lm}), encoding="utf-8")
                    counts["pose"] += 1
                if need_parse:
                    Image.fromarray(ann.labels(rgb)).save(parse_out)
                    counts["parse"] += 1
        finally:
            ann.close()

    if masks:
        (split_dir / "cloth-mask").mkdir(exist_ok=True)
        for p in _images(split_dir / "cloth")[:limit]:
            out = split_dir / "cloth-mask" / f"{p.stem}.png"
            if overwrite or not out.exists():
                Image.fromarray(cloth_mask(np.asarray(Image.open(p).convert("RGB")))).save(out)
                counts["cloth_mask"] += 1
    return counts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--pose", action="store_true")
    ap.add_argument("--parse", action="store_true")
    ap.add_argument("--cloth-mask", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, help="process at most N images (for a quick look)")
    args = ap.parse_args(argv)
    everything = not (args.pose or args.parse or args.cloth_mask)
    counts = run(args.root, args.split, args.pose or everything, args.parse or everything,
                 args.cloth_mask or everything, args.overwrite, args.limit)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
