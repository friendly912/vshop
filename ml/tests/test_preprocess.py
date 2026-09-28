import json

import numpy as np
import pytest
from PIL import Image

from vton import preprocess


def _garment_on_white():
    img = np.full((200, 150, 3), 255, np.uint8)
    img[50:150, 40:110] = (30, 60, 160)          # garment
    img[90:110, 65:85] = 255                      # white print inside it
    img[10:12, 10:12] = 0                         # dust specks on the background
    img[185:187, 140:142] = 0
    expected = np.zeros((200, 150), bool)
    expected[50:150, 40:110] = True
    return img, expected


def test_cloth_mask_fills_prints_and_drops_specks():
    img, expected = _garment_on_white()
    mask = preprocess.cloth_mask(img) > 0
    iou = (mask & expected).sum() / (mask | expected).sum()
    assert iou > 0.99
    assert not mask[10, 10] and not mask[186, 141]
    assert mask[100, 75]  # the white print is part of the garment


def test_cloth_mask_empty_for_blank_image():
    assert preprocess.cloth_mask(np.full((50, 40, 3), 255, np.uint8)).max() == 0


def test_run_writes_cloth_masks_only(dataset):
    (dataset / "test" / "cloth-mask" / "a.png").unlink()
    counts = preprocess.run(dataset, "test", pose=False, parse=False, masks=True, overwrite=False)
    assert counts["cloth_mask"] == 1  # b.png already existed
    assert (dataset / "test" / "cloth-mask" / "a.png").exists()


def test_mediapipe_pose_and_parse(dataset):
    pytest.importorskip("mediapipe")
    split = dataset / "test"
    for d in ("pose-mp", "parse-mp"):
        for f in (split / d).iterdir():
            f.unlink()
    counts = preprocess.run(dataset, "test", pose=True, parse=True, masks=False, overwrite=False)
    assert counts["pose"] == 2 and counts["parse"] == 2
    data = json.loads((split / "pose-mp" / "a.json").read_text(encoding="utf-8"))
    assert data["image_size"] == [64, 48]
    assert isinstance(data["landmarks"], list) and len(data["landmarks"]) in (0, 33)
    labels = np.asarray(Image.open(split / "parse-mp" / "a.png"))
    assert labels.shape == (64, 48) and labels.max() <= 5
