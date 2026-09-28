import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `vton` importable


def _rgb(rng, h, w):
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


@pytest.fixture
def dataset(tmp_path):
    """Tiny VITON-HD-layout dataset with every optional folder."""
    rng = np.random.default_rng(0)
    root = tmp_path / "data"
    split = root / "test"
    for d in ("image", "cloth", "cloth-mask", "parse-mp", "pose-mp"):
        (split / d).mkdir(parents=True)
    for name in ("a", "b"):
        _rgb(rng, 64, 48).save(split / "image" / f"{name}.jpg")
        _rgb(rng, 64, 48).save(split / "cloth" / f"{name}.jpg")
        Image.fromarray((rng.random((64, 48)) > 0.5).astype(np.uint8) * 255).save(split / "cloth-mask" / f"{name}.png")
        Image.fromarray(rng.integers(0, 6, (64, 48), dtype=np.uint8)).save(split / "parse-mp" / f"{name}.png")
    (split / "pose-mp" / "a.json").write_text(json.dumps(
        {"landmarks": [[0.5, 0.5, 0.0, 0.9]] * 33}), encoding="utf-8")
    (split / "pose-mp" / "b.json").write_text(json.dumps({"landmarks": []}), encoding="utf-8")
    (root / "test_pairs.txt").write_text("a.jpg b.jpg\nb.jpg a.jpg\n", encoding="utf-8")
    return root
