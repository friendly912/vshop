"""Export placeholder try-on models that follow the app's I/O contract.

The "model" alpha-blends the garment over the torso region of the person image.
It exists only to exercise the Android pipeline (loading, GPU/CPU fallback,
tensor layout, compositing) before the phase 2 student model is trained.

Usage:
    pip install tensorflow
    python ml/export_placeholder_model.py --out android/app/src/main/assets/models
"""

import argparse
from pathlib import Path

import tensorflow as tf

TIERS = {
    "viton_c.tflite": (256, 192),
    "viton_b.tflite": (512, 384),
}


def build(height: int, width: int) -> bytes:
    # Soft vertical band covering roughly the torso (rows 25%-70%, columns 20%-80%).
    ys = tf.linspace(0.0, 1.0, height)[:, None]
    xs = tf.linspace(0.0, 1.0, width)[None, :]
    band_y = tf.sigmoid((ys - 0.25) * 40.0) * tf.sigmoid((0.70 - ys) * 40.0)
    band_x = tf.sigmoid((xs - 0.20) * 40.0) * tf.sigmoid((0.80 - xs) * 40.0)
    mask = tf.reshape(band_y * band_x * 0.8, [1, height, width, 1])

    spec = tf.TensorSpec([1, height, width, 3], tf.float32)

    @tf.function(input_signature=[spec, spec])
    def tryon(person, garment):
        return person * (1.0 - mask) + garment * mask

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tryon.get_concrete_function()], tryon
    )
    return converter.convert()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (h, w) in TIERS.items():
        path = args.out / name
        path.write_bytes(build(h, w))
        print(f"wrote {path} ({h}x{w})")


if __name__ == "__main__":
    main()
