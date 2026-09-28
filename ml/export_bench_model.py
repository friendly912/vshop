"""Export the reference workload for the app's first-run device benchmark.

A MobileNetV2 (alpha 0.5) feature extractor at the Tier C input size (256x192) with fixed
random weights. Its conv/depthwise mix is representative of the phase 2 student model, so
its latency is a proxy for try-on speed. DeviceTierClassifier maps the measured time to a
device tier; the output also checks the GPU against the CPU (old drivers can be wrong).

Weights are stored as fp16 (about half the size); the GPU delegate runs fp16 natively.

Usage:
    python ml/export_bench_model.py --out android/app/src/main/assets/models
"""

import argparse
import contextlib
import io
import re
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

HEIGHT, WIDTH = 256, 192


def build() -> bytes:
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input(shape=(HEIGHT, WIDTH, 3), batch_size=1, name="image")
    model = tf.keras.applications.MobileNetV2(
        input_tensor=inp, alpha=0.5, include_top=False, weights=None)
    # from_keras_model freezes weights into constants and folds batch norm. Converting a
    # tf.function over the model instead keeps them as runtime variables (VAR_HANDLE ops),
    # which is slow and not supported by the GPU delegate.
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    return conv.convert()


def check_ops(data: bytes) -> list:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tf.lite.experimental.Analyzer.analyze(model_content=data, gpu_compatibility=True)
    text = buf.getvalue()
    ops = sorted(set(re.findall(r"Op#\d+ (\w+)\(", text)))
    if "VAR_HANDLE" in ops or "GPU COMPATIBILITY WARNING" in text:
        raise SystemExit(f"benchmark model is not frozen or not GPU-compatible: {ops}")
    return ops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = build()
    ops = check_ops(data)
    path = args.out / "bench_ref.tflite"
    path.write_bytes(data)

    it = tf.lite.Interpreter(model_content=data, num_threads=4)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    it.set_tensor(inp["index"], np.random.default_rng(0).uniform(-1, 1, inp["shape"]).astype(np.float32))
    for _ in range(3):
        it.invoke()
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        it.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"wrote {path} ({len(data) / 1e6:.1f} MB), in {list(inp['shape'])} -> out {list(out['shape'])}, "
          f"desktop CPU median {np.median(times):.1f} ms, ops {ops}")


if __name__ == "__main__":
    main()
