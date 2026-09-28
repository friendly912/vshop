"""Warping de-risk spike: which bilinear warp formulation can run on the LiteRT GPU delegate?

Every VITON model warps the garment with torch.nn.functional.grid_sample. LiteRT has no
native grid_sample op and old mobile GPUs (Android 8 era) support few gather-style ops,
so this script builds candidate formulations as TFLite models and, for each one:

  1. checks correctness against torch grid_sample (bilinear, zeros padding, align_corners=False),
  2. runs the LiteRT GPU compatibility analyzer,
  3. measures CPU latency here (relative numbers only; phones are measured in the app).

Variants:
  gathernd   - GATHER_ND bilinear, close to what converters emit for grid_sample
  gather     - 1-D GATHER on the flattened image
  shift_r{R} - sum of (2R+1)^2 shifted copies x bilinear tent weights; only PAD/SLICE/MUL/ADD.
               Exact while displacement <= R px; meant for coarse-to-fine residual warps.

The warp models are also written to the app's assets so WarpBenchmarkActivity can time them
on real phones (GPU vs CPU, and GPU-vs-CPU output diff to catch driver bugs).

Usage:
    python ml/warp_spike/build_warp_models.py
"""

import contextlib
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "android" / "app" / "src" / "main" / "assets" / "models" / "warp"
REPORT_DIR = Path(__file__).resolve().parent / "results"

SIZES = {"c": (256, 192), "ab": (512, 384)}
SHIFT_RADII = (2, 4)
SEED = 0


# ---------------------------------------------------------------- warp formulations (TF)

def _unnormalize(grid, h, w):
    """torch grid_sample convention with align_corners=False."""
    x = ((grid[..., 0] + 1.0) * w - 1.0) * 0.5
    y = ((grid[..., 1] + 1.0) * h - 1.0) * 0.5
    return x, y


def _bilinear_corners(x, y):
    x0 = tf.floor(x)
    y0 = tf.floor(y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0
    corners = (
        (x0, y0, (x1 - x) * (y1 - y)),
        (x0, y1, (x1 - x) * (y - y0)),
        (x1, y0, (x - x0) * (y1 - y)),
        (x1, y1, (x - x0) * (y - y0)),
    )
    return corners


def _valid(xi, yi, h, w):
    inside = tf.logical_and(
        tf.logical_and(xi >= 0.0, xi <= w - 1.0),
        tf.logical_and(yi >= 0.0, yi <= h - 1.0))
    return tf.cast(inside, tf.float32)


def warp_gathernd(img, grid):
    h, w = img.shape[1], img.shape[2]
    x, y = _unnormalize(grid, h, w)
    out = 0.0
    for xi, yi, wt in _bilinear_corners(x, y):
        xc = tf.cast(tf.clip_by_value(xi, 0.0, w - 1.0), tf.int32)
        yc = tf.cast(tf.clip_by_value(yi, 0.0, h - 1.0), tf.int32)
        idx = tf.stack([tf.zeros_like(xc), yc, xc], axis=-1)
        px = tf.gather_nd(img, idx)
        out += (wt * _valid(xi, yi, h, w))[..., None] * px
    return out


def warp_gather(img, grid):
    h, w = img.shape[1], img.shape[2]
    flat = tf.reshape(img, [h * w, 3])
    x, y = _unnormalize(grid, h, w)
    out = 0.0
    for xi, yi, wt in _bilinear_corners(x, y):
        xc = tf.cast(tf.clip_by_value(xi, 0.0, w - 1.0), tf.int32)
        yc = tf.cast(tf.clip_by_value(yi, 0.0, h - 1.0), tf.int32)
        idx = tf.reshape(yc * w + xc, [-1])
        px = tf.reshape(tf.gather(flat, idx, axis=0), [1, h, w, 3])
        out += (wt * _valid(xi, yi, h, w))[..., None] * px
    return out


def make_warp_shift(radius):
    def warp_shift(img, grid):
        h, w = img.shape[1], img.shape[2]
        x, y = _unnormalize(grid, h, w)
        dx = x - tf.range(w, dtype=tf.float32)[None, None, :]
        dy = y - tf.range(h, dtype=tf.float32)[None, :, None]
        r = radius
        padded = tf.pad(img, [[0, 0], [r, r], [r, r], [0, 0]])  # zeros = torch padding_mode
        out = 0.0
        for sy in range(-r, r + 1):
            wy = tf.nn.relu(1.0 - tf.abs(dy - sy))
            for sx in range(-r, r + 1):
                wx = tf.nn.relu(1.0 - tf.abs(dx - sx))
                shifted = padded[:, r + sy:r + sy + h, r + sx:r + sx + w, :]
                out += (wy * wx)[..., None] * shifted
        return out
    return warp_shift


# ---------------------------------------------------------------- test data + reference

def identity_grid(h, w):
    xs = (np.arange(w, dtype=np.float32) * 2 + 1) / w - 1
    ys = (np.arange(h, dtype=np.float32) * 2 + 1) / h - 1
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy], axis=-1)[None]  # [1,H,W,2]


def smooth_flow_px(rng, h, w, max_px):
    """Smooth random displacement field in pixels, |d| <= max_px per axis."""
    coarse = rng.uniform(-1, 1, size=(1, 2, 4, 3)).astype(np.float32)
    up = F.interpolate(torch.from_numpy(coarse), size=(h, w), mode="bicubic", align_corners=True)
    d = up.numpy().transpose(0, 2, 3, 1)
    d = d / max(np.abs(d).max(), 1e-6) * max_px
    return d  # [1,H,W,2] (dx, dy)


def to_grid(flow_px, h, w):
    g = identity_grid(h, w).copy()
    g[..., 0] += flow_px[..., 0] * 2.0 / w
    g[..., 1] += flow_px[..., 1] * 2.0 / h
    return g


def torch_reference(img, grid):
    out = F.grid_sample(torch.from_numpy(img).permute(0, 3, 1, 2), torch.from_numpy(grid),
                        mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.permute(0, 2, 3, 1).numpy()


# ---------------------------------------------------------------- tflite helpers

def convert(fn, h, w):
    spec_img = tf.TensorSpec([1, h, w, 3], tf.float32, name="image")
    spec_grid = tf.TensorSpec([1, h, w, 2], tf.float32, name="grid")
    f = tf.function(fn, input_signature=[spec_img, spec_grid])
    conv = tf.lite.TFLiteConverter.from_concrete_functions([f.get_concrete_function()], f)
    return conv.convert()


def run_tflite(model, img, grid, threads=4, runs=20):
    it = tf.lite.Interpreter(model_content=model, num_threads=threads)
    it.allocate_tensors()
    ins = {d["name"]: d["index"] for d in it.get_input_details()}
    img_idx = next(i for n, i in ins.items() if "image" in n)
    grid_idx = next(i for n, i in ins.items() if "grid" in n)
    out_idx = it.get_output_details()[0]["index"]
    it.set_tensor(img_idx, img)
    it.set_tensor(grid_idx, grid)
    for _ in range(3):
        it.invoke()
    t0 = time.perf_counter()
    for _ in range(runs):
        it.invoke()
    ms = (time.perf_counter() - t0) * 1000 / runs
    return it.get_tensor(out_idx), ms


def gpu_analysis(model):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tf.lite.experimental.Analyzer.analyze(model_content=model, gpu_compatibility=True)
    text = buf.getvalue()
    ops = sorted(set(re.findall(r"Op#\d+ (\w+)\(", text)))
    bad = sorted(set(re.findall(r"(\w+): (?:Not supported|OP is supported by GPU but)", text)))
    compatible = "GPU COMPATIBILITY WARNING" not in text
    return {"ops": ops, "gpu_compatible": compatible, "gpu_issues": bad, "raw": text}


# ---------------------------------------------------------------- main

def main():
    rng = np.random.default_rng(SEED)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    variants = [("gathernd", warp_gathernd, None), ("gather", warp_gather, None)]
    variants += [(f"shift_r{r}", make_warp_shift(r), r) for r in SHIFT_RADII]

    rows = []
    for size_name, (h, w) in SIZES.items():
        img = rng.uniform(-1, 1, size=(1, h, w, 3)).astype(np.float32)
        # Large warp: up to 20% of the width, some samples fall outside the image.
        big_grid = to_grid(smooth_flow_px(rng, h, w, 0.2 * w), h, w)

        for name, fn, radius in variants:
            model = convert(fn, h, w)
            fname = f"warp_{name}_{h}x{w}.tflite"
            (ASSET_DIR / fname).write_bytes(model)

            if radius is None:
                grid = big_grid
            else:
                grid = to_grid(smooth_flow_px(rng, h, w, radius - 0.01), h, w)
            out, ms = run_tflite(model, img, grid)
            err = float(np.abs(out - torch_reference(img, grid)).max())

            # For shift variants also show what happens beyond the radius (expected: wrong).
            over_err = None
            if radius is not None:
                g_over = to_grid(smooth_flow_px(rng, h, w, radius * 2), h, w)
                o, _ = run_tflite(model, img, g_over, runs=1)
                over_err = float(np.abs(o - torch_reference(img, g_over)).max())

            gpu = gpu_analysis(model)
            (REPORT_DIR / f"{fname}.analyzer.txt").write_text(gpu["raw"], encoding="utf-8")
            row = {
                "model": fname, "variant": name, "size": f"{h}x{w}",
                "bytes": len(model), "max_abs_err": err, "err_beyond_radius": over_err,
                "cpu_ms_x86": round(ms, 2), "ops": gpu["ops"],
                "gpu_compatible": gpu["gpu_compatible"], "gpu_issues": gpu["gpu_issues"],
            }
            rows.append(row)
            print(f"{fname:32s} err={err:.2e} cpu={ms:7.2f} ms  gpu_ok={gpu['gpu_compatible']}"
                  f"  issues={gpu['gpu_issues']}  ops={gpu['ops']}")

    (REPORT_DIR / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nModels -> {ASSET_DIR}\nReport -> {REPORT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
