# Models

Model files are git-ignored. Put them here for local builds; release builds should
deliver Tier A/B models through Play Asset Delivery.

| File | Required | Source |
|---|---|---|
| `viton_c.tflite` | Yes (baseline for every device) | Phase 2 micro student, 256x192 |
| `viton_b.tflite` | Optional | Phase 2 student, 512x384 |
| `viton_a.tflite` | Optional | Phase 2/6 flagship model, 512x384 |
| `pose_landmarker_lite.task` | Recommended (without it, the pose check is skipped) | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task |

If a tier's model is missing, the app falls back to the next lighter one
(`ModelSpec.resolve`).

## Try-on model I/O contract

| | Index | Shape (NHWC) | Type | Content |
|---|---|---|---|---|
| Input | 0 | `[1, H, W, 3]` | float32, [-1, 1] | Person image cropped to W:H = 3:4, RGB |
| Input | 1 | `[1, H, W, 3]` | float32, [-1, 1] | Garment image letterboxed on white, RGB |
| Output | 0 | `[1, H, W, 3]` | float32, [-1, 1] | Try-on result |

H x W is 256x192 for Tier C and 512x384 for Tiers A/B. INT8-quantized models must keep
float32 inputs and outputs (quantize internally). Avoid dynamic shapes, and avoid ops the
GPU delegate doesn't support on old GLES 3.1 drivers (see the grid_sample note in
`docs/ROADMAP.md`).

To test the app before a trained model exists, generate a placeholder with
`ml/export_placeholder_model.py`.
