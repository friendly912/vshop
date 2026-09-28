# Models

Model files are git-ignored. Put them here for local builds; release builds should
deliver Tier A/B models through Play Asset Delivery.

| File | Required | Source |
|---|---|---|
| `viton_c.tflite` | Yes (baseline for every device) | Phase 2 micro student, 256x192 |
| `viton_b.tflite` | Optional | Phase 2 student, 512x384 |
| `viton_a.tflite` | Optional | Phase 2/6 flagship model, 512x384 |
| `pose_landmarker_lite.task` | Recommended (without it, the pose check and hand keeping are skipped) | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task |
| `bench_ref.tflite` | Recommended (without it, the tier is guessed from RAM and Android version) | `python ml/export_bench_model.py --out <this folder>` |
| `selfie_multiclass_256x256.tflite` | Recommended (without it, face, hair and hands aren't kept from the original) | https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite |

The segmenter's classes (from its embedded `labels.txt`) are 0 background, 1 hair, 2 body skin,
3 face skin, 4 clothes and 5 others. It's float32 and adds 16 MB to the APK. Check its model
card license before release.

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
