# Warp test (phase 2 risk)

The question: which bilinear warp (the `grid_sample` step every VITON model uses) can run
fast and correctly on the Tier C phones (Android 8 era)?

Run `python ml/warp_spike/build_warp_models.py`. It writes the models into the app's assets
and a report to `results/summary.json`. On the phone, open **menu → Warp benchmark → Run**,
then use **Copy report**.

## Running without a local Android 8 phone

What matters is the GPU and driver generation (Adreno 5xx, Mali-G71/G72), not the Android
version. Any 2017–2018 phone works, even one updated to Android 9 or 10.

- **Samsung Remote Test Lab** (free, interactive): reserve a Galaxy S8/S9 or an older
  A-series phone, install `app-debug.apk`, open menu → Warp benchmark → Run, then Copy report.
- **Device farms with remote adb** (AWS Device Farm remote access, BrowserStack): connect with
  adb, then run `android/scripts/run_warp_bench.sh [-s SERIAL]`. It installs the app, runs
  the benchmark with no taps (`--ez warp_autorun true`) and saves the log to `results/`.
- **Firebase Test Lab** (Robo test): launch the app with the extra `warp_autorun=true` and read
  the `WarpBench` lines from the logcat artifact. The run is complete when you see `WARP_BENCH_DONE`.

The local API 26 emulator (`viton_api26b`) segfaults at startup on the dev PC with emulator
37.1.11 and 37.3.1 (WHPX path; it starts only with acceleration off). Android 8 API
compatibility is instead checked statically: `./gradlew lintDebug` reports 0 NewApi issues.

## Variants

| Variant | Ops | Limitation |
|---|---|---|
| A `gathernd` | GATHER_ND, CAST, FLOOR, LOGICAL_AND, … | Carries about 0.8 MB of constant index tensors at 512x384 |
| B `gather` | GATHER on the flattened image, CAST, FLOOR, … | None |
| C `shift_rR` | PAD, STRIDED_SLICE, ABS, MUL, ADD only | Exact only while displacement is at most R px (error about 1.0 beyond it) |
| D Java warp | `WarpOps.bilinear`, outside the model | Needs a GPU→CPU→GPU round trip between model stages |

## Desktop results (2026-09-28, x86 CPU, TFLite 2.21)

| Model | Max error vs torch | CPU ms (256x192) | CPU ms (512x384) | Analyzer |
|---|---|---|---|---|
| gathernd | 1.2e-7 | 6.5 | 31.9 | GPU-compatible |
| gather | 1.2e-7 | 4.6 | 21.5 | GPU-compatible |
| shift_r2 | 0 | 8.6 | 31.4 | GPU-compatible |
| shift_r4 | 0 | 27.0 | 65.5 | GPU-compatible |
| Java warp (D) | 0 | n/a | n/a | not applicable |

All variants are numerically correct. The analyzer's "GPU-compatible" verdict applies to
TFLite 2.21 only. It says nothing about LiteRT 1.0.1 on old Adreno 5xx or Mali-G71 drivers,
so **the phone benchmark decides**.

## How to decide from the phone reports

Collect reports from at least one Tier C phone (Android 8/9) and one recent phone.

1. **B `gather` runs on GPU on Tier C, with GPU-vs-CPU diff under 1e-2, and is fast:**
   use a standard single-model flow-warp architecture (PF-AFN / DM-VTON style). This is the
   simplest path.
2. **The GPU rejects gather, or its output is wrong, but C `shift` works:** design the
   student model as coarse-to-fine. Do a large warp at 1/8–1/16 resolution, where gather on
   CPU is cheap, then refine the residual warp at full resolution with shift-blend (R = 2).
3. **Neither works well:** split the model into a flow network, then the Java warp (D) on
   CPU, then the generator. Choose this if the Java warp at 256x192 stays in the low tens of
   milliseconds on Tier C.

Paths 2 and 3 can be combined. Record the decision in `docs/ROADMAP.md` (phase 2).
