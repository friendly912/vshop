# OmoiPassion: on-device virtual try-on

[![Android](https://github.com/friendly912/vshop/actions/workflows/android.yml/badge.svg)](https://github.com/friendly912/vshop/actions/workflows/android.yml)

- `android/`: standalone native Java app (minSdk 26 / Android 8.0) using LiteRT, CameraX and MediaPipe.
- `ml/`: Python model work: a placeholder model exporter and the warp test (`ml/warp_spike/`).
- `docs/ROADMAP.md`: the phased plan.

## Get a test APK without building

Every push to `main` builds a debug APK with all models included. Open the latest run under
**Actions → Android** and download the `app-debug-…` artifact (you must be signed in to
GitHub). The same run has the lint report and the warp test results.

## Build locally

1. Install the Python tools: `pip install -r ml/requirements.txt`.
2. Generate the models and download the pose model:
   ```
   python ml/export_placeholder_model.py --out android/app/src/main/assets/models
   python ml/warp_spike/build_warp_models.py
   ```
   Download `pose_landmarker_lite.task` into `android/app/src/main/assets/models/`. The link is in
   that folder's README.
3. Open `android/` in Android Studio, or run `./gradlew assembleDebug` from `android/`.
4. Run on a device. Take or pick a person photo, pick a garment photo, then tap **Try on**.
   The menu has the **Warp benchmark**.
