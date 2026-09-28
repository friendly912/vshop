# OmoiPassion: on-device virtual try-on

- `android/`: standalone native Java app (minSdk 26 / Android 8.0) using LiteRT, CameraX and MediaPipe.
- `ml/`: Python model work. It currently has one script that exports placeholder models for testing the app.
- `docs/ROADMAP.md`: the phased plan.

## Run the app

1. Open `android/` in Android Studio (Ladybug or newer) and let it sync. Gradle 8.11.1 is
   set in `gradle/wrapper/gradle-wrapper.properties`.
2. Generate placeholder models, then add the MediaPipe pose model:
   ```
   python ml/export_placeholder_model.py --out android/app/src/main/assets/models
   ```
   Download `pose_landmarker_lite.task` into the same folder. The link is in
   `android/app/src/main/assets/models/README.md`.
3. Run on a device. Take or pick a person photo, pick a garment photo, then tap **Try on**.
