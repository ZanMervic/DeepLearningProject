# Android Model Eval

Simple Android-only internal app for manually comparing exported person-detection models on captured or gallery-selected images.

## What it does

- lets you capture a photo or select one from the gallery
- runs one selected detector at a time
- shows detection boxes on the image
- displays model name, threshold, inference latency, and detection count
- saves an annotated image only on explicit request
- logs every completed run to a CSV file that can be shared from the app

## Current bundled model contract

The app expects TFLite models in:

- `app/src/main/assets/models/`

Initial manifest entries are already present for:

- `yolo26n_chv_wp_coco_och_640.tflite`
- `yolo26n_chv_wp_coco_och_960.tflite`

If those files are not present, the app still builds, but model loading will fail at runtime until the assets are added.

## Build

From `android_model_eval/`:

```powershell
.\gradlew.bat testDebugUnitTest
.\gradlew.bat assembleDebug
```

Open the folder in Android Studio if you want to run it on a device or emulator.
