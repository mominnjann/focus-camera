# FocusCamera 🚀

**FocusCamera** is a simple real-time webcam utility that automatically centers the frame on a detected face (auto-framing) and overlays a basic motion detector. It uses MediaPipe Face Detection with a short-range BlazeFace TFLite model.

---

## Features ✅

- Auto-framing around a detected face with smoothing
- Basic motion detection overlay inside the cropped frame
- Lightweight, single-file demo: `main.py`

---

## Requirements 🔧

- Python 3.8+
- Packages (install via pip):

```bash
pip install opencv-python mediapipe numpy
```

Note: On Windows, installing `mediapipe` via pip should work for modern Python versions. If you encounter issues, check MediaPipe installation instructions for Windows.

---

## Files

- `main.py` — demo script (runs webcam, auto-framing, motion detection)
- `models/blaze_face_short_range.tflite` — included face detection model used by the script

---

## Quick Start ▶️

1. Open a terminal in the `focuscamera` folder
2. (Optional) Create and activate a virtualenv
3. Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

4. Run:

```bash
python main.py
```

5. Press `q` in the display window to quit.

---

## Configuration (in `main.py`) ⚙️

- `INPUT_RES`, `OUTPUT_RES` — input/output resolution
- `CROP_SCALE` — proportion of frame used when cropping around the face
- `SMOOTH_TAU` — smoothing time constant for face-centering

You can edit these constants at the top of `main.py` to tune behavior.

---

## Troubleshooting ⚠️

- No camera detected: try a different camera index in `cv2.VideoCapture()`
- `mediapipe` import errors: confirm Python version compatibility or try a specific `mediapipe` wheel for Windows
- If detections are unstable, try adjusting `CROP_SCALE` and `SMOOTH_TAU` values

---

## Notes

This is a lightweight demo intended for experimentation and prototyping only.

---

