import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time, os

# --- Config ---
INPUT_RES = (1280, 720)
OUTPUT_RES = (1280, 720)
CROP_SCALE = 0.7
DEADBAND = 0.05
SMOOTH_TAU = 0.25

# Load Face Detection model
script_dir = os.path.dirname(os.path.abspath(__file__))
det_model_path = os.path.join(script_dir, "models", "blaze_face_short_range.tflite")
base_options = python.BaseOptions(model_asset_path=det_model_path)
detector = vision.FaceDetector.create_from_options(
    vision.FaceDetectorOptions(base_options=base_options)
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_RES[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_RES[1])

prev_center = None
last_t = time.time()
prev_gray = None   # for motion detection

def clamp(v, lo, hi): return max(lo, min(hi, v))

def compute_crop(frame_w, frame_h, face_center, prev_center, dt):
    if prev_center is None:
        smoothed = face_center
    else:
        alpha = min(1.0, dt / SMOOTH_TAU)
        smoothed = (alpha*np.array(face_center)+(1-alpha)*np.array(prev_center))
    cx, cy = smoothed
    crop_w = int(frame_w*CROP_SCALE)
    crop_h = int(frame_h*CROP_SCALE)
    x0 = int(clamp(cx-crop_w//2,0,frame_w-crop_w))
    y0 = int(clamp(cy-crop_h//2,0,frame_h-crop_h))
    return (x0,y0,crop_w,crop_h),(cx,cy)

print("Webcam started.")
while True:
    ret, frame = cap.read()
    if not ret: break
    dt = time.time()-last_t; last_t=time.time()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    frame_h, frame_w = frame.shape[:2]
    face_center = None
    if result.detections:
        box = result.detections[0].bounding_box
        face_center = (box.origin_x+box.width//2, box.origin_y+box.height//2)
    if face_center is None:
        face_center = prev_center or (frame_w//2, frame_h//2)

    crop, smoothed_center = compute_crop(frame_w, frame_h, face_center, prev_center, dt)
    prev_center = smoothed_center
    x0,y0,cw,ch = crop
    cropped = frame[y0:y0+ch, x0:x0+cw]
    out = cv2.resize(cropped, OUTPUT_RES)

    # --- Motion detector ---
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    if prev_gray is None:
        prev_gray = gray
    else:
        diff = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 500:  # ignore small changes
                continue
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(out, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(out, "MOTION", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        prev_gray = gray

    cv2.imshow("Auto-Frame + Motion Detector", out)
    if cv2.waitKey(1)&0xFF==ord('q'): break

cap.release(); cv2.destroyAllWindows()