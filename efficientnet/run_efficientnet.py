"""run_efficientnet.py
Run EfficientNetB0 model in real time using a webcam.
Configuration via command line arguments; uses OpenCV for face detection.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, Tuple

import cv2
import numpy as np
from tensorflow.keras.models import load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logger = logging.getLogger(__name__)

COLORS: Dict[str, Tuple[int, int, int]] = {
    "Angry": (0, 0, 255),
    "Disgust": (0, 128, 0),
    "Fear": (128, 0, 128),
    "Happy": (0, 255, 255),
    "Neutral": (200, 200, 200),
    "Sad": (255, 0, 0),
    "Surprise": (0, 165, 255),
}


def main() -> None:
    # Simple fixed configuration (uses your project paths)
    MODEL_PATH = r"C:\Users\Lazerai\Downloads\Emotion-Video\model_efficientnet.h5"
    CLASSES_PATH = r"C:\Users\Lazerai\Downloads\Emotion-Video\classes_efficientnet.json"
    IMG_SIZE = (224, 224)
    CAMERA = 0
    PROTOTXT = r"C:\Users\Lazerai\Downloads\Emotion-Video\deploy.prototxt"
    FACE_MODEL = r"C:\Users\Lazerai\Downloads\Emotion-Video\res10_300x300_ssd_iter_140000.caffemodel"
    MIN_CONFIDENCE = 0.5

    # Minimal startup messages
    print("Starting EfficientNet webcam script")

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return
    if not os.path.exists(CLASSES_PATH):
        print(f"Classes file not found: {CLASSES_PATH}")
        return

    model = load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r") as f:
        emotions = json.load(f)

    # Use DNN face detector if both files exist, otherwise Haar cascade
    dnn_net = None
    face_cascade = None
    if os.path.exists(PROTOTXT) and os.path.exists(FACE_MODEL):
        try:
            dnn_net = cv2.dnn.readNetFromCaffe(PROTOTXT, FACE_MODEL)
            print("Using DNN face detector")
        except Exception:
            print("Failed to init DNN detector, falling back to Haar cascade")
    if dnn_net is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        print("Using Haar cascade face detector")

    cap = cv2.VideoCapture(CAMERA)
    print("Webcam started, press 'q' to quit")

    img_size = IMG_SIZE

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            break

        # Detect faces using DNN (if available) or Haar cascade
        h_frame, w_frame = frame.shape[:2]
        faces = []
        if dnn_net is not None:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            dnn_net.setInput(blob)
            detections = dnn_net.forward()
            for i in range(0, detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf < MIN_CONFIDENCE:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w_frame, h_frame, w_frame, h_frame])
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(0, startX); startY = max(0, startY); endX = min(w_frame - 1, endX); endY = min(h_frame - 1, endY)
                faces.append((startX, startY, endX - startX, endY - startY))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face = frame[y : y + h, x : x + w]
            face = cv2.resize(face, img_size).astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)

            pred = model.predict(face, verbose=0)[0]
            idx = int(np.argmax(pred))
            emotion = emotions[idx]
            confidence = float(pred[idx]) * 100.0

            color = COLORS.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{emotion}: {confidence:.1f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("EfficientNetB0 Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Stopped")


if __name__ == "__main__":
    main()