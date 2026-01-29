"""run_deepface.py
Run a DeepFace-embedding-based classifier in real time.
Use this script to run the dense classifier trained on VGG-Face embeddings.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Tuple, List

import cv2
import numpy as np
from deepface import DeepFace
from tensorflow.keras.models import load_model
import pickle

os.environ["DEEPFACE_LOG_LEVEL"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logger = logging.getLogger(__name__)

COLORS = {
    "Angry": (0, 0, 255),
    "Disgust": (0, 128, 0),
    "Fear": (128, 0, 128),
    "Happy": (0, 255, 255),
    "Neutral": (200, 200, 200),
    "Sad": (255, 0, 0),
    "Surprise": (0, 165, 255),
}


def get_embedding(face_img: np.ndarray):
    try:
        result = DeepFace.represent(face_img, model_name="VGG-Face", enforce_detection=False)
        return np.array(result[0]["embedding"])
    except Exception:
        logger.exception("Failed to compute embedding for face.")
        return None


def main() -> None:
    # Simple fixed configuration (uses your project paths)
    MODEL_PATH = r"C:\Users\Lazerai\Downloads\Emotion-Video\model_deepface.h5"
    ENCODER_PATH = r"C:\Users\Lazerai\Downloads\Emotion-Video\encoder_deepface.pkl"
    IMG_SIZE = (224, 224)
    CAMERA = 0
    PROTOTXT = r"C:\Users\Lazerai\Downloads\Emotion-Video\deploy.prototxt"
    FACE_MODEL = r"C:\Users\Lazerai\Downloads\Emotion-Video\res10_300x300_ssd_iter_140000.caffemodel"
    MIN_CONFIDENCE = 0.5

    print("Starting DeepFace webcam script")

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return
    if not os.path.exists(ENCODER_PATH):
        print(f"Encoder file not found: {ENCODER_PATH}")
        return

    model = load_model(MODEL_PATH)

    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    emotions: List[str] = list(encoder.classes_)

    # Use DNN if both files exist, otherwise Haar
    dnn_net = None
    face_cascade = None
    if os.path.exists(PROTOTXT) and os.path.exists(FACE_MODEL):
        try:
            dnn_net = cv2.dnn.readNetFromCaffe(PROTOTXT, FACE_MODEL)
            print("Using DNN face detector")
        except Exception:
            print("Failed to init DNN detector, falling back to Haar")
    if dnn_net is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        print("Using Haar cascade face detector")

    cap = cv2.VideoCapture(CAMERA)
    print("Webcam started, press 'q' to quit")

    img_size: Tuple[int, int] = IMG_SIZE

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
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, img_size)

            embedding = get_embedding(face_resized)
            if embedding is None:
                continue

            embedding = embedding.reshape(1, -1)
            pred = model.predict(embedding, verbose=0)[0]
            idx = int(np.argmax(pred))
            emotion = emotions[idx]
            confidence = float(pred[idx]) * 100.0

            color = COLORS.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{emotion}: {confidence:.1f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("DeepFace Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Stopped")


if __name__ == "__main__":
    main()