"""train_deepface.py
Train a small dense classifier on DeepFace embeddings (VGG-Face).
This script extracts embeddings, trains a classifier, and saves the model
and label encoder.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Reduce TensorFlow and DeepFace verbosity
os.environ["DEEPFACE_LOG_LEVEL"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


logger = logging.getLogger(__name__)


def get_embedding(face_img: np.ndarray) -> Optional[np.ndarray]:
    """Return a 2622-d embedding for a face or None on failure."""
    try:
        result = DeepFace.represent(face_img, model_name="VGG-Face", enforce_detection=False)
        return np.array(result[0]["embedding"])
    except Exception:
        logger.exception("Failed to compute embedding for an image.")
        return None


def build_dataset(dataset_path: str, img_size: Tuple[int, int]) -> Tuple[np.ndarray, List[str]]:
    """Walk the dataset directory and return (X, y).

    Expects a structure like dataset_path/<label>/*.jpg
    """
    X: List[np.ndarray] = []
    y: List[str] = []

    emotions = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    logger.info("Found classes: %s", emotions)

    for emotion in emotions:
        emotion_path = os.path.join(dataset_path, emotion)
        images = os.listdir(emotion_path)
        logger.info("Processing %s (%d files)", emotion, len(images))

        extracted = 0
        for i, img_name in enumerate(images):
            img_path = os.path.join(emotion_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, img_size)
            embedding = get_embedding(img_resized)
            if embedding is not None:
                X.append(embedding)
                y.append(emotion)
                extracted += 1

            if (i + 1) % 100 == 0:
                logger.info("Processed %d/%d files in %s", i + 1, len(images), emotion)

        logger.info("Extracted %d embeddings for class %s", extracted, emotion)

    logger.info("Total samples: %d", len(X))
    return np.array(X), y


def build_classifier(input_dim: int, n_classes: int) -> Sequential:
    model = Sequential([
        Dense(512, activation="relu", input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    # Simple fixed configuration
    DATASET_PATH = r"C:\Users\Lazerai\Downloads\Emotion-Video\emotion__images"
    IMG_SIZE = (224, 224)
    EPOCHS = 40
    BATCH_SIZE = 32
    MODEL_PATH = r"C:\Users\Lazerai\Downloads\Emotion-Video\model_deepface.h5"
    ENCODER_PATH = r"C:\Users\Lazerai\Downloads\Emotion-Video\encoder_deepface.pkl"

    if not os.path.isdir(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        return

    print("Starting training using dataset:", DATASET_PATH)

    X, y = build_dataset(DATASET_PATH, IMG_SIZE)
    if len(X) == 0:
        print("No embeddings were extracted. Check dataset path and images.")
        return

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    n_classes = len(encoder.classes_)
    print("Classes:", list(encoder.classes_))

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    print("Train samples:", len(X_train), "| Test samples:", len(X_test))

    model = build_classifier(X.shape[1], n_classes)

    callbacks = [EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1)]

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.15, callbacks=callbacks, verbose=1)

    print("Evaluating on test set...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm.trace() / cm.sum()) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Encoder saved to {ENCODER_PATH}")


if __name__ == "__main__":
    main()