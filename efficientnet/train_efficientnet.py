"""train_efficientnet.py
Training script using EfficientNetB0 (transfer learning).
The script supports basic augmentation, optional fine-tuning, and
saves the best model and the class index mapping.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logger = logging.getLogger(__name__)


def build_generators(dataset: str, img_size: tuple, batch_size: int):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    train_gen = datagen.flow_from_directory(dataset, target_size=img_size, batch_size=batch_size, class_mode="categorical", subset="training", shuffle=True)
    val_gen = datagen.flow_from_directory(dataset, target_size=img_size, batch_size=batch_size, class_mode="categorical", subset="validation", shuffle=False)
    return train_gen, val_gen


def build_model(n_classes: int, input_shape=(224, 224, 3), base_trainable: bool = False) -> Model:
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = base_trainable

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    # Simple fixed configuration (uses your project dataset folder)
    DATASET_PATH = r"C:\Users\Lazerai\Downloads\Emotion-Video\emotion__images"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16
    EPOCHS = 30
    MODEL_PATH = r"C:\Users\Lazerai\Downloads\Emotion-Video\model_efficientnet.h5"
    CLASSES_PATH = r"C:\Users\Lazerai\Downloads\Emotion-Video\classes_efficientnet.json"

    if not os.path.isdir(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        return

    print("Loading dataset from", DATASET_PATH)

    train_gen, val_gen = build_generators(DATASET_PATH, IMG_SIZE, BATCH_SIZE)
    class_names: List[str] = list(train_gen.class_indices.keys())
    with open(CLASSES_PATH, "w") as f:
        json.dump(class_names, f)

    print("Classes:", class_names)
    print("Training samples:", train_gen.samples, "| Validation samples:", val_gen.samples)

    model = build_model(len(class_names), input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), base_trainable=False)
    print("Model parameters:", model.count_params())

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=callbacks, verbose=1)

    print("Evaluating on validation set")
    val_gen.reset()
    y_pred = np.argmax(model.predict(val_gen, verbose=1), axis=1)
    y_true = val_gen.classes
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    accuracy = (cm.trace() / cm.sum()) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved class mapping to {CLASSES_PATH}")


if __name__ == "__main__":
    main()