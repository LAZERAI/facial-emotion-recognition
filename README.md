# Emotion Recognition

Two approaches to detect facial emotions in real-time: DeepFace embedding-based classification and EfficientNetB0 transfer learning.

## What's included

- **`efficientnet/`** — EfficientNetB0 transfer learning implementation. Fast inference (30+ FPS), end-to-end trainable model.
- **`deepface/`** — DeepFace-based approach using VGG-Face embeddings. More robust to face angle, slightly slower.
- **`emotion__images/`** — Dataset directory (you provide images in emotion-labeled subfolders).
- **`requirements.txt`** — Python dependencies.
- **`deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`** — Pre-trained SSD face detector weights for OpenCV (optional; falls back to Haar if not available).

Both approaches use OpenCV for webcam inference and face detection.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.8+.

## Usage

### EfficientNet

Train a model:
```bash
python efficientnet/train_efficientnet.py
```

Run inference:
```bash
python efficientnet/run_efficientnet.py
```

### DeepFace

Train a model:
```bash
python deepface/train_deepface.py
```

Run inference:
```bash
python deepface/run_deepface.py
```

Both expect an `emotion__images/` directory with emotion subfolders (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise).

## Approaches

**EfficientNet**: Transfer learning with EfficientNetB0 (frozen base, trainable top layers). Real-time inference at 30+ FPS.

**DeepFace**: VGG-Face embeddings (2622-d) + lightweight dense classifier. More robust to angle variation but slightly slower.

## License

MIT