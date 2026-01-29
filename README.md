# Emotion Recognition

Real-time facial emotion detection using two parallel approaches: **EfficientNetB0** (fast transfer learning) and **DeepFace** (robust embeddings). Choose based on your speed/accuracy needs.

## Features

- **Two emotion recognition models** — compare approaches within one codebase
- **Real-time webcam inference** — detect emotions live with OpenCV
- **Fast (30+ FPS)** — EfficientNetB0 optimized for speed
- **Robust to pose variation** — DeepFace VGG-Face embeddings handle angles well
- **No pre-trained models** — train from your own data
- **Face detection built-in** — uses OpenCV DNN (SSD) with Haar fallback

## Tech Stack

- Python 3.8+
- TensorFlow / Keras
- DeepFace
- OpenCV
- NumPy, scikit-learn

## Project Structure

```
efficientnet/        # Transfer learning approach
  ├── train_efficientnet.py
  └── run_efficientnet.py

deepface/            # Embedding-based approach
  ├── train_deepface.py
  └── run_deepface.py

emotion__images/     # Your dataset (you provide)
requirements.txt
LICENSE
```

## How It Works

### EfficientNet Approach
1. Load pre-trained EfficientNetB0 (ImageNet weights)
2. Attach trainable classifier head (frozen base)
3. Train on your emotion dataset
4. Inference: detect face → crop → predict emotion in real-time (30+ FPS)

### DeepFace Approach
1. Extract 2622-d VGG-Face embeddings from each training image
2. Train lightweight dense classifier on embeddings
3. Inference: detect face → extract embedding → classify emotion (10-15 FPS)

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Train

Choose your approach:

```bash
# EfficientNet
python efficientnet/train_efficientnet.py

# DeepFace
python deepface/train_deepface.py
```

### Inference

```bash
# EfficientNet (faster)
python efficientnet/run_efficientnet.py

# DeepFace (more robust)
python deepface/run_deepface.py
```

Both expect `emotion__images/` with subfolders: `Angry/`, `Disgust/`, `Fear/`, `Happy/`, `Neutral/`, `Sad/`, `Surprise/`.

## Which Approach?

| Metric | EfficientNet | DeepFace |
|--------|--------------|----------|
| Speed | 30+ FPS | 10-15 FPS |
| Robustness | Good | Excellent |
| Angle handling | Moderate | Strong |
| Training time | Fast | Slower (embedding extraction) |
| Model size | ~20MB | ~10MB |

Use **EfficientNet** for real-time performance. Use **DeepFace** if accuracy matters more than speed.

## What's Included

- **`efficientnet/`** — EfficientNetB0 implementation
- **`deepface/`** — DeepFace + dense classifier
- **Face detection files** — `deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel` (optional; Haar cascade fallback if missing)
- **`requirements.txt`** — All dependencies with versions
- **`smoke_check.py`** — Quick environment validation

## License

MIT