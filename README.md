# Emotion Recognition

Real-time facial emotion detection. Two different ways to do it: **EfficientNetB0** (fast) or **DeepFace embeddings** (more accurate). Pick whichever fits what you're trying to do.

## What you get

- Two emotion recognition models to compare
- Real-time webcam stuff with OpenCV
- EfficientNetB0 runs at 30+ FPS, so it's actually usable
- DeepFace is slower but handles face angles better
- You train it on your own data (no pre-trained models included)
- Face detection built in, uses OpenCV DNN but falls back to Haar if needed

## Stack

- Python 3.8+
- TensorFlow / Keras
- DeepFace
- OpenCV
- NumPy, scikit-learn

## Folder layout

```
efficientnet/        # Fast transfer learning approach
  ├── train_efficientnet.py
  └── run_efficientnet.py

deepface/            # Embedding-based approach (more accurate)
  ├── train_deepface.py
  └── run_deepface.py

emotion__images/     # Put your dataset here
requirements.txt
LICENSE
```

## How it works

### EfficientNet way
1. Load EfficientNetB0 from ImageNet
2. Add a trainable head on top (keep base frozen)
3. Train it on your emotion images
4. Run inference: detect face → crop → predict (pretty fast)

### DeepFace way
1. Extract VGG-Face embeddings (2622-d vectors) from each image
2. Train a small dense classifier on those embeddings
3. Run inference: detect face → get embedding → classify

## Getting started

### Install

```bash
pip install -r requirements.txt
```

### Train a model

```bash
# Fast approach
python efficientnet/train_efficientnet.py

# Accurate approach
python deepface/train_deepface.py
```

### Run it

```bash
# EfficientNet (faster)
python efficientnet/run_efficientnet.py

# DeepFace (better accuracy)
python deepface/run_deepface.py
```

Just make sure you have `emotion__images/` with folders like `Angry/`, `Happy/`, `Sad/`, etc.

## EfficientNet vs DeepFace

| | EfficientNet | DeepFace |
| --- | --- | --- |
| Speed | 30+ FPS | 10-15 FPS |
| Accuracy | Good | Better |
| Face angles | Okay | Much better |
| Training | Fast | Slower (embeddings take time) |
| Model size | ~20MB | ~10MB |

**TL;DR** — Use EfficientNet if you need it to run fast. Use DeepFace if accuracy matters more.

## What's in here

- `efficientnet/` — EfficientNetB0 stuff
- `deepface/` — DeepFace stuff  
- Face detector files — `deploy.prototxt` and the caffemodel (optional, has fallback)
- `requirements.txt` — dependencies
- `smoke_check.py` — quick check if everything's installed

## License

MIT