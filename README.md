# Emotion-Video — Project Documentation

> Purpose: Emotion recognition experiments using two approaches:
> - Embedding-based classifier built on DeepFace (VGG-Face embeddings + small dense classifier)
> - Transfer learning using EfficientNetB0 (end‑to‑end CNN classifier)

---

## Table of contents
1. Overview
2. File map (what each file does)
3. Environment & required packages (with versions found on your machine)
4. Models & assets (what each file is for)
5. Data format and expected dataset structure
6. How to run (quick commands)
7. Training: step-by-step (DeepFace & EfficientNet)
8. Inference: step-by-step (webcam scripts)
9. Tips for performance and debugging
10. Troubleshooting & common errors
11. Notes & references

---

## 1. Overview
This repository provides two ways to build and run emotion recognition:
- `train_deepface.py` / `run_deepface.py`: Extract face embeddings using DeepFace (VGG-Face) and train a compact dense classifier on these embeddings. Inference extracts embeddings per detected face and classifies them with the trained dense model.
- `train_efficientnet.py` / `run_efficientnet.py`: Train a classifier on top of EfficientNetB0 using transfer learning. Inference runs EfficientNet on each detected face.

Both inference scripts use OpenCV face detection. They prefer the DNN SSD detector (a Caffe model provided here) when available, and fall back to the Haar cascade if not.

---

## 2. File map
- `run_deepface.py` — Webcam inference using DeepFace embeddings + dense classifier (expects `model_deepface.h5` + `encoder_deepface.pkl`).
- `train_deepface.py` — Build embeddings from images, train a dense classifier, save `model_deepface.h5` and `encoder_deepface.pkl`.
- `run_efficientnet.py` — Webcam inference using EfficientNetB0 (expects `model_efficientnet.h5` + `classes_efficientnet.json`).
- `train_efficientnet.py` — Train EfficientNetB0 transfer learning model and save `model_efficientnet.h5` and `classes_efficientnet.json`.
- `emotion__images/` — Your dataset directory (should contain subfolders per emotion label).
- `deploy.prototxt` — Caffe model definition for the SSD face detector (optional but recommended when paired with the model below).
- `res10_300x300_ssd_iter_140000.caffemodel` — Pretrained SSD face detector weights (recommended for DNN-based detection).

---

## 3. Environment & required packages
Packages used (based on your environment):
- tensorflow 2.20.0
- deepface 0.0.98
- opencv-python 4.13.0.90
- numpy 2.4.1
- scikit-learn 1.8.0
- h5py 3.15.1

Recommended install command:

```powershell
pip install tensorflow deepface opencv-python numpy scikit-learn h5py Pillow matplotlib tqdm
```

Notes:
- `deepface` will auto-download face models (VGG-Face, etc.) when needed.
- For systems without GPU or when disk space is limited, you can use `tensorflow-cpu` instead of full `tensorflow`.

---

## 4. Models & assets (what they are for)
- `model_efficientnet.h5` — Keras model file (EfficientNetB0 + classifier head). Used by `run_efficientnet.py`.
- `classes_efficientnet.json` — JSON list mapping class indices to label names used by EfficientNet inference.
- `model_deepface.h5` — Small dense classifier trained on DeepFace embeddings. Used by `run_deepface.py`.
- `encoder_deepface.pkl` — `LabelEncoder` saved with classes in training order. Used to map predictions to labels for DeepFace pipeline.
- `res10_300x300_ssd_iter_140000.caffemodel` + `deploy.prototxt` — Face detection DNN; both files together are required for the SSD DNN face detector.


**Note:** This repository does **not** include any pre-trained model files (`.h5`, `.pkl`, `.json`) or image data. You must:

- Provide your own dataset in the `emotion__images/` folder (see structure below).
- Train your own models using the provided training scripts.
- The face detector files (`deploy.prototxt`, `.caffemodel`) are also not included; download from the official OpenCV repository if needed.

If you want to use pre-trained models, you must train and export them yourself using your own data.

---

## 5. Data format and expected dataset structure
- The dataset folder is `emotion__images`.
- Expect this layout:

```
emotion__images/
  Angry/
    img1.jpg
    img2.jpg
  Happy/
    img1.jpg
    ...
  Neutral/
  ...
```

- Each folder name will become a class label. Filenames can be anything; supported image formats are those OpenCV can read (jpg, png, etc.).

---

## 6. How to run (quick commands)
Open PowerShell in the project folder and run:

- Run EfficientNet webcam inference (uses local files by default):
  python run_efficientnet.py

- Run DeepFace webcam inference:
  python run_deepface.py

- Train EfficientNet (will generate `model_efficientnet.h5` and `classes_efficientnet.json`):
  python train_efficientnet.py

- Train DeepFace embedding classifier (will generate `model_deepface.h5` and `encoder_deepface.pkl`):
  python train_deepface.py

If you prefer, you can modify the scripts to change paths; they are currently simplified to use the project-local fixed paths.

---

## 7. Training: step-by-step details
### train_deepface.py
1. Walk `emotion__images` and for each labelled image:
   - Read with OpenCV, convert to RGB and resize to 224x224.
   - Compute a 2622-d embedding using DeepFace `VGG-Face` (via `DeepFace.represent`).
2. Collect embeddings and labels; encode labels with `LabelEncoder` and save it (`encoder_deepface.pkl`).
3. Train a small dense classifier (architecture: 512 -> 256 -> 128 -> softmax, with dropout) with `sparse_categorical_crossentropy` loss.
4. Save trained weights to `model_deepface.h5`.

Notes:
- Embedding extraction is relatively slow (DeepFace internal models may run on CPU unless TensorFlow/GPU configured). Consider running on a machine with GPU for speed.
- If many images fail embedding extraction, check images for face visibility or try enlarging/enhancing input images.

### train_efficientnet.py
1. Uses `ImageDataGenerator` with basic augmentation and validation split (20%).
2. Loads EfficientNetB0 pretrained on ImageNet (top removed) and attaches GlobalAveragePooling + dense layers.
3. Trains classifier head first (base frozen). Optionally you can unfreeze and finetune the base for higher accuracy (code originally had an opt-in `--finetune`, but scripts here are simplified to a safe default).
4. Saves `model_efficientnet.h5` and `classes_efficientnet.json`.

Notes:
- EfficientNetB0 with frozen base typically gives fast convergence; fine-tuning the base requires smaller learning rates and more care.

---

## 8. Inference: step-by-step details
### Face detection options
- DNN SSD detector (recommended): uses the included `deploy.prototxt` and `.caffemodel` files. Better accuracy and robustness than Haar.
- Haar cascade (fallback): built into OpenCV and works without extra files.

Scripts will attempt to use the DNN detector when both files are present; otherwise they use Haar.

### run_efficientnet.py
- Read frames from webcam, detect faces, crop and resize to 224x224, normalize to [0,1], run the model, display predictions and confidence overlayed on the video.

### run_deepface.py
- Detect faces, convert face to RGB/224x224, get DeepFace embedding (VGG-Face), run the dense classifier on the embedding, and display predicted label + confidence.

---

## 9. Tips for performance and debugging
- GPU: If you have an NVIDIA GPU and want to train faster, install a matched CUDA/cuDNN version that is compatible with TensorFlow 2.20. Check the TensorFlow compatibility matrix.
- Memory: Reduce batch size if training runs out of memory.
- Dataset balance: Use stratified splits (already used in training) and consider augmenting underrepresented classes.
- Quick debug run: reduce `EPOCHS` to 1 and train on a subset to ensure code runs end-to-end.

---

## 10. Troubleshooting & common errors
- "ModuleNotFoundError": install missing package(s) listed in section 3.
- DeepFace errors during embedding extraction: check that faces are visible and clear; consider increasing image size or using the DNN detector to get better face crops.
- Missing model files: if `model_deepface.h5` or `model_efficientnet.h5` not present, run the corresponding training script.
- DNN detector fails to load: ensure both `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` are present and not corrupted. If you prefer not to use DNN, Haar cascade will be used automatically.

Useful commands to inspect files:
```powershell
Get-ChildItem -Path "C:\Users\Lazerai\Downloads\Emotion-Video" -Recurse -Include *.h5,*.pkl,*.json,*.prototxt,*.caffemodel | Format-Table FullName, Length
```

---

## 11. Notes & references
- DeepFace: https://github.com/serengil/deepface
- SSD Face detector reference (OpenCV DNN): typical pair `deploy.prototxt` + `res10_300x300_ssd_iter_140000.caffemodel` is widely used for face detection.
- EfficientNet paper: Tan & Le, 2019.

---

If you want, I can also:
- Add a `requirements.txt` and a quick PowerShell install snippet, or
- Add a short smoke-test script that checks imports and model file presence and prints a single-line summary.

---

## Appendix — Deep, practical details (expanded)
This appendix contains deeper explanations, step-by-step operational guidance, and practical tips for real projects. It is intentionally verbose so you can refer to it when needed.

### A. Environment and reproducibility
- Use a virtual environment to avoid package conflicts:
  - PowerShell (Windows):
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```
  - macOS/Linux:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
- Pin package versions in `requirements.txt` (recommended sample is provided below). If you want, I can generate a `requirements.txt` that matches your current environment.

### B. `requirements.txt` (recommended pins)
A minimal pinned file to reproduce the environment used when this README was created:

```
tensorflow==2.20.0
deepface==0.0.98
opencv-python==4.13.0.90
numpy==2.4.1
scikit-learn==1.8.0
h5py==3.15.1
Pillow
matplotlib
tqdm
```

- Use `pip freeze > requirements.txt` on your machine to capture exact installed versions for full reproducibility.

### C. The face detector files (what they are and how to use them)
- `deploy.prototxt` is the network definition (architecture) for the SSD face detector.
- `res10_300x300_ssd_iter_140000.caffemodel` are the pre-trained weights for that network.
- Both files are used together with OpenCV's dnn module: `cv2.dnn.readNetFromCaffe(prototxt, model)`.
- If either file is missing, the scripts will fall back to OpenCV's Haar cascade (`haarcascade_frontalface_default.xml`). That is slower and less robust but requires no extra files.

Why prefer DNN SSD detector?
- Better accuracy on real-world images (more robust to pose/lighting).
- Faster on modern CPUs and widely used in production for simple pipelines.

### D. DeepFace & embeddings (practical notes)
- DeepFace provides several face models (VGG-Face, Facenet, ArcFace). This project uses VGG-Face embeddings by default (result is a 2622-dimensional vector).
- Embedding extraction may be slower on CPU. For real-time, try:
  - Use lower-resolution crops (but keep enough pixels for face detail).
  - Use a lightweight embedding model (Facenet/LightFace) if lower latency is required — but this will require modifying `get_embedding()` in the code.
- DeepFace will automatically download required models to a cache folder (~/.deepface by default). Ensure the machine has internet on first run.

### E. Data curation and labels
- Folder names become labels. Keep them concise and standardized (e.g., `Angry`, `Neutral`, `Happy`, `Sad`, `Surprise`, `Fear`, `Disgust`).
- Typical pitfalls:
  - Mixed/incorrect labels (manual spot-checking helps).
  - Imbalanced classes — prefer balanced datasets or use augmentations for underrepresented classes.
  - Image quality: blurred, occluded, or tiny faces reduce embedding quality. Remove or filter extremely small face crops.

Practical steps to prepare a dataset:
1. Ensure each class folder contains only that label's images.
2. Remove duplicates and corrupt images.
3. Run a quick script to detect faces and discard images with no reliable face detection.
4. Use augmentation (rotation, flips, color jitter) for small classes.

### F. Training recommendations
- Train EfficientNet in two stages:
  1. Train only the head layers (base frozen) until validation loss stabilizes.
  2. Unfreeze some top base layers and continue training with a low learning rate (e.g., reduce by factor 10).
- EarlyStopping is used in the scripts — patience = 5 is a reasonable default.
- Use ReduceLROnPlateau if loss plateaus.
- Use `ModelCheckpoint(save_best_only=True)` to ensure the best weights are saved.

Hyperparameters to try:
- Learning rate: start 0.001 for classifier head, 1e-4 or lower when fine-tuning base.
- Batch size: depends on GPU memory; 8–32 is typical for 224x224 images.
- Augmentation: rotation up to ±15°, width/height shifts up to 0.1, horizontal flip for non-directional emotions.

### G. Inference considerations
- Confidence calibration: raw softmax scores are not calibrated probabilities; for critical applications, consider temperature scaling or Platt scaling.
- Smoothing: for webcam, stabilize predicted label by averaging predictions over the last N frames (e.g., N=5) to avoid flicker.
- Face alignment: cropping a face and aligning eyes/landmarks improves embedding robustness — the current scripts do center-crop/resizing only.

### H. Performance & hardware
- CPU-only inference: EfficientNetB0 is reasonably fast on a modern desktop CPU; DeepFace embedding extraction is slower.
- GPU: TensorFlow with GPU dramatically speeds up training and embedding extraction. Ensure correct CUDA/CuDNN versions for TF 2.20.
- For deployment on edge devices, consider converting models to TensorFlow Lite or using a smaller backbone (MobileNet, EfficientNet-lite).

### I. Debugging tips (practical logs to add)
- Add a frame-skip counter and log if no faces detected for many frames.
- Save a sample misclassified crop for manual inspection (helps diagnose label noise or difficult samples).
- If DeepFace embedding fails frequently, add try/except to save the problematic image and continue.

### J. Example troubleshooting scenarios
- "I get no faces detected": check that camera is enabled; test `opencv` sample code; test DNN detection on a saved image.
- "Model file not found": ensure you trained and saved models or copied pre-trained models to this folder.
- "Slow inference": try lowering image size, swap to Haar (if DNN is slow because of CPU), or use GPU.

---

### Want this expanded further?
I can:
- Add runnable snippets for each troubleshooting case (e.g., `test_detector.py`, `save_bad_crop.py`).
- Generate `requirements.txt` pinned to your current environment.
- Add a short smoke-test (`scripts/smoke_check.py`) to validate imports and file presence automatically.

If you say "do it", I will add the requested artifact and keep the README updated accordingly.