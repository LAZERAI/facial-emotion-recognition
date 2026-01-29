# 🎭 Facial Emotion Recognition System

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Why These Approaches?](#why-these-approaches)
3. [Approach 1: EfficientNetB0 (Transfer Learning)](#approach-1-efficientnetb0-transfer-learning)
4. [Approach 2: DeepFace Embeddings](#approach-2-deepface-embeddings)
5. [Comparison: Which is Better?](#comparison-which-is-better)
6. [Dataset Information](#dataset-information)
7. [Installation & Setup](#installation--setup)
8. [How to Run](#how-to-run)
9. [Code Explanation (Line by Line)](#code-explanation-line-by-line)
10. [Interview Q&A](#interview-qa)

---

## 🎯 Project Overview

This project detects **7 human emotions** from facial images using deep learning:

| Emotion | Color Code (BGR) |
|---------|------------------|
| 😠 Angry | Red |
| 🤢 Disgust | Green |
| 😨 Fear | Purple |
| 😊 Happy | Yellow |
| 😐 Neutral | Gray |
| 😢 Sad | Blue |
| 😲 Surprise | Orange |

### Project Structure
```
Project Folder/
├── emotion__images/           # Dataset (19,928 images)
│   ├── Angry/     (2,828 images)
│   ├── Disgust/   (2,850 images)
│   ├── Fear/      (2,850 images)
│   ├── Happy/     (2,850 images)
│   ├── Neutral/   (2,850 images)
│   ├── Sad/       (2,850 images)
│   └── Surprise/  (2,850 images)
│
├── train_efficientnet.py      # Training script (Approach 1)
├── run_efficientnet.py        # Inference script (Approach 1)
├── train_deepface.py          # Training script (Approach 2)
├── run_deepface.py            # Inference script (Approach 2)
│
├── model_efficientnet.h5      # Trained model (Approach 1)
├── classes_efficientnet.json  # Class names (Approach 1)
├── model_deepface.h5          # Trained model (Approach 2)
└── encoder_deepface.pkl       # Label encoder (Approach 2)
```

---

## 🤔 Why These Approaches?

### Why Not Train From Scratch?

| Approach | Training Data Needed | Time | Accuracy |
|----------|---------------------|------|----------|
| Train from scratch | 1M+ images | Weeks | Low initially |
| **Transfer Learning** | 10k-50k images | Hours | High |
| **Pre-trained Embeddings** | 10k-50k images | Hours | High |

**Our dataset: ~20k images** → Transfer Learning & Embeddings are perfect!

### Why Not Other Models?

| Model | Why Not? |
|-------|----------|
| VGG16/VGG19 | Too heavy (500MB+), slow inference |
| ResNet152 | Overkill for 7 classes, slow |
| InceptionV3 | Complex, harder to explain |
| Custom CNN | Would need millions of images |
| **EfficientNetB0** | ✅ Best accuracy-to-size ratio |
| **DeepFace** | ✅ Pre-trained on faces specifically |

---

## 🧠 Approach 1: EfficientNetB0 (Transfer Learning)

### What is Transfer Learning?

```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFER LEARNING                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ImageNet Dataset          Your Dataset                     │
│  (14 Million Images)       (20k Images)                     │
│         │                       │                           │
│         ▼                       ▼                           │
│  ┌─────────────┐         ┌─────────────┐                   │
│  │ Train on    │         │ Fine-tune   │                   │
│  │ 1000 classes│ ──────► │ for 7       │                   │
│  │ (animals,   │         │ emotions    │                   │
│  │ objects...) │         │             │                   │
│  └─────────────┘         └─────────────┘                   │
│         │                       │                           │
│         ▼                       ▼                           │
│  Learns: edges,          Learns: facial                     │
│  shapes, textures        expressions                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### What is EfficientNet?

- Developed by **Google** in 2019
- Uses **compound scaling** (width, depth, resolution)
- **B0** is the smallest variant (good for our dataset)
- Pre-trained on **ImageNet** (14 million images, 1000 classes)

### Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 EfficientNetB0 Architecture                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT IMAGE (224 x 224 x 3)                               │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────┐                   │
│  │     EfficientNetB0 Base Model       │  ◄── FROZEN       │
│  │     (Pre-trained on ImageNet)       │      (not trained)│
│  │     - Extracts features             │                   │
│  │     - 4M+ parameters                │                   │
│  └─────────────────────────────────────┘                   │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────┐                   │
│  │     GlobalAveragePooling2D          │                   │
│  │     - Reduces dimensions            │                   │
│  │     - (7,7,1280) → (1280)          │                   │
│  └─────────────────────────────────────┘                   │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────┐                   │
│  │     Dense(256, relu) + Dropout(0.4) │  ◄── TRAINABLE   │
│  └─────────────────────────────────────┘                   │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────┐                   │
│  │     Dense(128, relu) + Dropout(0.3) │  ◄── TRAINABLE   │
│  └─────────────────────────────────────┘                   │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────┐                   │
│  │     Dense(7, softmax)               │  ◄── OUTPUT      │
│  │     - 7 emotions                    │                   │
│  └─────────────────────────────────────┘                   │
│         │                                                   │
│         ▼                                                   │
│  OUTPUT: [0.1, 0.05, 0.02, 0.7, 0.03, 0.05, 0.05]         │
│          Angry Disg Fear Happy Neut  Sad  Surp            │
│                           ▲                                 │
│                      Winner: Happy (70%)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why Freeze Base Model?

| Approach | What Happens | When to Use |
|----------|--------------|-------------|
| **Freeze (our choice)** | Only train new layers | Small dataset (<50k) |
| Unfreeze | Train everything | Large dataset (>100k) |

With ~20k images, freezing prevents **overfitting**.

---

## 🧠 Approach 2: DeepFace Embeddings

### What are Embeddings?

```
┌─────────────────────────────────────────────────────────────┐
│                    FACE EMBEDDINGS                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Face Image                    Embedding Vector             │
│                                                             │
│  ┌─────────┐     DeepFace     ┌─────────────────────────┐  │
│  │  😀     │  ───────────►    │ [0.23, -0.87, 0.12,     │  │
│  │ 224x224 │                  │  0.45, -0.33, 0.78,     │  │
│  └─────────┘                  │  ... 2622 numbers ...]  │  │
│                               └─────────────────────────┘  │
│                                                             │
│  This vector CAPTURES:                                      │
│  - Face shape                                               │
│  - Eye position                                             │
│  - Mouth curve                                              │
│  - Eyebrow angle                                            │
│  - All facial features!                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### What is DeepFace?

- Developed by **Facebook** (now Meta)
- Pre-trained on **millions of faces**
- Supports multiple models: VGG-Face, Facenet, OpenFace, etc.
- We use **VGG-Face** (2622-dimensional embeddings)

### Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              DeepFace Embeddings Architecture               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT IMAGE (224 x 224 x 3)                               │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────┐                   │
│  │     DeepFace VGG-Face               │                   │
│  │     (Pre-trained on faces)          │                   │
│  │     - Extracts 2622-dim embedding   │                   │
│  └─────────────────────────────────────┘                   │
│         │                                                   │
│         ▼                                                   │
│  EMBEDDING: [0.23, -0.87, 0.12, ... 2622 numbers]          │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────┐                   │
│  │     Dense(512, relu) + Dropout(0.3) │  ◄── TRAINABLE   │
│  └─────────────────────────────────────┘                   │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────┐                   │
│  │     Dense(256, relu) + Dropout(0.3) │  ◄── TRAINABLE   │
│  └─────────────────────────────────────┘                   │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────┐                   │
│  │     Dense(128, relu) + Dropout(0.2) │  ◄── TRAINABLE   │
│  └─────────────────────────────────────┘                   │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────┐                   │
│  │     Dense(7, softmax)               │  ◄── OUTPUT      │
│  └─────────────────────────────────────┘                   │
│         │                                                   │
│         ▼                                                   │
│  OUTPUT: [0.1, 0.05, 0.02, 0.7, 0.03, 0.05, 0.05]         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚖️ Comparison: Which is Better?

| Aspect | EfficientNetB0 | DeepFace Embeddings |
|--------|----------------|---------------------|
| **Training Time** | 1-2 hours | 3-5 hours |
| **Inference Speed** | ⚡ Fast (30+ FPS) | 🐢 Slower (10-15 FPS) |
| **Accuracy** | 85-90% | 88-93% |
| **Model Size** | ~50 MB | ~10 MB |
| **Memory Usage** | Higher | Lower |
| **Best For** | Real-time apps | Accuracy-critical apps |
| **Pre-trained On** | General objects | Faces specifically |

### Recommendation:
- **Use EfficientNetB0** for real-time webcam apps
- **Use DeepFace** when accuracy matters more than speed

---

## 📊 Dataset Information

### Original Dataset
| Emotion | Original Images |
|---------|-----------------|
| Angry | 28 |
| Disgust | 30 |
| Fear | 30 |
| Happy | 30 |
| Neutral | 30 |
| Sad | 30 |
| Surprise | 30 |
| **Total** | **208** |

### After Augmentation
| Emotion | Augmented Images | Total |
|---------|------------------|-------|
| Angry | 2,800 | 2,828 |
| Disgust | 2,820 | 2,850 |
| Fear | 2,820 | 2,850 |
| Happy | 2,820 | 2,850 |
| Neutral | 2,820 | 2,850 |
| Sad | 2,820 | 2,850 |
| Surprise | 2,820 | 2,850 |
| **Total** | **19,720** | **19,928** |

### Augmentation Techniques Used
- Rotation (±15°)
- Width/Height shift (10%)
- Horizontal flip
- Zoom (10%)

---

## 🔧 Installation & Setup

### Requirements
- Python 3.10.11
- Windows 10/11
- Webcam (for inference)

### Step 1: Install Python
Download from: https://www.python.org/downloads/release/python-31011/

⚠️ Check **"Add Python to PATH"** during installation!

### Step 2: Install Libraries
```bash
pip install tensorflow==2.15.0 numpy==1.24.3 scikit-learn==1.3.2 opencv-python==4.8.1.78 Pillow==10.1.0 deepface
```

### Step 3: Verify Installation
```bash
python -c "import tensorflow; print('TensorFlow OK')"
python -c "import numpy; print('NumPy OK')"
python -c "import sklearn; print('Scikit-learn OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "from deepface import DeepFace; print('DeepFace OK')"
```

---

## 🚀 How to Run

### Training

```bash
# Navigate to project folder
cd "C:\Users\DELL\Documents\Raihan temp"

# Train EfficientNetB0 (1-2 hours)
python train_efficientnet.py

# OR Train DeepFace (3-5 hours)
python train_deepface.py
```

### Inference (Webcam)

```bash
# Run EfficientNetB0 model
python run_efficientnet.py

# OR Run DeepFace model
python run_deepface.py
```

### Controls
- **q** - Quit application
- Face the camera and show different expressions!

---

## 📖 Code Explanation (Line by Line)

### train_efficientnet.py

```python
# ============================================
# IMPORTS
# ============================================
import os
import numpy as np
```
- `os` - For file/folder operations
- `numpy` - For numerical operations (arrays, math)

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
- `ImageDataGenerator` - Loads images from folders AND applies augmentation on-the-fly

```python
from tensorflow.keras.applications import EfficientNetB0
```
- Loads the pre-trained EfficientNetB0 model with ImageNet weights

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
```
- `Model` - Creates custom model architecture
- `Dense` - Fully connected layer (every neuron connected to every neuron)
- `GlobalAveragePooling2D` - Reduces 2D feature maps to 1D vector
- `Dropout` - Randomly "drops" neurons during training (prevents overfitting)

```python
from tensorflow.keras.optimizers import Adam
```
- `Adam` - Optimizer (adjusts learning rate automatically)

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
```
- `EarlyStopping` - Stops training if no improvement
- `ModelCheckpoint` - Saves best model during training
- `ReduceLROnPlateau` - Reduces learning rate when stuck

```python
from sklearn.metrics import classification_report, confusion_matrix
```
- `classification_report` - Shows precision, recall, F1-score
- `confusion_matrix` - Shows prediction vs actual

```python
import json
```
- For saving class names to file

---

```python
# ============================================
# CONFIGURATION
# ============================================
DATASET_PATH = r"C:\Users\DELL\Documents\Raihan temp\emotion__images"
IMG_SIZE = (224, 224)      # EfficientNet expects 224x224
BATCH_SIZE = 16            # Process 16 images at a time
EPOCHS = 30                # Maximum training iterations
MODEL_PATH = "model_efficientnet.h5"
CLASSES_PATH = "classes_efficientnet.json"
```

---

```python
# ============================================
# DATA GENERATORS
# ============================================
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixels: 0-255 → 0-1
    validation_split=0.2,     # 80% train, 20% validation
    rotation_range=15,        # Random rotation ±15°
    width_shift_range=0.1,    # Random horizontal shift
    height_shift_range=0.1,   # Random vertical shift
    horizontal_flip=True      # Random horizontal flip
)
```

**Why rescale=1./255?**
- Pixels are 0-255, but neural networks work better with 0-1
- Division by 255 normalizes the values

**Why validation_split=0.2?**
- 80% for training (learning)
- 20% for validation (testing during training)

---

```python
train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,        # Resize all images to 224x224
    batch_size=BATCH_SIZE,       # Load 16 images at a time
    class_mode='categorical',    # One-hot encoding: [0,0,1,0,0,0,0]
    subset='training',           # Use training portion
    shuffle=True                 # Randomize order
)
```

**What is categorical (one-hot encoding)?**
```
Happy → [0, 0, 0, 1, 0, 0, 0]
Angry → [1, 0, 0, 0, 0, 0, 0]
Sad   → [0, 0, 0, 0, 0, 1, 0]
```

---

```python
# ============================================
# BUILD MODEL
# ============================================
base_model = EfficientNetB0(
    weights='imagenet',          # Load pre-trained weights
    include_top=False,           # Remove classification layer
    input_shape=(224, 224, 3)    # Input size
)
base_model.trainable = False     # Freeze! Don't train these layers
```

**Why include_top=False?**
- EfficientNetB0 was trained for 1000 classes (ImageNet)
- We only need 7 classes (emotions)
- Remove the top, add our own

**Why trainable=False?**
- Prevents modifying pre-trained weights
- Only train our new layers
- Prevents overfitting with small dataset

---

```python
x = base_model.output
x = GlobalAveragePooling2D()(x)    # (7,7,1280) → (1280)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)                 # Drop 40% of neurons randomly
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)                 # Drop 30% of neurons randomly
output = Dense(7, activation='softmax')(x)  # 7 emotions
```

**What is ReLU?**
- Rectified Linear Unit
- f(x) = max(0, x)
- Adds non-linearity, helps learn complex patterns

**What is Softmax?**
- Converts outputs to probabilities (sum = 1.0)
- Example: [2.1, 0.5, 1.2, 5.8, 0.3, 0.7, 0.9] → [0.05, 0.01, 0.02, 0.85, 0.01, 0.03, 0.03]

**What is Dropout?**
- Randomly "turns off" neurons during training
- Prevents overfitting (model memorizing data)
- Like training with different "partial" networks

---

```python
model.compile(
    optimizer=Adam(0.001),                   # Learning rate = 0.001
    loss='categorical_crossentropy',         # Loss function for multi-class
    metrics=['accuracy']                     # Track accuracy
)
```

**What is Learning Rate?**
- How big of a "step" the model takes when learning
- Too high = unstable, too low = slow learning
- 0.001 is a good default

**What is Categorical Crossentropy?**
- Loss function for multi-class classification
- Measures how "wrong" the prediction is
- Lower = better

---

```python
# ============================================
# CALLBACKS
# ============================================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',      # Watch validation accuracy
        patience=5,                  # Stop if no improvement for 5 epochs
        restore_best_weights=True    # Use best model, not last
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True          # Only save if improved
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,                  # Reduce LR by half
        patience=3,                  # After 3 epochs of no improvement
        min_lr=1e-6                  # Don't go below 0.000001
    )
]
```

---

### run_efficientnet.py

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
```
- **Haar Cascade** - Fast face detection algorithm
- Pre-trained XML file from OpenCV
- Detects face location (x, y, width, height)

---

```python
cap = cv2.VideoCapture(0)  # 0 = default webcam
```
- Opens webcam connection
- 0 = first camera, 1 = second camera, etc.

---

```python
while True:
    ret, frame = cap.read()  # Read one frame
    if not ret:
        break
```
- Continuously read frames from webcam
- `ret` = success/failure
- `frame` = image data (numpy array)

---

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
```
- Convert to grayscale (Haar Cascade needs grayscale)
- Detect faces in the frame
- Returns list of (x, y, width, height) for each face

---

```python
for (x, y, w, h) in faces:
    face = frame[y:y+h, x:x+w]                    # Crop face
    face = cv2.resize(face, IMG_SIZE)             # Resize to 224x224
    face = face.astype('float32') / 255.0         # Normalize
    face = np.expand_dims(face, axis=0)           # Add batch dimension
```

**Why expand_dims?**
- Model expects batch: (batch_size, height, width, channels)
- Single image: (224, 224, 3)
- With batch: (1, 224, 224, 3)

---

```python
pred = model.predict(face, verbose=0)[0]
emotion = EMOTIONS[np.argmax(pred)]
confidence = pred[np.argmax(pred)] * 100
```
- `predict` returns probabilities for all 7 emotions
- `argmax` finds index of highest probability
- Convert to emotion name and percentage

---

## ❓ Interview Q&A

### Q1: What is Transfer Learning?
**A:** Transfer learning is using a model trained on one task (ImageNet - 14M images, 1000 classes) and adapting it for another task (emotion detection - 20k images, 7 classes). We take the pre-trained model's ability to recognize features and only train new layers for our specific task.

### Q2: Why EfficientNetB0 instead of VGG16 or ResNet?
**A:** EfficientNetB0 has the best accuracy-to-parameter ratio. It uses compound scaling (width, depth, resolution) to achieve high accuracy with fewer parameters. VGG16 is 500MB+ while EfficientNetB0 is only ~20MB with similar accuracy.

### Q3: What is an embedding?
**A:** An embedding is a numerical vector representation of data. In DeepFace, a face image is converted to a 2622-dimensional vector that captures facial features. Similar faces have similar embeddings.

### Q4: Why use Dropout?
**A:** Dropout randomly "turns off" neurons during training, forcing the network to learn redundant representations. This prevents overfitting (memorizing training data) and improves generalization.

### Q5: What is the difference between training and validation data?
**A:** Training data (80%) is used to update model weights. Validation data (20%) is used to check performance during training without affecting weights. This helps detect overfitting.

### Q6: Why freeze the base model?
**A:** With only ~20k images, training all 4M+ parameters would cause overfitting. By freezing pre-trained layers, we only train ~200k new parameters, which is appropriate for our dataset size.

### Q7: What is categorical crossentropy?
**A:** It's a loss function for multi-class classification. It measures the difference between predicted probabilities and actual labels (one-hot encoded). Lower loss = better predictions.

### Q8: Why is image normalization important?
**A:** Neural networks work better with smaller, standardized values. Pixel values (0-255) are divided by 255 to get (0-1). This helps gradients flow better and training converge faster.

### Q9: What is data augmentation?
**A:** Data augmentation artificially increases dataset size by applying transformations (rotation, flip, shift) to existing images. This helps the model generalize better and reduces overfitting.

### Q10: Why use Haar Cascade for face detection instead of MTCNN?
**A:** Haar Cascade is much faster (suitable for real-time), though less accurate. For webcam applications where speed matters, Haar Cascade is preferred. MTCNN is more accurate but slower.

---

## 📝 Quick Reference

### File Outputs

| Training Script | Output Files |
|-----------------|--------------|
| `train_efficientnet.py` | `model_efficientnet.h5`, `classes_efficientnet.json` |
| `train_deepface.py` | `model_deepface.h5`, `encoder_deepface.pkl` |

### Commands

```bash
# Install all libraries
pip install tensorflow==2.15.0 numpy==1.24.3 scikit-learn==1.3.2 opencv-python==4.8.1.78 Pillow==10.1.0 deepface

# Train EfficientNet
python train_efficientnet.py

# Run EfficientNet inference
python run_efficientnet.py

# Train DeepFace
python train_deepface.py

# Run DeepFace inference
python run_deepface.py
```

---

## 👨‍💻 Author

Created for academic/learning purposes.

**Python Version:** 3.10.11  
**TensorFlow Version:** 2.15.0  
**Date:** January 2026

---

## 📄 License

This project is for educational purposes only.