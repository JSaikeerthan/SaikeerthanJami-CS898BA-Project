# Plant Disease Detection Using Image Processing and Deep Learning

## CS898BA - Image Analysis and Computer Vision

**Student:** Saikeerthan Jami  
**Course:** CS898BA - Image Analysis and Computer Vision  
**Instructor:** Cody Farlow

---

# Project Overview

This project develops an automated plant disease detection system using image processing techniques and a Convolutional Neural Network (CNN). The goal is to accurately classify plant leaf diseases from images in the PlantVillage dataset.

The project combines classical image preprocessing with deep learning to improve feature extraction and classification performance.

---

# Problem Statement

Plant diseases significantly reduce agricultural productivity worldwide. Manual diagnosis requires expert knowledge and is time-consuming.

This project aims to build an intelligent image classification system capable of automatically identifying plant diseases from leaf images.

---

# Dataset

**Dataset:** PlantVillage

- Total Images: **54,305**
- Classes: **38**
- Image Size: **224 × 224**

Dataset structure:

```
PlantVillage/
└── color/
    ├── Apple___Apple_scab
    ├── Apple___healthy
    ├── Tomato___Late_blight
    ├── Tomato___healthy
    └── ...
```

---

# Project Structure

```
SaikeerthanJami-CS898BA-Project/

├── assets/
│   ├── pipeline/
│   ├── preprocessing/
│   └── results/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── figures/
│   ├── history/
│   ├── metadata/
│   ├── models/
│   └── reports/
│
├── src/
│   ├── evaluation/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   ├── dataset.py
│   └── config.py
│
├── tests/
│
├── README.md
├── AI_Log.md
├── requirements.txt
└── .gitignore
```

---

# Image Processing Pipeline

The preprocessing pipeline improves image quality before model training.

### 1. Image Resizing

- Resize all images to **224 × 224**

### 2. Gaussian Blur

- Removes high-frequency image noise

### 3. CLAHE

Contrast Limited Adaptive Histogram Equalization improves local image contrast while preserving details.

### 4. HSV Color Space Conversion

Converts RGB images into HSV representation for enhanced color analysis.

Pipeline:

```
Raw Image
      │
      ▼
Resize
      │
      ▼
Gaussian Blur
      │
      ▼
CLAHE
      │
      ▼
HSV Conversion
      │
      ▼
CNN
```

---

# CNN Architecture

The baseline model consists of:

- Rescaling Layer
- Conv2D
- MaxPooling2D
- Conv2D
- MaxPooling2D
- Conv2D
- MaxPooling2D
- Flatten
- Dense (128)
- Dropout
- Output Layer (38 Classes)

Input Shape:

```
224 × 224 × 3
```

Loss Function:

```
Sparse Categorical Crossentropy
```

Optimizer:

```
Adam
```

Evaluation Metric:

```
Accuracy
```

---

# Training Configuration

| Parameter | Value |
|-----------|------:|
| Epochs | 5 |
| Batch Size | 16 |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Image Size | 224×224 |
| Validation Split | 20% |

Callbacks:

- EarlyStopping
- ModelCheckpoint

---

# Evaluation

The trained model is evaluated using:

- Validation Accuracy
- Classification Report
- Confusion Matrix
- Sample Predictions

Generated outputs include:

```
outputs/

figures/
├── accuracy.png
├── loss.png
├── confusion_matrix.png
└── sample_predictions.png

reports/
└── classification_report.txt
```

---

# Results

| Metric | Value |
|---------|--------|
| Validation Accuracy | **90.57%** |
| Number of Classes | 38 |
| Total Images | 54,305 |

Example inference:

```
Ground Truth

Apple___healthy

Prediction

Apple___healthy

Confidence

98.15%
```

---

# Running the Project

## Clone Repository

```bash
git clone https://github.com/JSaikeerthan/SaikeerthanJami-CS898BA-Project.git

cd SaikeerthanJami-CS898BA-Project
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Model

```bash
python -m src.training.train
```

---

## Evaluate Model

```bash
python -m src.evaluation.evaluate
```

---

## Predict Disease

```bash
python -m src.evaluation.predict
```

---

## Generate Sample Predictions

```bash
python -m src.evaluation.sample_predictions
```

---

## Generate Training Curves

```bash
python -m src.evaluation.training_curves
```

---

# Technologies Used

- Python 3.13
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- tqdm
- Git
- GitHub
- VS Code

---

# Future Improvements

Future work will include:

- Transfer Learning (EfficientNetB0)
- MobileNetV2 comparison
- Data Augmentation
- Learning Rate Scheduling
- Grad-CAM Visualization
- Streamlit Web Application
- Docker Deployment
- Model Quantization

---

# Acknowledgements

- PlantVillage Dataset
- TensorFlow
- OpenCV
- Scikit-learn

---