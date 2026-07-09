# SaikeerthanJami-CS898BA-Project
# Plant Leaf Disease Detection Using Image Processing and Deep Learning
## Course

CS 898BA – Image Analysis and Computer Vision

## Student

Saikeerthan Jami(Q459V832)

## Project Description

This project develops a computer vision system capable of detecting plant leaf diseases using image preprocessing techniques and deep learning.

The project combines:

- OpenCV for image preprocessing
- TensorFlow/Keras for deep learning
- EfficientNet transfer learning
- Performance evaluation using standard classification metrics

## Repository Structure

```text
data/
docs/
outputs/
src/
tests/
notebooks/
```

## Status

Project setup completed.

## Image Preprocessing Pipeline

The project applies several preprocessing operations before model training:

1. Image resizing (224 × 224)
2. Gaussian Blur for noise reduction
3. CLAHE for contrast enhancement

These steps improve image quality and highlight disease-related features while maintaining consistency across the dataset.