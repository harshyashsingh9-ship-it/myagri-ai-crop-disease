# 🌱 AI-Based Crop Disease Prediction & Explainable Advisory System

An AI-powered computer vision system for detecting crop diseases from leaf
images using deep learning and providing confidence-aware, explainable
predictions for farmers.

## 🚀 Features

- Multi-crop disease classification
- Deep learning-based image classification
- Image quality validation
- Confidence score
- Grad-CAM visual explanations
- Disease severity estimation
- Farmer-friendly recommendations
- Hindi/English interface
- Model comparison
- Reproducible training pipeline
- Web-based inference interface

## 🎯 Problem Statement

Crop diseases significantly affect agricultural productivity.
Many farmers do not have immediate access to agricultural experts for
identifying diseases from visible symptoms.

This project aims to provide an accessible AI-based preliminary
screening system using smartphone leaf images.

## 💡 Proposed Solution

The system processes a crop leaf image through an image-quality
validation stage followed by a deep-learning classifier.

The prediction pipeline provides:

1. Crop identification
2. Disease prediction
3. Confidence score
4. Visual explanation
5. Severity estimate
6. Recommended next steps

## 🧠 Machine Learning

Models evaluated:

- MobileNetV3
- ResNet50
- EfficientNet

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Inference time

## 🔬 Explainable AI

Grad-CAM is used to visualize the image regions that contributed
to the model's prediction.

## 🏗️ Architecture

[architecture diagram]

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| MobileNetV3 | XX | XX | XX | XX |
| ResNet50 | XX | XX | XX | XX |
| EfficientNet | XX | XX | XX | XX |

## 🖥️ Demo

[Add screenshot/GIF]

## ⚙️ Installation

```bash
git clone YOUR_REPOSITORY_URL
cd ai-crop-disease-prediction

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
