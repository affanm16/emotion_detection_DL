# Emotion Detection Using CNN and FER-2013 Dataset

## Overview

This project aims to detect human emotions in images using Convolutional Neural Networks (CNNs) and the FER-2013 dataset. We address the class imbalance in the dataset through image augmentation and class weights, enhancing the model's robustness. Various CNN architectures were explored, including ResNet50v2, VGG16, InceptionResNetV2, EfficientNetV2S, ConvNeXtXLarge, and custom CNN models, to optimize performance. The final model, based on ResNet50v2, achieved an overall accuracy of 61% on emotion classification, with detailed evaluation metrics such as precision, recall, and F1-scores provided for each of the 7 emotion labels.

Additionally, we implemented real-time emotion detection in live video streams using Gradio and OpenCV. This allows for dynamic visualization of emotion labels directly on the video feed, demonstrating the practical application of the trained model.

## Tech Stack

- Python
- TensorFlow
- Keras
- ResNet50v2
- VGG16
- OpenCV
- Gradio

## Key Features

- Addressed class imbalance in FER-2013 dataset using image augmentation and class weights.
- Explored various CNN architectures to optimize performance.
- Achieved 61% overall accuracy on emotion classification.
- Implemented real-time emotion detection in live video streams using Gradio and OpenCV.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/affanm16/emotion_detection_DL.git

```
2. Install the dependencies:

```bash
pip install -r requirements.txt

```
3. Run real-time emotion detection on live video stream:
```bash
python emotion_app.py
```

## Dataset
The FER-2013 (Facial Expression Recognition 2013) dataset contains 48x48 pixel grayscale images of faces, annotated with 7 emotion labels: anger, disgust, fear, happiness, sadness, surprise, and neutral.