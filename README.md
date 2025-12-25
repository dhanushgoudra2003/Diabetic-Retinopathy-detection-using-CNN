🩺 Hybrid Attention-Based CNN for Diabetic Retinopathy Detection

📌 Overview

Diabetic Retinopathy (DR) is a leading cause of vision loss worldwide. Early and accurate detection is critical for preventing irreversible damage. This project presents a CNN-based deep learning approach with hybrid attention mechanisms to automatically detect and classify diabetic retinopathy from retinal fundus images.

The model is designed to capture both local lesion-level details and global structural patterns, enabling effective multi-stage classification of diabetic retinopathy severity.


🎯 Objectives

1) Detect diabetic retinopathy from retinal fundus images

2) Perform multi-stage classification of DR severity

3) Enhance image quality using advanced preprocessing techniques

4) Improve model performance using attention mechanisms

5) Evaluate model effectiveness using robust medical metrics


🧠 Methodology

1) Image Preprocessing

Contrast Limited Adaptive Histogram Equalization (CLAHE)

Normalization

Data augmentation for better generalization

2) Model Architecture

CNN backbone based on EfficientNet-B0

Hybrid Attention Mechanism (Channel + Spatial attention)

Captures fine-grained retinal features and global context

3) Training

Optimizer: Adam

Loss Function: Cross-Entropy Loss

Framework: PyTorch

4) Evaluation

Accuracy

Quadratic Weighted Kappa (QWK)

Precision & Recall


📊 Results

1) Accuracy: 75.78%

2) Quadratic Weighted Kappa (QWK): 0.8581

The high QWK score indicates strong agreement between predicted and actual disease stages, making the model suitable for medical screening assistance.


🛠 Tech Stack

Language: Python

Deep Learning: PyTorch, Torchvision

Computer Vision: OpenCV

Machine Learning: scikit-learn

Data Processing: NumPy, Pandas

Augmentation: TensorFlow (Keras ImageDataGenerator)

Visualization: Matplotlib

Platform: Google Colab


📂 Dataset

Kaggle Diabetic Retinopathy Dataset

Retinal fundus images graded into 5 severity levels:

0: No DR

1: Mild

2: Moderate

3: Severe

4: Proliferative DR


🚀 Applications

1) Early diagnosis and screening of diabetic retinopathy

2) Tele-ophthalmology for remote and rural healthcare

3) Clinical decision support for ophthalmologists

4) Cost-effective large-scale screening programs


