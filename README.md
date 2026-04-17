
## 📌 Overview

Diabetic Retinopathy (DR) is a serious diabetes complication that affects the eyes and can lead to **permanent blindness** if not detected early.

This project uses **Deep Learning (CNNs)** to automatically classify retinal fundus images into different stages of DR severity, helping in **early diagnosis and clinical decision support**.

---

## 🎯 Objectives

* Detect Diabetic Retinopathy from retinal images
* Classify disease severity into 5 stages
* Handle real-world challenges like class imbalance
* Build a scalable and reusable deep learning pipeline

---

## 🚀 Key Features

* 📥 Automated dataset download using Kaggle API
* 🧹 Robust data preprocessing pipeline
* ⚖️ Class imbalance handling techniques
* 🔄 Advanced image augmentation
* 📊 Data visualization & analysis
* 🧠 CNN-based classification model
* 📈 Improved generalization and performance

---

## 📂 Dataset

**Dataset:** *Diabetic Retinopathy 2015 Dataset (Colored & Resized)*
Source: Kaggle

### 🏷️ Classes

| Label | Description      |
| ----- | ---------------- |
| 0     | No DR            |
| 1     | Mild             |
| 2     | Moderate         |
| 3     | Severe           |
| 4     | Proliferative DR |

---

## 🛠️ Tech Stack

| Category         | Tools             |
| ---------------- | ----------------- |
| Language         | Python            |
| Deep Learning    | TensorFlow, Keras |
| Image Processing | OpenCV            |
| Data Handling    | Pandas, NumPy     |
| Visualization    | Matplotlib        |

---

## ⚙️ Project Workflow

### 1️⃣ Dataset Acquisition

* Download dataset using Kaggle API
* Extract and organize into directories

### 2️⃣ Data Preprocessing

* Image-label mapping
* Data cleaning and validation
* Structured dataset creation

### 3️⃣ Handling Class Imbalance

* Reduced dominance of majority class
* Applied sampling techniques

### 4️⃣ Data Augmentation

* Rotation
* Width & height shift
* Zoom
* Horizontal flip

### 5️⃣ Visualization

* Class distribution analysis
* Before & after balancing comparison

### 6️⃣ Model Training

* CNN-based architecture
* Trained on augmented dataset
* Optimized for classification accuracy

---

## 🧠 Model Architecture (Example)

* Convolutional Layers
* MaxPooling Layers
* Dropout for regularization
* Fully Connected Layers
* Softmax Output (5 classes)

---

## 📊 Results

* Improved dataset balance
* Better generalization on unseen data
* Accurate classification of DR stages


---

## ▶️ How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Setup Kaggle API

* Download `kaggle.json` from Kaggle
* Place it in:

```bash
~/.kaggle/kaggle.json
```

### 4️⃣ Run the Project

```bash
jupyter notebook notebooks/Project_code.ipynb
```

---

## 📈 Future Improvements

* 🔍 Use Transfer Learning (ResNet, EfficientNet)
* ⚡ Hyperparameter tuning
* 🌐 Deploy as a web app (Streamlit / Flask)
* 📱 Mobile-based diagnosis system
* 🧪 Improve performance on minority classes

---

## 💡 Use Cases

* Early detection of diabetic eye disease
* Assist ophthalmologists in diagnosis
* Healthcare AI applications
* Medical imaging research

