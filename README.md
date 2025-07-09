# 🌿 Plant Identifier - Deep Learning Image Classifier

This is a deep learning-powered web app that identifies **plant types** from images using a Convolutional Neural Network (CNN). Built with TensorFlow and deployed using Gradio, it achieves **97% accuracy** on unseen plant images.

👉 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Devligan/PlantIdentifier)**

---

## 🚀 Features

- 🌱 Identifies plant type from uploaded images
- 📸 Simple drag-and-drop image upload
- ⚙️ Powered by a CNN model trained on real-world plant datasets
- 🌐 Deployed using Gradio on Hugging Face Spaces

---

## 🧠 Model Overview

- **Architecture**: EfficientNetB0 (fine-tuned with dropout and global average pooling)
- **Accuracy**: ~97% on validation set
- **Framework**: TensorFlow / Keras
- **Input Shape**: 224x224 RGB images
- **Dataset**: Trained on the [Plants Type Dataset](https://www.kaggle.com/datasets/yudhaislamisulistya/plants-type-datasets) from Kaggle
- **Notebook**: Full training and evaluation in:  
  🚀 [`Plant_Image_Model_TensorFlow_Builder.ipynb`](https://huggingface.co/spaces/Devligan/PlantIdentifier/blob/main/Plant_Image_Model_TensorFlow_Builder.ipynb)

---

## 🛠 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **EfficientNetB0**
- **Gradio** (for web-based image upload UI)
- **Google Colab** (for model training with GPU support)
- **Matplotlib / Seaborn** (for evaluation visuals)
- **scikit-learn** (for metrics and confusion matrix)

---
