# AI-Powered Skin Cancer Detection Using Deep Learning

## Overview

This project presents an end-to-end deep learning framework for automated skin cancer detection and classification from dermoscopic images. The system integrates lesion segmentation, multi-class skin lesion classification, and explainable artificial intelligence (XAI) to support early diagnosis and assist healthcare professionals in clinical decision-making.

The framework employs DeepLabV3+ for lesion segmentation, EfficientNet-B0 for skin lesion classification, and Grad-CAM for visual interpretation of model predictions. Additional experiments were conducted using ResNet50 and DenseNet121 for performance comparison across multiple benchmark datasets.

This project was developed as part of the IEEE Engineering in Medicine and Biology Society (EMBS) Pune Chapter AI/ML Program.

---

## Objectives

* Develop an automated skin lesion analysis system using deep learning.
* Perform accurate lesion segmentation from dermoscopic images.
* Classify skin lesions into clinically relevant categories.
* Compare multiple deep learning architectures.
* Improve model transparency through Explainable AI.
* Provide an easy-to-use web application for prediction and visualization.

---

## Key Features

* Automated lesion segmentation using DeepLabV3+
* Multi-class skin lesion classification
* Transfer learning with EfficientNet-B0
* Comparative analysis using ResNet50 and DenseNet121
* Explainable AI using Grad-CAM
* Image preprocessing and data augmentation
* Interactive Streamlit web application
* Confidence score generation
* Downloadable prediction reports

---

## System Architecture

The proposed framework consists of the following stages:

1. Image Acquisition
2. Image Preprocessing
3. Lesion Segmentation (DeepLabV3+)
4. Skin Lesion Classification (EfficientNet-B0)
5. Explainable AI using Grad-CAM
6. Prediction Report Generation
7. Streamlit Web Application

---

## Datasets

The models were trained and evaluated using the following publicly available datasets:

### ISIC 2018

* 10,015 dermoscopic images
* Seven skin lesion classes
* Includes segmentation masks
* Used for lesion classification and segmentation

### ISIC 2019

* 25,331 dermoscopic images
* Eight lesion categories
* Multi-class skin lesion classification benchmark

### ISIC 2020

* 33,126 dermoscopic images
* Binary classification (Benign vs Malignant)
* Includes patient metadata
* Designed for melanoma detection

### MILK10K

* 10,480 clinical and dermoscopic images
* 11 diagnostic categories
* Multi-modal skin lesion dataset
* Includes comprehensive clinical metadata

---

## Technologies Used

| Category             | Technology            |
| -------------------- | --------------------- |
| Programming Language | Python                |
| Deep Learning        | TensorFlow, Keras     |
| Image Processing     | OpenCV                |
| Data Processing      | NumPy, Pandas         |
| Visualization        | Matplotlib            |
| Segmentation         | DeepLabV3+            |
| Classification       | EfficientNet-B0       |
| Comparison Models    | ResNet50, DenseNet121 |
| Explainability       | Grad-CAM              |
| Web Framework        | Streamlit             |

---

## Model Pipeline

```
Input Image
      │
      ▼
Image Preprocessing
      │
      ▼
Lesion Segmentation (DeepLabV3+)
      │
      ▼
Lesion Classification (EfficientNet-B0)
      │
      ▼
Grad-CAM Visualization
      │
      ▼
Prediction & Confidence Score
      │
      ▼
Downloadable Report
```

---

## Evaluation Metrics

The proposed models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Macro F1-Score
* Macro Recall
* ROC-AUC
* Dice Coefficient (Segmentation)
* Intersection over Union (IoU)

---

## Experimental Models

The following deep learning architectures were evaluated:

* EfficientNet-B0
* ResNet50
* DenseNet121

Comparative performance analysis was conducted across all four datasets.

---

## Project Structure

```
Skin-Cancer-Detection/
│
├── datasets/
│   ├── isic2018/
│   ├── isic2019/
│   ├── isic2020/
│   └── milk10k/
│
├── segmentation/
│
├── classification/
│
├── gradcam/
│
├── streamlit_app/
│
├── models/
│
├── notebooks/
│
├── results/
│
├── requirements.txt
│
└── README.md
```

---

## Results

The proposed framework demonstrates competitive performance across multiple benchmark datasets through:

* Accurate lesion segmentation
* Robust skin lesion classification
* Comparative evaluation of multiple CNN architectures
* Explainable predictions using Grad-CAM
* Deployment through a Streamlit-based web application

---

## Future Work

Potential extensions of this project include:

* Vision Transformer (ViT)-based architectures
* Mobile application deployment
* Integration of patient clinical metadata
* Real-time inference on edge devices
* Federated learning for privacy-preserving training
* Multi-modal diagnostic frameworks

---

## Disclaimer

This project is intended solely for research and educational purposes. The predictions generated by the system are not a substitute for professional medical diagnosis or clinical decision-making. Healthcare professionals should always be consulted for medical evaluation and treatment.

---

## Authors

* Subhasree Paul
* Snehansha Das
  

Mentor: Dr. Ajey Kumar

IEEE Engineering in Medicine and Biology Society (EMBS), Pune Chapter
