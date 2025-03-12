<p align="center">
<img  width='400' height='300'  src="https://github.com/SinghPriya5/Advance_DeepLearning_Brain_Tumor_Classification/blob/main/templates/static/image/download.jpg"></p>



<h1 style='color:blue'>𓂀 𝔅𝔯𝔞𝔦𝔫 𝔗𝔲𝔪𝔬𝔯 ℭ𝔩𝔞𝔰𝔰𝔦𝔣𝔦𝔠𝔞𝔱𝔦𝔬𝔫 𝔲𝔰𝔦𝔫𝔤 𝔇𝔢𝔢𝔭 𝔏𝔢𝔞𝔯𝔫𝔦𝔫𝔤 𓂀</h1>
<img align="right" width="300" height="300" src="https://github.com/SinghPriya5/Advance_DeepLearning_Brain_Tumor_Classification/blob/main/templates/static/image/images.jpg">

## Overview
This project aims to classify brain MRI images into two categories: **Tumorous** and **Non-Tumorous** using a deep learning approach. The model is trained on an augmented dataset and deployed using Flask for real-time predictions.

Brain tumors are abnormal growths of cells in the brain, which can be either malignant (cancerous) or benign (non-cancerous). Early detection is crucial for timely treatment and improved patient outcomes. This project leverages **convolutional neural networks (CNNs)** to automate tumor detection, making the process faster and more reliable than traditional manual diagnosis.

## Dataset
The dataset consists of MRI scans categorized into:
   
Dataset link:-  [Dataset](https://github.com/SinghPriya5/Advance_DeepLearning_Brain_Tumor_Classification/tree/main/data)

- **Tumorous** (MRI scans showing brain tumors)
- **Non-Tumorous** (MRI scans without tumors)

### Data Splitting
The dataset is divided into three parts:
- **Training Set** (80%)
- **Testing Set** (10%)
- **Validation Set** (10%)

[Augmented_Data](https://github.com/SinghPriya5/Advance_DeepLearning_Brain_Tumor_Classification/tree/main/augmented_data)

[Tumorous_And_Nontumorous](https://github.com/SinghPriya5/Advance_DeepLearning_Brain_Tumor_Classification/tree/main/tumorous_and_nontumorous)

Directory Structure:
```
Brain Tumor Classification/
│── augmented_data/
│   │── yes/ (Tumorous images)
│   │── no/ (Non-Tumorous images)
│── tumorous_and_nontumorous/
│   │── train/
│   │   │── tumorous/
│   │   │── nontumorous/
│   │── test/
│   │   │── tumorous/
│   │   │── nontumorous/
│   │── valid/
│   │   │── tumorous/
│   │   │── nontumorous/
```

## Data Augmentation
To enhance model performance, data augmentation techniques such as **rotation, flipping, and zooming** were applied using TensorFlow's `ImageDataGenerator`. This helps in increasing dataset diversity and reducing overfitting.

## Model Architecture
The project employs **VGG15**, a modified version of VGG16, to extract meaningful features from MRI scans. The model consists of multiple convolutional layers followed by max-pooling layers and fully connected dense layers to classify images accurately.

### Hyperparameters:
- Optimizer: **Adam**
- Loss function: **Categorical Crossentropy**
- Activation: **ReLU & Softmax**
- Epochs: **50**
- Batch Size: **32**

## Model Performance
Three different models were trained, and their performances were evaluated on the test dataset:

| Model | Accuracy | Loss |
|--------|----------|------|
| Model 01 | 84.26% | 0.4130 |
| Model 02 | 82.63% | 0.4727 |
| Model 03 (Selected) | 87.55% | 0.5012 |

**Final model:** Model 03 was selected and deployed using Flask.

## Deployment with Flask
The trained model was integrated into a **Flask web application** to allow users to upload MRI images and receive predictions in real-time.

### Steps to Run the Flask App:
1. Install dependencies:
   ```bash
   pip install tensorflow flask numpy pandas
   ```
2. Run the Flask server:
   ```bash
   python app.py
   ```
3. Open the browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## Prediction Page
<p align="center">
  <img src="https://github.com/SinghPriya5/Advance_DeepLearning_Brain_Tumor_Classification/blob/main/templates/static/image/Screenshot%202025-03-12%20023443.png" alt="Predicted Value 1" width="500" height="700">
  <img src="https://github.com/SinghPriya5/Advance_DeepLearning_Brain_Tumor_Classification/blob/main/templates/static/image/non.png" alt="Predicted Value 1" width="500" height="700">
</p>

### Sample Code for Making Predictions
```python
import requests
files = {'file': open('sample_mri.jpg', 'rb')}
response = requests.post('http://127.0.0.1:5000/predict', files=files)
print(response.json())
```

The output will indicate whether the uploaded MRI image is **Tumorous** or **Non-Tumorous**.

## Future Improvements
- Experimenting with **ResNet and EfficientNet** for better accuracy
- Implementing **Grad-CAM** for explainable AI
- Deploying the model on **AWS/GCP for cloud accessibility**

## Conclusion
This project demonstrates the power of deep learning in medical imaging, offering an AI-powered solution for early brain tumor detection. The integration of **VGG15**, **data augmentation**, and **Flask deployment** showcases the potential of AI in real-world healthcare applications.

By improving classification accuracy and making the model accessible through a web interface, this project contributes to the field of **AI-driven medical diagnostics**. Future work will focus on enhancing model performance and expanding its usability in clinical settings. 🚀
