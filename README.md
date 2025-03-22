# MRI Brain Tumor Classification using VGG16 and LIME Explanation

This project focuses on classifying MRI brain tumor images using a deep learning model based on the VGG16 architecture. The model is trained on a dataset of MRI images, and the predictions are explained using LIME (Local Interpretable Model-agnostic Explanations) to provide insights into the model's decision-making process. The model achieves an overall accuracy of `91.84%`, demonstrating strong performance across all classes.

---

## Dataset

The [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?select=Training) is sourced from Kaggle. The original dataset is structured as follows:

- Training Data: 5712 images
- Testing Data: 1311 images
- Classified into 4 classes: glioma, meningioma, no tumor, and pituitary.

## Data Splitting and Augmentation
- The Training Dataset is further split into 80% (4571 images) for training and 20% (1141 images) for validation.
- Data augmentation is applied to the training set to increase the robustness of the model. The augmentation techniques include random rotations, shifts, flips, brightness adjustments, and zooming.

---

## Model Architecture
The model architecture is as follows:
- Base Model: VGG16 (pre-trained on ImageNet)
- Flatten
- Dropout (0.3)
- Dense (128 units, ReLU activation)
- Dropout (0.2)
- Dense (output units, Softmax activation)

![image](https://github.com/user-attachments/assets/06d68389-c8d4-410c-9010-cd8da7a81817)

## Training
The model is trained with the following parameters:
- Optimizer: Adam (learning rate = 0.0001)
- Loss Function: Categorical Crossentropy
- Epochs: 20
- Batch Size: 32
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

![image](https://github.com/user-attachments/assets/ab40c29b-aa15-4323-bd66-0c0d60abe779)


---

## Results

The model achieves the following performance metrics on the test dataset:

- **Test Accuracy**: `0.9184`
- **Test Loss**: `0.2920`

### Classification Report

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Glioma       | 0.95      | 0.85   | 0.90     | 300     |
| Meningioma   | 0.90      | 0.84   | 0.87     | 306     |
| Notumor      | 0.97      | 0.98   | 0.97     | 405     |
| Pituitary    | 0.84      | 0.99   | 0.91     | 300     |

- **Macro Avg**: Precision = `0.92`, Recall = `0.91`, F1-Score = `0.91`
- **Weighted Avg**: Precision = `0.92`, Recall = `0.92`, F1-Score = `0.92`

### Confusion Matrix
![image](https://github.com/user-attachments/assets/19afd4d0-6511-412a-a645-8d45ef948959)


## LIME Explanation
LIME (Local Interpretable Model-agnostic Explanations) is used to explain the model's predictions by highlighting the regions of the image that contributed most to the prediction.
- Perturbation: LIME generates perturbed versions of the input image by randomly masking parts of the image.
- Explanation: LIME fits a simple interpretable model (e.g., linear regression) to explain the relationship between the perturbed images and the model's predictions.
- Visualization: The most important regions of the image are highlighted to show their contribution to the prediction.

![image](https://github.com/user-attachments/assets/e7a89fab-d40e-4edf-a607-6474d23ca8ed)

### Key Findings from LIME Analysis
- LIME successfully identified relevant regions in all five selected images, demonstrating its effectiveness in explaining the model's predictions.
- Entire Tumor Highlighted: In Figure 2 and Figure 3, LIME highlighted the entire tumor region, indicating that the model correctly identified and focused on the complete tumor area.
- Partial Tumor Highlighted: In the remaining three figures, LIME highlighted only part of the tumor. This could be due to:
  - The tumor's irregular shape or size.
  - The presence of overlapping structures or noise in the image.
  - Limitations in the perturbation process or the interpretable model used by LIME.
- Relevance of Highlighted Regions:
  - The highlighted regions were relevant overall, meaning that the model focused on meaningful features (e.g., tumor regions) rather than irrelevant areas (e.g., background or noise).
  - This indicates that the model is learning to identify important patterns associated with brain tumors, which aligns with the high accuracy achieved.
