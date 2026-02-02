# Wildlife Detection Using Handcrafted Features  
**DS203 – Introduction to Data Science | Course Project**

## Project Overview
This project focuses on **wildlife detection in aerial images without using Convolutional Neural Networks (CNNs)**.  
Instead, we design a classical machine learning pipeline using **handcrafted visual features** combined with traditional ML classifiers.

The problem is formulated as a **grid-level binary classification task**, where each image is divided into an **8×8 grid (64 cells)** and each cell is classified as:
- **1** → Wildlife present  
- **0** → Wildlife absent  

---

## Problem Statement
Given standardized aerial images of size **800×600**, the objective is to:
- Detect wildlife at a **localized (cell-level)** rather than whole-image level
- Build a complete ML pipeline using **only handcrafted features**
- Generate interpretable outputs:
  - Highlighted grid cells containing wildlife
  - CSV file with predictions for each grid cell (col1–col64)

### Key Challenges
- Varying illumination and image quality
- Severe class imbalance (few wildlife-positive cells)
- Small or partially visible animals
- High-dimensional feature space

---

## Dataset & Preprocessing

### Image Standardization
- Enforced **4:3 aspect ratio** using center cropping
- Resized images to **800×600** (only downscaling)
- Logged preprocessing metadata in a manifest CSV for reproducibility

### Grid Generation
- Each image divided into an **8×8 grid (64 cells)**
- Grid overlays generated for visualization and labeling

---

## Ground Truth Creation
- Manual labeling of each grid cell:
  - **1** → Wildlife present
  - **0** → No wildlife
- Labels stored in a structured CSV file
- Custom interactive labeling interface developed to:
  - Reduce mislabeling
  - Improve speed and consistency
  - Directly map labels to grid cells

**Total labeled images:** 426

---

## Exploratory Data Analysis (EDA)
EDA was performed to analyze:
- Distribution of wildlife-positive cells per image
- Spatial distribution of wildlife across grid cells
- Severity of class imbalance

### Observations
- Wildlife appears most frequently near **central grid cells**
- Distribution of positive cells per image is **right-skewed**
- Strong photographer bias toward centering animals

---

## Feature Engineering
For each grid cell, the following handcrafted features were extracted:

### 1. Color Features
- RGB and HSV histograms
- Mean, standard deviation, skewness, kurtosis
- Capture vegetation background, lighting variation, and animal fur tones

### 2. Texture Features
- Grey-Level Co-occurrence Matrix (GLCM):
  - Contrast, Correlation, Energy, Homogeneity
- Local Binary Patterns (LBP)
- Gabor filter responses

### 3. Edge & Shape Features
- Histogram of Oriented Gradients (HOG)
- Canny and Sobel edge densities
- Harris corner statistics

All features are concatenated into a single feature vector per cell.

---

## Modeling Approach

### Attempt 1: Baseline Models
- **Features:** HOG + Color Histograms
- **Models:**
  - Logistic Regression
  - Random Forest
- Established baseline performance

### Attempt 2: Texture-Enhanced Pipeline
- Added **GLCM and LBP** texture features
- Retrained Logistic Regression and Random Forest
- Improved class separability and robustness

### Attempt 3: Advanced Classifiers
- **Features:** Color + Texture + Shape
- **Models:**
  - Support Vector Machine (SVM)
  - XGBoost
- Hyperparameter tuning using grid search and cross-validation

**Final Model Selected:** **XGBoost**  
(best overall accuracy, recall, and F1-score)

---

## Evaluation
Models were evaluated using:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC and Precision–Recall curves

**Important:**  
Train/Validation/Test splits were performed at the **image level** to avoid data leakage across grid cells.

---

## Outputs
The final pipeline produces:
1. **Highlighted images** showing predicted wildlife grid cells
2. **CSV output** containing:
