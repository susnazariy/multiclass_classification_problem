# Wine Multiclass Classification

A machine learning project for classifying wines into **3 cultivar classes** based on chemical analysis. This project demonstrates handling **multiclass classification** problems with appropriate metrics and visualization techniques.

## Overview

This project classifies wines derived from three different grape cultivars grown in the same region of Italy. It follows a complete ML pipeline with special attention to multiclass-specific considerations like macro/weighted averaging and One-vs-Rest ROC curves.

## Dataset

The Wine dataset contains **178 samples** with **13 chemical features**:

| Feature | Description |
|---------|-------------|
| Alcohol | Alcohol content |
| Malic acid | Malic acid concentration |
| Ash | Ash content |
| Alcalinity of ash | Alkalinity measurement |
| Magnesium | Magnesium content |
| Total phenols | Total phenolic compounds |
| Flavanoids | Flavanoid concentration |
| Nonflavanoid phenols | Non-flavanoid phenols |
| Proanthocyanins | Proanthocyanin content |
| Color intensity | Wine color intensity |
| Hue | Color hue |
| OD280/OD315 | Protein content indicator |
| Proline | Proline amino acid content |

**Target Variable (3 Classes):**
| Class | Cultivar | Samples | Percentage |
|-------|----------|---------|------------|
| 0 | Cultivar 1 | 59 | 33.1% |
| 1 | Cultivar 2 | 71 | 39.9% |
| 2 | Cultivar 3 | 48 | 27.0% |

## Project Pipeline

### 1. Data Cleaning
- Verified no missing values
- Outlier detection and visualization with boxplots

### 2. Exploratory Data Analysis
- **3-class target distribution** (pie chart & bar plot)
- Feature distributions by class (histograms, boxplots)
- Correlation analysis with heatmap
- Pairplots and 2D scatter plots

### 3. Feature Engineering
- **StandardScaler**: Normalized all features
- **PCA Analysis**: 2D and 3D projections showing class separability
- Identified highly correlated feature pairs

### 4. Model Training & Comparison

Implemented a custom `MultiClassifier` class with multiclass-specific metrics:

| Model | Description |
|-------|-------------|
| Logistic Regression | Multinomial classifier |
| Decision Tree | Tree-based classifier |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential ensemble |
| SVM | Support Vector Machine (OvR) |
| KNeighbors | Instance-based learning |
| Naive Bayes | Probabilistic classifier |
| AdaBoost | Adaptive boosting |

### 5. Hyperparameter Tuning
- GridSearchCV with 5-fold cross-validation
- Compared default vs. tuned performance

## Multiclass-Specific Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Macro Average** | Unweighted mean across classes | Equal class importance |
| **Weighted Average** | Weighted by class support | Imbalanced datasets |
| **ROC AUC (OvR)** | One-vs-Rest strategy | Overall discriminative ability |
| **Per-Class Metrics** | Individual precision/recall/F1 | Identify weak classes |

## Results

All models were evaluated using a consistent train/test split and multiclass-aware metrics. The table below summarizes exact test-set performance, macro-averaged metrics, One-vs-Rest ROC AUC, and execution time.

| Model               | Accuracy (Test) | Precision (Macro) | Recall (Macro) | F1 (Test, Macro) | F1 (Test, Weighted) | ROC AUC (OvR) | Time (s)   |
| ------------------- | --------------- | ----------------- | -------------- | ---------------- | ------------------- | ------------- | ---------- |
| Logistic Regression | **1.0000**      | **1.0000**        | **1.0000**     | **1.0000**       | **1.0000**          | **1.0000**    | 5.89       |
| Random Forest       | **1.0000**      | **1.0000**        | **1.0000**     | **1.0000**       | **1.0000**          | **1.0000**    | 1.39       |
| KNN                 | **1.0000**      | **1.0000**        | **1.0000**     | **1.0000**       | **1.0000**          | **1.0000**    | 0.36       |
| Naive Bayes         | 0.9722          | 0.9744            | 0.9762         | 0.9743           | 0.9723              | **1.0000**    | **0.0014** |
| SVM                 | 0.9722          | 0.9778            | 0.9667         | 0.9710           | 0.9720              | **1.0000**    | 0.07       |
| Decision Tree       | 0.9444          | 0.9583            | 0.9389         | 0.9457           | 0.9450              | 0.9493        | 0.18       |
| Gradient Boosting   | 0.9444          | 0.9505            | 0.9429         | 0.9453           | 0.9443              | 0.9978        | 1.50       |
| AdaBoost            | 0.9167          | 0.9267            | 0.9190         | 0.9198           | 0.9165              | 0.9978        | 0.71       |


## Visualizations

The notebook includes:
- **3x3 Confusion Matrices** for multiclass evaluation
- **Per-Class ROC Curves** (One-vs-Rest)
- **Decision Boundary Plots** on 2D PCA projection
- **Per-Class F1-Score Comparison** across models
- **3D PCA Scatter Plot** showing class separation

## Tech Stack

- **Python 3.13**
- **pandas** — Data manipulation
- **NumPy** — Numerical computing
- **scikit-learn** — ML algorithms, preprocessing & metrics
- **Matplotlib & Seaborn** — Visualization

## Project Structure

```
├── WineMulticlassML.ipynb    # Main notebook with complete analysis
├── README.md                  # Project documentation
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/wine-multiclass-classification.git
   cd wine-multiclass-classification
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook WineMulticlassML.ipynb
   ```

## Key Findings

- **Flavanoids**, **Color Intensity**, and **Proline** are the most predictive features
- PCA reveals excellent class separation in 2D/3D projections
- All three classes achieve similar performance (no neglected class)
- Ensemble methods and SVM perform best

