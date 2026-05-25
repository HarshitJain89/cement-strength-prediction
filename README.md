# 🏗️ Cement Strength Prediction using ML & Neural Networks

## Overview
A regression project that predicts the compressive strength of cement mixtures based on their chemical composition and age. Compares multiple modelling approaches - from simple linear regression to deep neural networks - achieving an **R² score of 0.95 on the test set** using an Artificial Neural Network.

## Problem Statement
The compressive strength of cement is a critical quality metric in construction. Testing it physically is time-consuming and expensive. Can a machine learning model predict strength from a mixture's known ingredients and curing age?

## Approach

A full ML pipeline was built and compared across 4 models:

| Model | Test R² Score |
|-------|--------------|
| Simple Linear Regression | ~0.75-0.77 |
| Polynomial Regression (degree 2) | Improved |
| Polynomial + K-Fold Cross Validation | Mean R² ~0.84 |
| **Artificial Neural Network** | **0.97 (train) / 0.95 (test)** |

## Exploratory Data Analysis
- Univariate and multivariate analysis including pairwise plots and correlation heatmap
- Outlier detection via box plots and IQR method
- Feature importance analysis - low-importance features identified and dropped

## Gaussian Mixture Model (Clustering)
A GMM with 4 clusters fitted to identify natural age groupings in the dataset - visualised with cluster means and distribution shapes.

## Key Takeaway
Simple linear models explained ~75% of variance. Polynomial features pushed this to ~84%. A neural network achieved 95% - demonstrating the value of deep learning for non-linear regression in engineering.

## Tech Stack
```
Python | scikit-learn | TensorFlow/Keras | NumPy | Pandas | Matplotlib | Seaborn
```

## How to Run
1. Clone the repo
2. Install dependencies: `pip install tensorflow scikit-learn numpy pandas matplotlib seaborn`
3. Open in Jupyter or Google Colab and run all cells

---
*Completed as part of the Post-Graduate Program in AI & Machine Learning - University of Texas at Austin (2021)*
