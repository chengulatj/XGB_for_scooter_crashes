# Predicting Scooter Crash Risk with XGBoost Across Spatial Buffers

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
   - [ Preparing Data](#1-preparing-data)
   - [Splitting Data](#2-splitting-data)
   - [Training the Model](#3-training-the-model)
   - [Evaluating the Model](#4-evaluating-the-model)
   - [Feature Importance & SHAP Values](#5-feature-importance--shap-values)
   - [Repeating for All Buffer Distances](#6-repeating-for-all-buffer-distances)
6. [Conclusion](#example-of-performance-metrics)
7. - [250 ft Buffer: Model Performance Summary](#250-ft-buffer-model-performance-summary)
   - [250 ft Buffer: Feature Importance and SHAP Values](#250-ft-buffer-feature-importance-and-shap-values)
8. [Additional Resources](#additional-resources)


## Introduction
In this repository, the code predicts scooter-related crashs using XGBoost. The dataset includes various buffer distances around crash locations (ranging from 250 ft to 5 ft). By analyzing these distances, we seek to understand their influence on crash occurrences and improve prediction accuracy. The code also implements model interpretability techniques such as feature importance and SHAP values to provide insights into the most relevant factors for crashes.

## Features
- **Buffer-Based Predictions**: Training models for different buffer distances (250 ft, 150 ft, 100 ft, 50 ft, 25 ft, 10 ft, and 5 ft) around crash sites.
- **Performance Metrics**: accuracy, precision, recall, F1 score, sensitivity, and specificity for each buffer distance model.
- **Feature Importance**: Visualize the importance of each feature in predicting crashes.
- **SHAP Values**: Leverage SHAP explanations to interpret how individual features affect model predictions.

## Requirements
- Python 3.x
- Libraries:
  - `pandas`
  - `xgboost`
  - `scikit-learn`
  - `matplotlib`
  - `shap`
 
## Installation
To install the necessary dependencies, run the following command:
```bash
pip install pandas xgboost scikit-learn matplotlib shap
```
Ensure that the dataset is available at the specified path. Example:

```bash
/content/drive/MyDrive/scooter_crashes_cleaned2.0.csv
```

## Usage

### 1. Preparing Data
For each buffer distance, the dataset is filtered to include only the columns relevant to the buffer (e.g., `BUFF_250`, `BUFF_150`, etc.). The target variable (`scooter`) indicates whether the crash involved a scooter.

### 2. Splitting Data
The dataset is split into training and testing sets using either an 80-20 or 70-30 split.

### 3. Training the Model
An `XGBClassifier` is initialized and trained on the training data for each buffer distance.

### 4. Evaluating the Model
   For each buffer distance model, the following metrics are calculated:
   - **Accuracy**: The percentage of correctly predicted instances.
   - **Precision**: The proportion of true positive predictions among all positive predictions.
   - **Recall (Sensitivity)**: The proportion of true positive instances among all actual positive instances.
   - **F1 Score**: The harmonic mean of precision and recall.
   - **Specificity**: The proportion of true negative predictions among all actual negative instances.
   These metrics are printed to provide a clear view of model performance.

### 5. Feature Importance & SHAP Values
Feature importance is plotted to identify key factors influencing scooter crashes. SHAP (SHapley Additive exPlanations) values are computed to provide a detailed explanation of how features affect the model’s predictions.

### 6. Repeating for All Buffer Distances
   The above steps are repeated for all buffer distances: 250 ft, 150 ft, 100 ft, 50 ft, 25 ft, 10 ft, and 5 ft. The goal is to compare the model performance across different buffer distances and determine which distance gives the best prediction accuracy.

## Conclusion
This study evaluates the performance of the XGBoost model across various buffer distances ranging from 5 ft to 250 ft to predict scooter crashes. A key observation is that as the buffer distance decreases, model sensitivity improves, while accuracy and specificity gradually decline.

For instance, at the widest buffer of 250 ft, the model achieved a high accuracy of 96% and specificity of 99%, but with a lower sensitivity of 25%. Conversely, at the closest buffer of 5 ft, sensitivity peaked at 96%, though with a slight drop in accuracy and specificity.

### 250 ft Buffer: Model Performance Summary:
- **Accuracy**: 0.96
- **Specificity**: 0.99
- **Sensitivity (Recall)**: 0.25
- **Precision**: 0.47
- **F1 Score**: 0.32

### 250 ft Buffer: Feature Importance and SHAP Values

- **Feature Importance**: The most influential feature in predicting scooter crashes at a 250 ft buffer is **vehicle type**, followed by **vehicle maneuver** and **right turn**. These factors significantly impact the likelihood of a crash, highlighting the importance of vehicle interactions and environmental features like **Traffic Control Devices (TCDs)** and **road conditions**.

- **SHAP Values**: SHAP analysis further explains these results by showing that **vehicle type** has the largest effect on predictions, with higher SHAP values increasing crash risks. Features like **vehicle maneuver** and **log_AADT** (traffic data) also contribute significantly, while elements like **junction type** and **season** affect crash likelihood at varying levels.

## Additional Resources

- [Published Paper](https://doi.org/10.1016/j.mlwa.2024.100574): Full details on the methodology, experiments, and results.

