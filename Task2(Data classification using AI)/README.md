# Task 2 - Data Classification using KNN

## Objective
Implement a K-Nearest Neighbors (KNN) classifier using the Iris dataset and evaluate its performance.

## Technologies Used
- Python
- NumPy
- Scikit-learn

## Dataset
- Iris Dataset (from scikit-learn)

## Features
- Load the Iris dataset
- Split dataset into training and testing sets
- Standardize the features
- Train a KNN classifier
- Predict test data
- Calculate accuracy
- Generate confusion matrix
- Generate classification report
- Calculate F1-score

## How to Run

1. Install the required libraries:

```bash
pip install numpy scikit-learn
```

2. Run the program:

```bash
python Dataclassification.py
```

## Output
The program displays:

- Accuracy Score
- Confusion Matrix
- Classification Report
- F1 Score

---

# Advanced Data Classification Using AI

## Overview
This advanced version of the Data Classification project improves the basic K-Nearest Neighbors (KNN) classifier by automatically selecting the optimal number of neighbors using Cross Validation. It also includes multiple evaluation metrics and visualizations to better analyze the model's performance.

## Features
- Iris dataset classification using K-Nearest Neighbors (KNN)
- Feature scaling using StandardScaler
- Train-Test Split with Stratified Sampling
- Automatic K-value selection using 5-Fold Cross Validation
- Performance evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - Classification Report
- Elbow Curve visualization for selecting the best K
- Confusion Matrix visualization
- 2D Decision Boundary visualization

## Technologies Used
- Python
- NumPy
- Matplotlib
- Scikit-learn

## Files
- `advanceddataclassification.py` – Advanced KNN classification program
- `elbow_curve.png` – Graph showing the optimal K value
- `confusion_matrix.png` – Visualization of prediction performance
- `decision_boundary.png` – Visualization of KNN decision regions

## Workflow
1. Load the Iris dataset.
2. Scale the features using StandardScaler.
3. Split the dataset into training and testing sets.
4. Determine the optimal K value using 5-Fold Cross Validation.
5. Train the KNN classifier.
6. Predict the test dataset.
7. Evaluate model performance.
8. Generate graphical visualizations.

## Output
The program generates:
- Classification Report
- Accuracy, Precision, Recall and F1 Score
- Confusion Matrix
- Elbow Curve
- Decision Boundary Plot

## Learning Outcomes
This project demonstrates:
- Data preprocessing techniques
- Feature scaling
- KNN classification
- Hyperparameter tuning using Cross Validation
- Model evaluation
- Data visualization using Matplotlib