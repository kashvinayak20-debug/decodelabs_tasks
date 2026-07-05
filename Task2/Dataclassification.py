
# Project 2: Data Classification Using AI


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

# ── STEP 1: LOAD THE DATASET ─────────────────────────────────
# Iris: 150 samples, 3 classes, 4 features
# Features: sepal_length, sepal_width, petal_length, petal_width
# Classes:  0=Setosa, 1=Versicolor, 2=Virginica

iris = load_iris()
X = iris.data    # shape: (150, 4) — feature matrix
y = iris.target  # shape: (150,)  — class labels

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {list(iris.target_names)}")

# ── STEP 2: TRAIN-TEST SPLIT ─────────────────────────────────
# test_size=0.2   -> 20% test (30 samples), 80% train (120)
# random_state=42 -> reproducible shuffle
# stratify=y      -> keep class balance in both sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ── STEP 3: FEATURE SCALING (MANDATORY FOR KNN) ─────────────
# StandardScaler: z = (x - mean) / std
# Fit ONLY on training data, transform both, to avoid data leakage

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ── STEP 4: TRAIN THE KNN MODEL ─────────────────────────────
# n_neighbors=5: check 5 closest training points, majority vote
# KNN is "lazy" — fit() just stores the data; all the work
# happens at prediction time

model = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean",
    weights="uniform",
)
model.fit(X_train, y_train)
print("Model trained successfully!")

# ── STEP 5: MAKE PREDICTIONS ─────────────────────────────────

y_pred = model.predict(X_test)

# ── STEP 6: EVALUATE PERFORMANCE ────────────────────────────

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
cm = confusion_matrix(y_test, y_pred)

print("\n=== RESULTS ===")
print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.1f}%)")
print(f"F1 Score : {f1:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
print("\nDetailed Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ── STEP 7: PREDICT A NEW FLOWER ────────────────────────────
# Must scale the new input with the SAME fitted scaler

new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # a classic Setosa
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)
probabilities = model.predict_proba(new_flower_scaled)

print(f"\nNew flower measurements: {new_flower[0]}")
print(f"Predicted class: {iris.target_names[prediction[0]]}")
print(f"Confidence: {probabilities[0].max() * 100:.0f}%")