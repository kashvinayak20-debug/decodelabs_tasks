"""
Project 2: Data Classification Using AI
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay)

# PHASE 1 (INPUT): LOAD & UNDERSTAND THE DATASET
iris = load_iris()
X = iris.data   # (150, 4): sepal_len, sepal_wid, petal_len, petal_wid
y = iris.target # (150,): 0=setosa, 1=versicolor, 2=virginica
feature_names = iris.feature_names
target_names = iris.target_names

# PHASE 2 (INPUT): FEATURE SCALING -- "The Gatekeeper Rule"
# KNN is distance-based -- every feature must sit on the same scale.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PHASE 3 (PROCESS): TRAIN / TEST SPLIT -- "Structural Integrity"
# Shuffle removes order bias, stratify keeps class balance in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=True,
    stratify=y, random_state=42
)

# PHASE 4 (PROCESS): CHOOSING K -- the elbow method
# K is tuned via 5-fold CV on the TRAINING set only.
# The test set stays locked until the one-time final check --
# tuning against it would quietly leak information into an
# metric meant to be an honest, unseen check.
k_range = range(1, 31)
cv_error_rates = []
for k in k_range:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn_k, X_train, y_train, cv=5)
    cv_error_rates.append(1 - cv_scores.mean())

best_k = list(k_range)[int(np.argmin(cv_error_rates))]
# -> best_k = 6 (lowest CV error = 0.033)

plt.plot(list(k_range), cv_error_rates, marker="o")
plt.axvline(best_k, linestyle="--")
plt.savefig("elbow_curve.png")

# PHASE 5 (PROCESS): INSTANTIATE / FIT / PREDICT
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)          # FIT: memorize the map
predictions = model.predict(X_test)    # PREDICT: apply the logic

# PHASE 6 (OUTPUT): VALIDATION -- beyond the "accuracy mirage"
acc = accuracy_score(y_test, predictions)
prec = precision_score(y_test, predictions, average="macro")
rec = recall_score(y_test, predictions, average="macro")
f1 = f1_score(y_test, predictions, average="macro")
# -> accuracy 0.967, precision 0.944, recall 0.933, f1 0.933

print(classification_report(y_test, predictions, target_names=target_names))

cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Blues")
plt.savefig("confusion_matrix.png")

# BONUS: 2D decision boundary using sepal length + petal width,
# mirroring the "Feature_01 / Feature_02" illustration in the deck.
f1_idx, f2_idx = 0, 3
X2 = X_scaled[:, [f1_idx, f2_idx]]
model_2d = KNeighborsClassifier(n_neighbors=best_k)
model_2d.fit(X2, y)
# ... meshgrid + contourf -> decision_boundary.png (full version in the downloaded file)
# ----------------- Decision Boundary Plot -----------------

# Define the minimum and maximum values for both features
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1

# Create a grid of points covering the entire feature space
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

# Predict the class for every point in the grid
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])

# Reshape predictions back to grid shape
Z = Z.reshape(xx.shape)

# Plot the colored decision regions
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")

# Plot the actual flower samples
scatter = plt.scatter(
    X2[:, 0],
    X2[:, 1],
    c=y,
    cmap="viridis",
    edgecolor="black"
)

# Add labels and title
plt.xlabel(feature_names[f1_idx])
plt.ylabel(feature_names[f2_idx])
plt.title("KNN Decision Boundary (2 Features)")

# Add legend
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=target_names,
    title="Classes"
)

# Save the figure
plt.savefig("decision_boundary.png")
plt.close()