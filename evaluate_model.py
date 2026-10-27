
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# -----------------------------
# File paths
# -----------------------------
MODEL_PATH = Path("gesture_model.pkl")
TEST_DATA_PATH = Path("gesture_test_data.npz")


# -----------------------------
# Check required files
# -----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "gesture_model.pkl not found. Run train_model.py first."
    )

if not TEST_DATA_PATH.exists():
    raise FileNotFoundError(
        "gesture_test_data.npz not found. "
        "Run train_model.py first to create the test data."
    )


# -----------------------------
# Load model and test data
# -----------------------------
model = joblib.load(MODEL_PATH)

data = np.load(TEST_DATA_PATH, allow_pickle=False)

X_test = data["X_test"]
y_test = data["y_test"]
labels = data["labels"].tolist()


# -----------------------------
# Make predictions
# -----------------------------
predictions = model.predict(X_test)


# -----------------------------
# Calculate evaluation metrics
# -----------------------------
accuracy = accuracy_score(y_test, predictions)

report = classification_report(
    y_test,
    predictions,
    labels=labels,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)


# -----------------------------
# Display results
# -----------------------------
print("\n========== MODEL EVALUATION ==========")
print(f"Total test samples: {len(y_test)}")
print(f"Correct predictions: {np.sum(y_test == predictions)}")
print(f"Incorrect predictions: {np.sum(y_test != predictions)}")
print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(report)

print("Gesture classes:", labels)
print("\nConfusion Matrix:")
print(matrix)


# -----------------------------
# Save confusion matrix image
# -----------------------------
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=labels
)

display.plot(
    xticks_rotation=45,
    values_format="d"
)

plt.title("Gesture Recognition Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("\nConfusion matrix saved as confusion_matrix.png")


# -----------------------------
# Save text report
# -----------------------------
with open("accuracy_report.txt", "w", encoding="utf-8") as file:
    file.write("GESTURE RECOGNITION MODEL EVALUATION\n")
    file.write("=" * 40 + "\n")
    file.write(f"Test samples: {len(y_test)}\n")
    file.write(f"Accuracy: {accuracy:.2%}\n\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(str(matrix))
    file.write("\nGesture order: " + ", ".join(labels))

print("Evaluation report saved as accuracy_report.txt")
print("\nEvaluation completed successfully.")
