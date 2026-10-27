
import cv2
import numpy as np
import joblib
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils import resample
import mediapipe as mp


# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH = Path("data/train")
MODEL_PATH = Path("gesture_model.pkl")
TEST_DATA_PATH = Path("gesture_test_data.npz")

GESTURES = ["fist", "like", "no_gesture", "palm", "point"]
SAMPLES_PER_CLASS = 200

X = []
y = []

mp_hands = mp.solutions.hands


# -----------------------------
# Extract hand landmarks
# -----------------------------
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:

    for gesture in GESTURES:
        folder = DATASET_PATH / gesture

        if not folder.exists():
            print(f"Missing folder: {folder}")
            continue

        features = []

        image_paths = [
            p for p in folder.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        print(f"\nProcessing {gesture}: {len(image_paths)} images")

        for image_path in image_paths:
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            if not result.multi_hand_landmarks:
                continue

            landmarks = result.multi_hand_landmarks[0]
            coords = []

            for landmark in landmarks.landmark:
                coords.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            if len(coords) == 63:
                features.append(coords)

        print(f"Valid hand samples: {len(features)}")

        if len(features) == 0:
            print(f"ERROR: No valid samples for {gesture}")
            continue

        # Balance the number of samples per gesture
        if len(features) >= SAMPLES_PER_CLASS:
            rng = np.random.default_rng(42)
            indices = rng.choice(
                len(features),
                SAMPLES_PER_CLASS,
                replace=False
            )
            features = [features[i] for i in indices]
        else:
            features = resample(
                features,
                replace=True,
                n_samples=SAMPLES_PER_CLASS,
                random_state=42
            )

        X.extend(features)
        y.extend([gesture] * len(features))


# -----------------------------
# Validate dataset
# -----------------------------
counts = Counter(y)
print("\nBalanced sample counts:", counts)

if len(counts) != len(GESTURES):
    raise RuntimeError(
        "Some gesture classes have no valid hand samples. "
        "Check the dataset folders."
    )

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)


# -----------------------------
# Split training and test data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save the held-out test set for later evaluation
np.savez(
    TEST_DATA_PATH,
    X_test=X_test,
    y_test=y_test,
    labels=np.asarray(GESTURES)
)

print(f"\nTest data saved to: {TEST_DATA_PATH.resolve()}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# -----------------------------
# Train Random Forest model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)


# -----------------------------
# Evaluate model
# -----------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nTest accuracy: {accuracy:.2%}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=GESTURES,
        zero_division=0
    )
)

print("\nConfusion matrix (rows=true, columns=predicted):")
print(GESTURES)
print(confusion_matrix(y_test, predictions, labels=GESTURES))


# -----------------------------
# Save trained model
# -----------------------------
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH.resolve()}")
print("\nTraining and evaluation completed successfully.")
