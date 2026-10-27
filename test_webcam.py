
import cv2
import time
import joblib
import numpy as np
import pyautogui
import mediapipe as mp
import os

from pathlib import Path
from collections import deque


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "gesture_model.pkl"
PPT_PATH = BASE_DIR / "presentation.pptx"

CONFIDENCE_THRESHOLD = 0.70
HISTORY_SIZE = 7
REQUIRED_VOTES = 5

COOLDOWN_SECONDS = 2.0
RELEASE_DELAY = 0.8


# ==========================================
# GESTURE MAPPING
# ==========================================

GESTURE_ACTIONS = {
    "like": ("right", "Next slide"),
    "fist": ("left", "Previous slide"),
    "palm": ("b", "Toggle black screen"),
    "point": ("space", "Play/Pause media"),
}


# ==========================================
# PYAUTOGUI SETTINGS
# ==========================================

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


# ==========================================
# OPEN POWERPOINT
# ==========================================

def open_presentation():
    if not PPT_PATH.exists():
        print(f"\nPresentation not found: {PPT_PATH}")
        print("Save your file as presentation.pptx in this folder.")
        return False

    try:
        os.startfile(str(PPT_PATH))
        print("\nPowerPoint presentation opened.")
        return True

    except OSError as error:
        print(f"Could not open PowerPoint: {error}")
        return False


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise SystemExit(
        f"Model not found: {MODEL_PATH}\n"
        "Run python train_model.py first."
    )

model = joblib.load(MODEL_PATH)

print("Gesture model loaded successfully.")
print("Classes:", model.classes_)


# ==========================================
# OPEN PRESENTATION
# ==========================================

presentation_opened = open_presentation()

if presentation_opened:
    input(
        "\nStart the PowerPoint slide show with F5, "
        "then press ENTER here to start the webcam."
    )
else:
    answer = input("Continue without PowerPoint? (y/n): ").strip().lower()

    if answer != "y":
        raise SystemExit("Program stopped.")


# ==========================================
# INITIALIZE MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

history = deque(maxlen=HISTORY_SIZE)

last_action_time = 0.0
no_hand_since = None
gesture_armed = True

last_prediction = "Waiting for hand"
last_confidence = 0.0


# ==========================================
# START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit(
        "Webcam could not open. Check camera permissions "
        "or try cv2.VideoCapture(1)."
    )

print("\nWebcam started.")
print("LIKE  -> Next slide")
print("FIST  -> Previous slide")
print("PALM  -> Toggle black screen")
print("POINT -> Play/Pause media")
print("Remove your hand between actions.")
print("Press Q in the webcam window to quit.")


# ==========================================
# MAIN GESTURE RECOGNITION LOOP
# ==========================================

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            result = hands.process(rgb_frame)
            current_time = time.time()

            if result.multi_hand_landmarks:

                no_hand_since = None

                hand = result.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                # Extract 21 landmarks x 3 coordinates = 63 features
                features = []

                for landmark in hand.landmark:
                    features.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                X = np.asarray(
                    features,
                    dtype=np.float32
                ).reshape(1, -1)

                # Predict gesture
                probabilities = model.predict_proba(X)[0]
                best_index = int(np.argmax(probabilities))

                gesture = str(model.classes_[best_index])
                confidence = float(probabilities[best_index])

                last_prediction = gesture
                last_confidence = confidence

                # Build a history of confident predictions
                if confidence >= CONFIDENCE_THRESHOLD:
                    history.append(gesture)
                else:
                    history.clear()

                # Confirm a stable gesture
                if len(history) == HISTORY_SIZE:

                    stable_gesture = max(
                        set(history),
                        key=history.count
                    )

                    vote_count = history.count(stable_gesture)

                    if (
                        vote_count >= REQUIRED_VOTES
                        and gesture_armed
                        and stable_gesture in GESTURE_ACTIONS
                        and current_time - last_action_time
                        >= COOLDOWN_SECONDS
                    ):

                        key, action_name = GESTURE_ACTIONS[stable_gesture]

                        print(
                            f"{stable_gesture.upper()}: {action_name} "
                            f"(confidence: {confidence:.2f})"
                        )

                        # Send keyboard command to focused application
                        pyautogui.press(key)

                        last_action_time = current_time
                        gesture_armed = False
                        history.clear()

            else:
                # Hand is not visible
                history.clear()

                if no_hand_since is None:
                    no_hand_since = current_time

                # Rearm after hand removal
                if current_time - no_hand_since >= RELEASE_DELAY:
                    gesture_armed = True

                last_prediction = "No hand detected"
                last_confidence = 0.0

            # ==========================================
            # DISPLAY INFORMATION
            # ==========================================

            status = "READY" if gesture_armed else "REMOVE HAND"

            cv2.putText(
                frame,
                f"Gesture: {last_prediction}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {last_confidence:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Status: {status}",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )

            cv2.putText(
                frame,
                "Q = Quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Gesture-Controlled PowerPoint",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\nGesture-control program stopped.")
