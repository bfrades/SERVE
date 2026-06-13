import cv2
import numpy as np
import os
from utils import mediapipe_detection, draw_landmarks, extract_keypoints
import mediapipe as mp

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "MP_Data")

print("Saving dataset to:", os.path.abspath(DATA_PATH))

actions = [
    "0","1","2","3","4","5","6","7","8","9",
    "all","anything","available","change","else",
    "good_day","have","hello","here","how_many",
    "no","none","only","please","sorry",
    "thank_you","that_is","what","yes"
]

no_sequences = 50
sequence_length = 30

# ---------------- CREATE FOLDERS ----------------
for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# ---------------- MEDIAPIPE ----------------
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not detected!")
    exit()

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    for action in actions:
        for sequence in range(no_sequences):

            print(f"\nCollecting {action} sequence {sequence}")

            for frame_num in range(sequence_length):

                ret, frame = cap.read()

                if not ret:
                    print("Failed to grab frame")
                    continue

                # ---------------- MIRROR CAMERA ----------------
                frame = cv2.flip(frame, 1)

                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)

                cv2.putText(image,
                            f"{action} Seq {sequence} Frame {frame_num}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255), 2)

                cv2.imshow('OpenCV Feed', image)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("Exiting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                # ---------------- SAVE DATA ----------------
                keypoints = extract_keypoints(results)

                file_path = os.path.join(
                    DATA_PATH,
                    action,
                    str(sequence),
                    str(frame_num)
                )

                np.save(file_path, keypoints)

                print(f"Saved: {file_path}.npy")

cap.release()
cv2.destroyAllWindows()