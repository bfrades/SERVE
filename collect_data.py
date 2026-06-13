import cv2
import numpy as np
import os
from utils import mediapipe_detection, draw_landmarks, extract_keypoints
import mediapipe as mp

# ---------------- CONFIG ----------------
DATA_PATH = "MP_Data"
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

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    for action in actions:
        for sequence in range(no_sequences):

            print(f"Collecting {action} sequence {sequence}")

            for frame_num in range(sequence_length):

                ret, frame = cap.read()

                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)

                # WAIT BEFORE STARTING
                if frame_num == 0:
                    cv2.putText(image, "STARTING COLLECTION",
                                (120,200),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0,255,0), 2)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    keypoints = extract_keypoints(results)

                    file_path = os.path.join(
                        DATA_PATH,
                        action,
                        str(sequence),
                        str(frame_num)
                    )

                    np.save(file_path, keypoints)

                    cv2.putText(image, f"{action} Seq {sequence}",
                                (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0,0,255), 1)

                    cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()