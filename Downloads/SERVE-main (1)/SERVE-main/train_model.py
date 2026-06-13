import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================
# PATH
# =========================
DATA_PATH = os.path.join('MP_Data')

# =========================
# GET CLASSES (folder names)
# =========================
actions = np.array(sorted(os.listdir(DATA_PATH)))
print("Detected classes:", actions)

label_map = {label: num for num, label in enumerate(actions)}

# =========================
# LOAD DATA
# =========================
sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)

    if not os.path.isdir(action_path):
        continue

    for sequence in os.listdir(action_path):
        sequence_path = os.path.join(action_path, sequence)

        if not os.path.isdir(sequence_path):
            continue

        window = []

        # =========================
        # SAFE FRAME LOADING (FIXED)
        # =========================
        frame_files = [
            f for f in os.listdir(sequence_path)
            if f.endswith('.npy')
        ]

        frame_files = sorted(frame_files)

        for frame_file in frame_files:
            file_path = os.path.join(sequence_path, frame_file)

            try:
                res = np.load(file_path)
                window.append(res)
            except:
                continue

        # Only keep valid sequences
        if len(window) > 0:
            sequences.append(window)
            labels.append(label_map[action])

# =========================
# FIX RAGGED SEQUENCES
# (make all same length)
# =========================
min_len = min(len(seq) for seq in sequences)
sequences = [seq[:min_len] for seq in sequences]

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42
)

# =========================
# MODEL
# =========================
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# =========================
# TRAIN
# =========================
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# =========================
# SAVE MODEL
# =========================
model.save('action.h5')

print("Training complete. Model saved as action.h5")