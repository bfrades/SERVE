import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# ---------------- CONFIG ----------------
DATA_PATH = "MP_Data"

actions = np.array([
    "0","1","2","3","4","5","6","7","8","9",
    "all","anything","available","change","else",
    "good_day","have","hello","here","how_many",
    "no","none","only","please","sorry",
    "thank_you","that_is","what","yes"
])

sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}

# ---------------- LOAD DATA ----------------
sequences, labels = [], []

for action in actions:
    for sequence in range(50):

        window = []

        for frame_num in range(sequence_length):
            res = np.load(os.path.join(
                DATA_PATH,
                action,
                str(sequence),
                f"{frame_num}.npy"
            ))
            window.append(res)

        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05
)

# ---------------- MODEL ----------------
model = Sequential()

model.add(LSTM(64, return_sequences=True, activation='relu',
               input_shape=(30, 126)))

model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, activation='relu'))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(len(actions), activation='softmax'))

model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- TRAIN ----------------
model.fit(
    X_train,
    y_train,
    epochs=100
)

# ---------------- SAVE MODEL ----------------
model.save("models/sign_model.h5")

print("MODEL SAVED SUCCESSFULLY")