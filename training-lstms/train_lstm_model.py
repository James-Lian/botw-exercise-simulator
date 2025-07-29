import numpy as np
from lstm_model import *

import pandas as pd
from collections import defaultdict
import random

import pickle

name = input("Name the model: ")

body_scope = "arms"

if body_scope == "arms":
    data_range = [67, 103]
elif body_scope == "legs":
    data_range = [37, 67]
elif body_scope == "full":
    data_range = [1, -1]

filepath = input("Open a csv file (full path): ")
df = pd.read_csv(filepath)

data_context = []
joint_time_data = defaultdict(list)
label_data = {}

curr_action = 0
for index, row in df.iterrows():
    joint_time_data[row.iloc[0]].append([row.iloc[3]] + list(row.iloc[data_range[0]:data_range[1]]))
    label_data[row.iloc[0]] = row.iloc[1]

### LEGS LSTM
# sliding window
TIME_STEPS = 15
NUM_WINDOWS = 20

sequences = []
labels = []
for clip_id in joint_time_data:
    # selecting random sequences of motion data of length [TIME_STEPS] to feed as training data
    numbers = random.sample(range(0, len(joint_time_data[clip_id]) - TIME_STEPS), min(NUM_WINDOWS, len(joint_time_data[clip_id]) - TIME_STEPS))

    for i in range(len(numbers)):
        sequences.append(joint_time_data[clip_id][numbers[i]:numbers[i]+TIME_STEPS])
        labels.append(label_data[clip_id])

sequences = np.array(sequences, dtype=np.float32)
labels = np.array(labels)

num_labels = len(set(labels))

print("Data fully formatted.")

model = ActionRecognitionModel(input_shape=(TIME_STEPS, sequences.shape[2]), num_classes=num_labels, filename=name, epochs=100, patience=20)

print("Model loaded. Training...")

history, X_test, y_test = model.train(
    sequences,
    labels,
)
model.plot_training_history()

results = model.evaluate(X_test, y_test)
model.plot_confusion_matrix(results)

print("Test accuracy:", results['test_accuracy'])

save = input("Want to save? (y/n): ")
if save.lower() == "y":
    model.save_model()
    with open(model.filename + ".pkl", "wb") as f:
        pickle.dump(model.label_encoder, f)
    print("Label encoder saved to " + model.filename + ".pkl")