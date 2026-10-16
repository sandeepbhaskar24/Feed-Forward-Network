import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Training data
# [hours studied, attendance]
X = np.array([
    [1, 40],
    [2, 50],
    [3, 55],
    [4, 65],
    [5, 70],
    [6, 75],
    [7, 85],
    [8, 90],
    [9, 95]
], dtype=float)

# 0 - Fail / 1 - Pass
Y = np.array([
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1]
])

# creating the neural network
model = Sequential([
    Dense(8, activation='relu', input_shape=(2,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

# configure the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(X, Y,
          epochs=500,
          verbose=1)

# Test with a new student
new_student = np.array([
    [3, 30]
])

prediction = model.predict(new_student, verbose=0)
print('Pass probability:', prediction)

if prediction[0][0] >= 0.5:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")