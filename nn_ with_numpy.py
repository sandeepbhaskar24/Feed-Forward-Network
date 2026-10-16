import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Inputs
x = np.array([0.1, 0.2])

# weights for hidden layer 1 and 2
w1 = np.array([
    [0.1, 0.2],
    [0.02, 0.03]
])

# bias for hidden layer 1 and 2
b1 = np.array([1, 1])

# weights for output layer 
w2 = np.array([
    [0.5],
    [0.8]
])

# bias for output layer
b2 = np.array([1])

# Forward pass
hidden = sigmoid(x @ w1 + b1)
output = sigmoid(hidden @ w2 + b2)

print('Hidden layer:', hidden)
print('Output:', output)
print('Result:', 'Pass' if output > 0.5 else 'Fail')