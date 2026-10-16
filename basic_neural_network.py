import math

def sigmoid(x):
    return 1/(1 + math.exp(-x))

# inputs
hours = 6
attendence = 80

# weights for hidden layer 1
w1 = 0.1
w2 = 0.2
b1 = -1

# calculate hidden neuron 1
z1 = hours * w1 + attendence * w2 + b1
h1 = sigmoid(z1)

# weights for hidden layer 2
w3 = 0.2
w4 = 0.3
b2 = -1

# calculate hidden layer 2
z2 = hours * w3 + attendence * w4 + b2
h2 = sigmoid(z2)

# output neuron
w5 = 1.0
w6 = 1.0
b3 = -1

# calculate output neuron
z3 = hours * w5 + attendence * w6 + b3
output = sigmoid(z3)

print('Prediction:', output)