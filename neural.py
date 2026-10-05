import numpy as np


# XOR dataset
X = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1]
])

Y = np.array([
    [0, 1, 1, 0]
])


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# Network: 2 → 3 → 1
np.random.seed(42)

W1 = np.random.randn(3, 2)
b1 = np.zeros((3, 1))

W2 = np.random.randn(1, 3)
b2 = np.zeros((1, 1))


# Training settings
learning_rate = 2.0
epochs = 10000


loss_history = []


# Training
for epoch in range(epochs):

    # Forward propagation
    Z1 = W1 @ X + b1
    A1 = sigmoid(Z1)

    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)


    # Calculate loss
    m = X.shape[1]

    loss = -np.mean(
        Y * np.log(A2 + 1e-8) +
        (1 - Y) * np.log(1 - A2 + 1e-8)
    )

    loss_history.append(loss)


    # Backpropagation
    dZ2 = A2 - Y

    dW2 = (dZ2 @ A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dA1 = W2.T @ dZ2
    dZ1 = dA1 * sigmoid_derivative(Z1)

    dW1 = (dZ1 @ X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m


    # Update parameters
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.6f}")


# Test the trained network
Z1 = W1 @ X + b1
A1 = sigmoid(Z1)

Z2 = W2 @ A1 + b2
A2 = sigmoid(Z2)    


print("\nPredictions:")
print(A2)

print("\nRounded predictions:")
print(np.round(A2))

print("\nExpected:")
print(Y)


# Show learned parameters
print("\nW1:")
print(W1)

print("\nb1:")
print(b1)

print("\nW2:")
print(W2)

print("\nb2:")
print(b2)