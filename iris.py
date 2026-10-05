from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


iris = load_iris()

X = iris.data.T
Y = np.eye(3)[iris.target].T

m = X.shape[1]


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    


def sigmoid_der(x):
    s = sigmoid(x)
    return s * (1 - s)


# Network: 4 -> 8 -> 3

W1 = np.random.randn(8, 4)
B1 = np.zeros((8, 1))

W2 = np.random.randn(3, 8)
B2 = np.zeros((3, 1))

learning_rate = 0.01
epochs = 20000

loss_history = []


for epoch in range(epochs):

    # Forward propagation
    Z1 = W1 @ X + B1
    A1 = sigmoid(Z1)

    Z2 = W2 @ A1 + B2
    A2 = softmax(Z2)

    # Loss
    loss = -np.sum(Y * np.log(A2 + 1e-8)) / m
    loss_history.append(loss)

    # Back propagation

    dZ2 = A2 - Y

    dW2 = (dZ2 @ A1.T) / m
    dB2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dA1 = W2.T @ dZ2

    dZ1 = dA1 * sigmoid_der(Z1)

    dW1 = (dZ1 @ X.T) / m
    dB1 = np.sum(dZ1, axis=1, keepdims=True) / m

    # Update parameters

    W2 -= learning_rate * dW2
    B2 -= learning_rate * dB2

    W1 -= learning_rate * dW1
    B1 -= learning_rate * dB1


# Predictions

predictions = np.argmax(A2, axis=0)
actual = np.argmax(Y, axis=0)

accuracy = np.mean(predictions == actual)

print("Accuracy:", accuracy)


# Plot loss

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()