import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_data


# Initialize weights and biases for a simple 2 layer neural network
def init_params():
    # 784 input features, 10 hidden units
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5

    # 10 hidden units, output classes (digits 0-9)
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

# replaces negative values with 0
def ReLU(Z):
    return np.maximum(Z, 0)

# converts raw scores into probabilities
def softmax(Z):
    # Subtract max for numerical stability 
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)

# Forward pass through the network
def forward_prop(W1, b1, W2, b2, X):
    # First layer
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    # Output layer
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    # Check activation behavior
    if np.random.rand() < 0.01: 
        print("A1 stats:", np.min(A1), np.max(A1), np.mean(A1))
        print("A2 stats:", np.min(A2), np.max(A2), np.mean(A2))

    return Z1, A1, Z2, A2

# Convert labels into one-hot encoded format
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

# Derivative of ReLU
def deriv_ReLU(Z):
    return Z > 0

# Compute gradients for all parameters
def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)

    # Output layer gradients
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    # Hidden layer gradients (chain rule)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    # Check gradient sizes
    if np.random.rand() < 0.01:
        print("dW1 norm:", np.linalg.norm(dW1))
        print("dW2 norm:", np.linalg.norm(dW2))


    return dW1, db1, dW2, db2

# Update weights and biases using gradient descent
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    

    return W1, b1, W2, b2

# Get predicted class
def get_predictions(A2):
    return np.argmax(A2, 0)

# Compute accuracy by comparing predictions to labels
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def compute_loss(A2, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    return -np.sum(one_hot_Y * np.log(A2 + 1e-8)) / m

def evaluate(X, Y, W1, b1, W2, b2):
    # run a full forward pass on the dataset
    # this gives the predicted probabilities for each class
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)

    # pick the class with the highest probability for each example
    preds = get_predictions(A2)

    # compare predictions to actual labels and return accuracy
    return get_accuracy(preds, Y)


def predict_single(index, X, Y, W1, b1, W2, b2):
    # take one example from the dataset and reshape it into a column vector
    x = X[:, index].reshape(-1, 1)

    # run it through the network to get prediction probabilities
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, x)

    # choose the most likely class
    pred = get_predictions(A2)[0]

    # print both prediction and true label so we can compare
    print("Prediction:", pred)
    print("Actual label:", Y[index])


def show_prediction(index, X, Y, W1, b1, W2, b2):
    # grab one image and reshape it for the network
    x = X[:, index].reshape(-1, 1)

    # run forward pass to get prediction
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, x)
    pred = get_predictions(A2)[0]

    # reshape the flat vector back into a 28x28 image for display
    image = X[:, index].reshape(28, 28)

    # show the image along with prediction and actual label
    plt.imshow(image, cmap='gray')
    plt.title(f"Prediction: {pred}, Label: {Y[index]}")
    plt.axis('off')
    plt.show()

# Training loop
def gradient_descent(X, Y, alpha, iterations):
    losses = []
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        # Forward pass
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        # Backward pass
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)

        # Parameter update
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # Print progress every 50 iterations
        loss = compute_loss(A2, Y)
        losses.append(loss)

        if i % 50 == 0:
            print(f"Iteration: {i}")
            print("Loss:", loss)
            print(f"Iteration: {i}")
            print("Loss:", compute_loss(A2, Y))
            print("Accuracy:", get_accuracy(get_predictions(A2), Y))
            print("Dev accuracy:", evaluate(X_dev, Y_dev, W1, b1, W2, b2))

            # Activation stats
            print("A1 stats:", np.min(A1), np.max(A1), np.mean(A1))
            print("A2 stats:", np.min(A2), np.max(A2), np.mean(A2))

            # Gradient norms
            print("dW1 norm:", np.linalg.norm(dW1))
            print("dW2 norm:", np.linalg.norm(dW2))

    return W1, b1, W2, b2, losses

# Load dataset and train the model
if __name__ == "__main__":
    X_train, Y_train, X_dev, Y_dev = load_data()

    W1, b1, W2, b2, losses = gradient_descent(X_train, Y_train, 0.10, 500)

    print("Dev accuracy:", get_accuracy(
        get_predictions(forward_prop(W1, b1, W2, b2, X_dev)[3]),
        Y_dev
    ))

    predict_single(0, X_dev, Y_dev, W1, b1, W2, b2)
    show_prediction(0, X_dev, Y_dev, W1, b1, W2, b2)