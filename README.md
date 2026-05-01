# MNIST Neural Network from Scratch

## Overview
This project is a simple neural network built from scratch using only NumPy.  
The goal is to classify handwritten digits from the MNIST dataset without using deep learning frameworks like PyTorch or TensorFlow.

This project focuses on understanding the core mechanics of how neural networks actually learn.

---

## What This Model Does
- Takes a 28x28 grayscale image of a digit
- Flattens it into a vector of 784 values
- Passes it through a small neural network
- Outputs probabilities for digits 0 through 9
- Chooses the most likely digit as the prediction

---

## Architecture
This is a 2-layer fully connected neural network:

- Input layer: 784 features
- Hidden layer: 10 neurons with ReLU activation
- Output layer: 10 neurons with Softmax

Flow of data:

Input → Linear → ReLU → Linear → Softmax

---

## Key Concepts Used

### Forward Propagation
Data is passed through the network layer by layer to produce predictions.

### Activation Functions
- *ReLU*: removes negative values and introduces non-linearity  
- *Softmax*: converts outputs into probabilities that sum to 1  

### Loss Function
- *Cross-Entropy Loss* is used to measure how wrong the predictions are

### Backpropagation
Gradients are computed using the chain rule to understand how each parameter affects the loss.

### Gradient Descent
Weights and biases are updated by moving in the direction that reduces the loss.

---

## Training Process
For each iteration:
1. Perform forward propagation
2. Compute loss
3. Run backpropagation to get gradients
4. Update weights using gradient descent

This loop repeats until the model improves.


---

## Additional Features

### Loss Curve
The training loss is tracked over time to visualize how the model improves.

### Evaluation Function
A separate function checks accuracy on validation data to measure generalization.

### Prediction Demo
You can:
- Predict a single digit
- Compare prediction vs actual label
- Visualize the input image with its prediction

---

## Example Usage

```python
predict_single(0, X_dev, Y_dev, W1, b1, W2, b2)
show_prediction(0, X_dev, Y_dev, W1, b1, W2, b2)