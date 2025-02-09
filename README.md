README - Neural Network Training & Optimization
Home Assignment 1

Student Information

Name: [venkata nanda krishna yaram]
Student ID: [7000765514]
Course: [CS5720 NEURAL NETWORK AND DEEP LEARNING CRN23848]

Project Overview

This project focuses on tensor manipulations, loss functions, optimizer comparison, and logging with TensorBoard using TensorFlow and Keras. We implemented different machine learning workflows on the MNIST dataset, analyzed model performance, and visualized training results.

Contents

Q1.ipynb Tensor Manipulations & Reshaping

Q2.ipynb Loss Functions & Hyperparameter Tuning

Q3.ipynb Training a Model with Different Optimizers

Q4.ipynb Training a Neural Network with TensorBoard Logging

1. Tensor Manipulations & Reshaping

Objective:

Generate random tensors, check their rank and shape, reshape and transpose them.

Perform tensor broadcasting and addition.

Steps:

Create a random tensor of shape (4,6).

Find its rank and shape using TensorFlow functions.

Reshape it into (2,3,4) and transpose it to (3,2,4).

Broadcast a smaller tensor (1,6) to match the larger tensor and add them.

Explain the concept of broadcasting in TensorFlow.

2. Loss Functions & Hyperparameter Tuning

Objective:

Implement and compare Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE) loss functions.

Observe how small changes in predictions impact loss values.

Visualize the loss function comparisons.

Steps:

Define true values (y_true) and model predictions (y_pred).

Compute MSE and CCE using TensorFlow’s loss functions.

Modify predictions slightly and observe changes in loss values.

Plot loss function values using Matplotlib to compare trends.

3. Training a Model with Different Optimizers

Objective:

Train an MNIST classification model using Adam and SGD optimizers.

Compare their training and validation accuracy trends.

Visualize performance differences.

Steps:

Load and preprocess the MNIST dataset (normalize pixel values).

Define a simple feedforward neural network with an input, hidden, and output layer.

Train two models:

One with Adam optimizer.

One with SGD optimizer.

Compare training and validation accuracy trends.

Plot accuracy curves using Matplotlib for comparison.

4. Training a Neural Network with TensorBoard Logging

Objective:

Train an MNIST model while logging training data to TensorBoard.

Analyze accuracy and loss trends in TensorBoard.

Steps:

Load and preprocess the MNIST dataset.

Define a simple neural network model.

Set up TensorBoard logging using tf.keras.callbacks.TensorBoard().

Train the model for 5 epochs while storing logs in the logs/fit/ directory.

Launch TensorBoard using:

tensorboard --logdir logs/fit/

Analyze training vs. validation accuracy and loss trends.

Conclusion

This project covered various essential deep-learning tasks, including tensor operations, loss function analysis, optimizer comparison, and TensorBoard visualization. Key insights include:

Adam optimizer outperforms SGD in early training epochs.

Categorical Cross-Entropy (CCE) is more sensitive to prediction changes than MSE.

TensorBoard provides an intuitive way to track training progress.

By applying these concepts, we can further improve model performance and debugging strategies. 
