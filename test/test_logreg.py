"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression import logreg, utils

def test_prediction():
	# all code below suggested by VScode
	X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # all combinations of two binary features
	W = np.array([1, 1, -1]) # weights for the two features
	# We will need to add a bias term to X, so we can test the make_prediction function directly
	X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))]) # adding column of ones at the end as bias "feature"
	expected_weighted_features = X_with_bias @ W # this is the linear combination of features and weights, including bias
	expected_predictions = 1 / (1 + np.exp(-expected_weighted_features)) # applying sigmoid function to get probabilities
	log_model = logreg.LogisticRegressor(num_feats=2, learning_rate=0.01, tol=0.001, max_iter=100, batch_size=10)
	log_model.W = W # directly set the weights to our test weights
	predictions = log_model.make_prediction(X_with_bias) # get predictions from the model
	assert np.allclose(predictions, expected_predictions), "Predictions do not match expected values."
	

def test_loss_function():
	# all code below suggested by VScode
	y_true = np.array([0, 0, 1, 1]) # true labels for our test cases
	y_pred = np.array([0.1, 0.2, 0.8, 0.9]) # predicted probabilities for class 1
	expected_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) # calculate expected loss using binary cross-entropy formula
	log_model = logreg.LogisticRegressor(num_feats=2, learning_rate=0.01, tol=0.001, max_iter=100, batch_size=10)
	loss = log_model.loss_function(y_true, y_pred) # get loss from the model
	assert np.isclose(loss, expected_loss), "Loss does not match expected value."

def test_gradient():
	# all code below suggested by VScode
	X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # all combinations of two binary features
	y_true = np.array([0, 0, 1, 1]) # true labels for our test cases
	W = np.array([1, 1, -1]) # weights for the two features and bias
	X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))]) # adding column of ones at the end as bias "feature"
	log_model = logreg.LogisticRegressor(num_feats=2, learning_rate=0.01, tol=0.001, max_iter=100, batch_size=10)
	log_model.W = W # directly set the weights to our test weights
	expected_y_pred = log_model.make_prediction(X_with_bias) # get predicted probabilities from the model
	expected_gradient = X_with_bias.T @ (expected_y_pred - y_true) / X_with_bias.shape[0] # calculate expected gradient using the condensed formula
	gradient = log_model.calculate_gradient(y_true, X_with_bias) # get gradient from the model
	assert np.allclose(gradient, expected_gradient), "Gradient does not match expected values."

def test_training():
	# all code below suggested by VScode
	# This test will check that the weights are updating during training.
	X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # all combinations of two binary features
	y_train = np.array([0, 0, 0, 1]) # only the case where both features are 1 is class 1
	X_val = X_train # for simplicity, we can use the same data for validation
	y_val = y_train
	log_model = logreg.LogisticRegressor(num_feats=2, learning_rate=0.1, tol=0.001, max_iter=10, batch_size=4)
	initial_W = log_model.W.copy() # save initial weights
	log_model.train_model(X_train, y_train, X_val, y_val) # train the model
	final_W = log_model.W # get final weights after training
	assert not np.allclose(initial_W, final_W), "Weights did not update during training."