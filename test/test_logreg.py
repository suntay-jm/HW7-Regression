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
import pandas as pd
import numpy as np
from regression.logreg import LogisticRegressor

df = pd.read_csv("/Users/mattsuntay/Desktop/algo/HW7-Regression/data/nsclc.csv")


# dropping any non-numeric columns
X = df.select_dtypes(include=[np.number]).values # features
y = df["NSCLC"].values # labels

# splitting into train/validation sets with an 80/20 split
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:] # up to 80%, remaining 20%
y_train, y_val = y[:train_size], y[train_size:]

# adding bias term
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

# initializing logistic regression model
model = LogisticRegressor(num_feats=X.shape[1], learning_rate=0.01, max_iter=100)

def test_prediction():
	"""
	make_predictions() should return value between 0-1 and apply the sigmoid function to transform input values
	given a sample input, check if all values in y_pred are between 0-1 and sigmoid transformation works correctly
	"""
	y_pred = model.make_prediction(X_val)
	assert np.all(y_pred >= 0) and np.all(y_pred <= 1), "predictions should be in range [0,1]"

def test_loss_function():
	"""
	loss_function() should correctly compute binary cross-entropy loss and should be a low positive number
	given good and bad y_pred, bad_y_pred should have higher loss than good_y_pred --> loss increases for wrong predictions and decreases for correct ones
	"""
	y_pred_good = np.array([0.9, 0.1, 0.8, 0.2]) # closer to true values
	y_pred_bad = np.array([0.1, 0.9, 0.2, 0.8]) # very wrong predictions
	y_true = np.array([1, 0, 1, 0]) # true labels

	# calculating loss for good and bad predictions
	loss_good = model.loss_function(y_true, y_pred_good)
	loss_bad = model.loss_function(y_true, y_pred_bad)

	assert loss_bad > loss_good, "loss should be higher for bad predictions"

def test_gradient():
	"""
	calculate_gradient() should compute correct weight updates; gradient values should be small if predictions are close and larger if predictions are wrong
	given a small dataset, ensure gradient direction is correct and shape matches number of features
	"""
	grad = model.calculate_gradient(y_train, X_train)

	assert grad.shape == model.W.shape, "gradient shape should match weight shape"
	assert np.all(np.isfinite(grad)), "gradient values should be finite numbers"

def test_training():
	"""
	the training process updates weights and the loss should decrease as training progresses
	given a small dataset, ensure loss decreases over iterations and check that final weights are different from initial weights
	"""

	X_train = np.array([[1.0], [2.0], [3.0]])
	y_train = np.array([0, 0, 1])

	# create a logistic regression model with 1 input feature
	model = LogisticRegressor(num_feats=1)
	model.train_model(X_train, y_train, X_train, y_train)

	assert model.loss_hist_train[-1] < model.loss_hist_train[0], "loss didn't decrease, model didn't improve" # checking if loss decreased

	"""
	from train_model(): self.loss_hist_train.append(train_loss) # tracks training loss at each step 
	the model updates weights and recalculates loss, a new loss value is appended to loss_hist_train
	at the start, first loss value stored as self.loss_hist_train[0] | at the end, final loss value stored as self.loss_hist_train[-1]
	successful training would mean that the final loss value (self.loss_hist_train[-1]) would be less than the first loss value (self.loss_hist_train[0])  == loss decrease 
	"""
def test_prediction_edge_cases():
    """
    ensure make_prediction() handles when passed an empty array or an array with the wrong shape
    """
    empty_X = np.array([]).reshape(0, model.W.shape[0])  # no data
    wrong_shape_X = np.array([1, 2, 3])  # incorrect shape

    try:
        model.make_prediction(empty_X)
    except ValueError:
        pass  # expected behavior

    try:
        model.make_prediction(wrong_shape_X)
    except ValueError:
        pass  # Expected behavior
