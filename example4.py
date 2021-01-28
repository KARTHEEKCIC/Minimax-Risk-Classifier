"""
Simple example of using CMRC with 0-1 loss.

"""

import numpy as np
import warnings
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

from minimax_risk_classifiers.CMRC import CMRC

#import the datasets
from datasets import load_mammographic

if __name__ == '__main__':

	# Supress the warnings
	warnings.simplefilter('ignore')

	# Loading the dataset
	X, Y = load_mammographic(return_X_y=True)
	r = len(np.unique(Y))

	# Preprocess
	trans = SimpleImputer(strategy='median')
	X = trans.fit_transform(X, Y)

	# Fit the MRC model
	clf = CMRC(r=r, phi='gaussian').fit(X, Y)

	# Prediction
	print('\n\nThe predicted values for the first 3 instances are : ')
	print(clf.predict(X[:3, :]))

	# Predicted probabilities
	print('\n\nThe predicted probabilities for the first 3 instances are : ')
	print(clf.predict_proba(X[:3, :]))

	print('\n\nThe score is : ')
	print(clf.score(X, Y))