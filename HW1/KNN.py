
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


X_train = pd.read_csv("/home/pradhiksha/Desktop/589/HW1/hw1/Data/x_train.csv", header=None) 
y_train = pd.read_csv("/home/pradhiksha/Desktop/589/HW1/hw1/Data/y_train.csv", header=None) 

n_neighbours = [3,5,10,20,25]

num_folds = 5
X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

for k in n_neighbours:
	sum=0
	for i in range(num_folds):
		
		X_val = X_train_folds[i]
		y_val = y_train_folds[i] 
		y_val = np.reshape(y_val, (y_val.shape[0],))

		X_train_fold = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
		y_train_fold = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])
		y_train_fold = np.reshape(y_train_fold, (y_train_fold.shape[0],))

		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(X_train_fold, y_train_fold)
		y_pred_val = knn.predict(X_val)
		
		score_test = f1_score(y_val, y_pred_val)
		sum += score_test
		
	avgf1 = sum/5
	print("The F1_score for k = %d would be %f" % (k, avgf1))

