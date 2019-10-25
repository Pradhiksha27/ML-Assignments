import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


X_train = pd.read_csv("/home/pradhiksha/Desktop/589/HW1/hw1/Data/x_train.csv", header=None) 
y_train = pd.read_csv("/home/pradhiksha/Desktop/589/HW1/hw1/Data/y_train.csv", header=None) 


num = [3, 6, 9, 12, 15]
n_time = []

num_folds = 5
X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

for k in num:
	sum=0
	start = time.time()
	for i in range(num_folds):
		
		X_val = X_train_folds[i]
		y_val = y_train_folds[i] 
		y_val = np.reshape(y_val, (y_val.shape[0],))

		X_train_fold = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
		y_train_fold = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])
		y_train_fold = np.reshape(y_train_fold, (y_train_fold.shape[0],))

		clf = DecisionTreeClassifier(max_depth=k)
		clf.fit(X_train_fold, y_train_fold)
		y_pred_val = clf.predict(X_val)
		
		score_test = f1_score(y_val, y_pred_val)
		sum += score_test
		
	avgf1 = sum/5
	print("The F1_score for k = %d would be %f" % (k, avgf1))
	end = time.time()
	n_time.append((end - start)*1000)

plt.plot(num, n_time)
plt.scatter(num, n_time)
plt.ylabel('Time taken in milliseconds')
plt.xlabel('Depth of the decision tree')
plt.show()
