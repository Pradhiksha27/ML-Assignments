def gs_f():
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.metrics import f1_score
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.model_selection import cross_val_score
	from sklearn.model_selection import GridSearchCV

	X_train = pd.read_csv("/home/pradhiksha/Desktop/589/HW1/hw1/Data/x_train.csv", header=None) 
	y_train = pd.read_csv("/home/pradhiksha/Desktop/589/HW1/hw1/Data/y_train.csv", header=None) 


	X_train_folds = []
	y_train_folds = []

	clf = DecisionTreeClassifier()
	param = {'criterion':['gini','entropy'], 'max_depth': [6,7,8,9], 'class_weight' : ['balanced'], 'min_samples_split':[30,40,50,60]}

	grid = GridSearchCV(clf, param, cv=5)
	grid.fit(X_train, y_train)

	best_param = grid.best_params_
	print("Best params ", best_param)
	print(grid.best_score_)
