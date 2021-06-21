import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('datasets/2/pima_indians_diabetes.csv')
y = data['class'].values
x = data.drop(['class'], axis=1).values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, shuffle=False)

table = []

#svm
clf = SVC(kernel='rbf',gamma=0.00009)
clf.fit(X_train,y_train)
table.append(['SVM',clf.score(X_test,y_test)])

#bagging-LR
base = LogisticRegression(max_iter=1000)
clf = BaggingClassifier(base_estimator= base,n_estimators=40,random_state=0)
clf.fit(X_train,y_train)
table.append(['Bagging LogisticRegression',clf.score(X_test,y_test)])

#Adaboost
clf = AdaBoostClassifier(random_state=0,n_estimators=50)
clf.fit(X_train,y_train)
table.append(['AdaBoost',clf.score(X_test,y_test)])


print(table)