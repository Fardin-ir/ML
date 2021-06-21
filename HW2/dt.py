import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def get_confusion_matrix(actual, predicted):
    labels = np.unique(actual) 
    matrix = pd.DataFrame(np.zeros((len(labels), len(labels))))
    for i in range(len(labels)):
        for j in range(len(labels)):

            matrix.iloc[i, j] = np.sum((actual == labels[i]) & (predicted == labels[j]))
    return matrix

def get_accuracy(actual, predicted):
    return (actual == predicted).sum() / float(len(actual))


train = pd.read_csv('dataset/DT/breast-cancer-wisconsin-train.csv',na_values = '?').drop('Sample code number', axis=1)
test = pd.read_csv('dataset/DT/breast-cancer-wisconsin-test.csv',na_values = '?').drop('Sample code number', axis=1)

#fill missing data with mode
for column in train.columns:
    train[column].fillna(train[column].mode()[0], inplace=True)

for column in test.columns:
    test[column].fillna(train[column].mode()[0], inplace=True)

x_train = train.drop('Class', axis=1)
y_train = train['Class']

x_test = test.drop('Class', axis=1)
y_test = test['Class']

#train model
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

#get output for test set
y_pred = classifier.predict(x_test)

confusion_matrix = get_confusion_matrix(y_test,y_pred)
accuracy = get_accuracy(y_test,y_pred)
print(confusion_matrix)
print(accuracy)