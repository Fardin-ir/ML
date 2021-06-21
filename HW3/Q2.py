import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
from numpy import random

labels = [0,1,2,3,4,5,6,7,8,9]
x_train = pd.read_csv('dataset/mnist_train.csv')
y_train = x_train['label']
x_train = x_train.drop('label',axis=1)
x_test = pd.read_csv('dataset/mnist_test.csv')
y_test = x_test['label']
x_test = x_test.drop('label',axis=1)

#returns confusion matrix
def get_confusion_matrix(actual, predicted,labels):
    #create labels x labels dataframe 
    matrix = pd.DataFrame(np.zeros((len(labels), len(labels))),index=labels,columns=labels)
    #(i,j) element of matrix is where actual lable is 'i' but predicted label is 'j'
    for i in labels:
        for j in labels:
            matrix.loc[i,j] = np.sum((actual == i) & (predicted == j))
    return matrix
#return error for given lists
def get_error(actual, predicted):
    return 1-(np.asarray(actual) == np.asarray(predicted)).sum() / float(len(actual))
#main funnction
#return error and predicted label
def one_vs_all(train,train_label,test,test_label):
    train= train.copy()
    train_label= train_label.copy()
    test=test.copy()
    test_label=test_label.copy()
    labels = [0,1,2,3,4,5,6,7,8,9]
    predicted_probs = []
    #one vs all
    #set first class label 1 and other 0, predict prob with LR
    #repeat for other classes
    for label in labels:
        new_train_label = train_label.copy()
        new_test_label = test_label.copy()
        new_train_label[new_train_label==label] = 1
        new_train_label[new_train_label!=1] = 0
        new_test_label[new_test_label==label] = 1
        new_test_label[new_test_label!=1] = 0
        lr = LogisticRegression(random_state=0).fit(train.values, new_train_label.values)
        predicted_probs.append(np.asarray(lr.predict_proba(test.values))[:,1])
 
    predicted_probs = pd.DataFrame(predicted_probs, index=labels) 
    pred_labels = []
    for col in predicted_probs:
        pred_labels.append(predicted_probs[col].idxmax())
    pred_labels = np.asarray(pred_labels)
    error = get_error(test_label.values, pred_labels)
    return pred_labels,error

#call main function

pred,error = one_vs_all(x_train,y_train,x_train,y_train)
print('confusion_matrix for train:')
print(get_confusion_matrix(y_train.values,pred,labels))
print('error for train:')
print(error)

pred,error = one_vs_all(x_train,y_train,x_test,y_test)
print('confusion_matrix for test:')
print(get_confusion_matrix(y_test.values,pred,labels))
print('error for test:')
print(error)

# part 2
fig, axs = plt.subplots(5, 5)
fig.tight_layout()
for i in range(5):
    for j in range(5):
        x = random.randint(9999)
        axs[i,j].imshow(x_test.iloc[x,:].values.reshape(28,28),cmap='gray')
        y_actual = y_test.values[x]
        y_pred = pred[x]
        axs[i,j].set_title(f'y_actual={y_actual},y_pred={y_pred}',size=10)
plt.show()





