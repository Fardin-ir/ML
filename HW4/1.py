import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#import data
data = pd.read_csv('datasets/1/parkinsons.data').sample(frac=1)
y = data['status'].values
x = data.drop(['status','name'], axis=1).values

#svm with linear kernel
def linear(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,shuffle = False)

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    return accuracy,f1

#svm with poly kernel
def poly(x,y,r,d):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,shuffle = False)

    svclassifier = SVC(kernel='poly',coef0=r,degree=d)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    return accuracy,f1

#svm with rbf kernel
def rbf(x,y,gamma):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,shuffle = False)

    svclassifier = SVC(kernel='rbf',gamma=gamma)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    return accuracy,f1

#svm with sigmoid
def sigmoid(x,y,r):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,shuffle = False)

    svclassifier = SVC(kernel='sigmoid',coef0=r)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    return accuracy,f1

#parameter arrays,diffrent values for parameters
gamma_arr = [0.001,0.01,0.1,1,10]
r_arr = [0,1,3,5,10]
d_arr = [2,4,8,10,12]


#create and print a table for linear-kernel svm
linear_table = pd.DataFrame([linear(x,y)], columns = ['accuracy', 'f1']) 
print('linear:')
print(linear_table)

#create and print a table for poly-kernel svm. each row contain diffrent parameters(r,d).
#loop through 'parameter arrays'(defined in line 62 to 64) with inline for-loop
poly_table = pd.DataFrame([[d,r]+list(poly(x,y,r,d)) for d in d_arr for r in r_arr], columns = ['d','r','accuracy', 'f1']) 
print('poly:')
print(poly_table)

#create and print a table for rbf-kernel svm. each row contain diffrent parameter(gamma).
#loop through 'parameter array'(defined in line 62 to 64) with inline for-loop
rbf_table = pd.DataFrame([[gamma] + list(rbf(x,y,gamma)) for gamma in gamma_arr], columns = ['gamma','accuracy', 'f1']) 
print('rbf:')
print(rbf_table)

#create and print a table for sigmoid-kernel svm. each row contain diffrent parameter(r).
#loop through 'parameter array'(defined in line 62 to 64) with inline for-loop
sigmoid_table = pd.DataFrame([[r] + list(sigmoid(x,y,r)) for r in r_arr], columns = ['r','accuracy', 'f1']) 
print('sigmoid:')
print(sigmoid_table)

#save results in excel file, 'openpyxl' is needed to run this part
#install openpyxl with command 'pip install openpyxl'
'''
with pd.ExcelWriter('output.xlsx') as writer:  
    linear_table.to_excel(writer,sheet_name='linear')  
    poly_table.to_excel(writer,sheet_name='poly')  
    rbf_table.to_excel(writer,sheet_name='rbf')  
    sigmoid_table.to_excel(writer,sheet_name='sigmoid')  
'''