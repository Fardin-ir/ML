import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#import data
data = pd.read_csv('datasets/2/pima_indians_diabetes.csv')
y = data['class'].values
x = data.drop(['class'], axis=1).values

#main function
def rand_forest(x,y,n_estimators,max_features,max_depth):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, shuffle=False)
    clf = RandomForestClassifier(max_depth=max_depth,max_features=max_features,n_estimators=n_estimators)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train,y_train)
    test_acc = clf.score(X_test,y_test)
    return train_acc,test_acc

#parameter arrays,diffrent values for parameters
n_estimators_arr = [10,100,1000]
max_features_arr = [2,5,8]
max_depth_arr = [2,5,8]

#create and print a table for random forest. each row contain diffrent parameters.
#loop through 'parameter arrays' with inline for-loop
acc_table = pd.DataFrame([[ne,mf,md]+list(rand_forest(x,y,ne,mf,md)) for ne in n_estimators_arr for mf in max_features_arr for md in max_depth_arr], columns = ['n_estimators','max_features','max_depth','train accuracy','test_accuracy']) 
print(acc_table)

#save results in excel file, 'openpyxl' is needed to run this part
#install openpyxl with command 'pip install openpyxl'
'''
acc_table.to_excel('output2-1.xlsx')  
'''