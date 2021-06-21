import numpy as np
import pandas as pd
import math


df = pd.read_csv('dataset/Regression/regression.csv', na_values = '?')

m=5

#normalizes all column matrix 'mat' with given parameters
def normalize(mat,mean,std):
    print(mean)
    for col in std:
        x = mat[col]
        x_norm = (x - mean[col][0]) / (std[col][0])
        mat.loc[:,col] = x_norm
    return mat
#returns mse error
def error(real,predicted):
    loss = np.subtract(real,predicted)
    return np.sum(np.power(loss,2))/len(real)

#returns mode of array 'x'
def mode1(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m]

#splite data to train and test set
def train_test_data(data,frac):
    n = math.floor(frac*data.shape[0])
    train_data = data.iloc[:n,:]
    test_data = data.iloc[n:,:]
    return train_data,test_data

def euclidean_distance(x,y):
    return np.linalg.norm(x - y)

#sorts data of matrix 'data' based on their distance from point 'x', distance is calculated with 'distance_method'
def get_neighbors_sorted(data,x,distance_method):
    global m
    z = np.zeros((len(data),1), dtype='int64')
    data = np.append(data, z, axis=1)
    for row in range(len(data)):
        data[row,m+1] = distance_method(x,data[row,:m])
    return data[data[:,-1].argsort()] 

#gets train and test set;number of nearest neighbors as 'k' and 'distance method'
#predicts output for each data of test-set and add it to a new colomn of test-set
#then returns test-set
def knn(train,test,distance_method,k):
    global m
    #convert data_frame to numpy matrix to speed-up calculations
    train = train.values
    test = test.values
    #add new column for predicted output to test-set
    z = np.zeros((len(test),1), dtype='int64')
    test = np.append(test, z, axis=1)
    #predict output for each point in test-set and return it
    for row in range(len(test)):
        x = test[row][:m]
        nearest_neighbors = get_neighbors_sorted(train,x,distance_method)[:k]
        test[row][m+1] = np.mean(nearest_neighbors[:,m])
    return test


#main function
#gets data, distance method and number of nearest neighbors
#return mse error for test and train test
def reg(data,distance_method,k):
    #create train and test set
    train,test = train_test_data(df,0.7)
    #normalize all column of train and test(except price) with train-set parameters
    #mean = train.drop('Price',axis=1).mean().to_frame().T
    #std = train.drop('Price',axis=1).std().to_frame().T
    #train = normalize(train,mean,std)
    #test = normalize(test,mean,std)

    #get predicted output for both train and test set
    new_train = knn(train,train,distance_method,k)
    new_test = knn(train,test,distance_method,k)
    
    error_train = error(new_train[:,m],new_train[:,m+1])
    error_test = error(new_test[:,m],new_test[:,m+1])
    return error_test, error_train

print(reg(df,euclidean_distance,19))
        
