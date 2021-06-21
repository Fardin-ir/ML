import numpy as np
import pandas as pd
import math
import time
start_time = time.time()

df = pd.read_csv('dataset\knn\mammographic_masses.csv', na_values = '?')

#number of features
m=5

#fills missing values with mode
for column in df.columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

#normalizes 'column' col of matrix 'mat' with given parameters
def normalize(mat,col,mean,std):
    x = mat[col]
    x_norm = (x - mean) / (std)
    mat[col] = x_norm
    return mat

#returns mode of array 'x'
def mode1(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m]

#gets number of folds and returns 'i'th fold as test-set and remaining data as train-fold
def train_test_data(data,fold,i):
    sub_arrays = np.array_split(data,fold)
    test = sub_arrays[i]
    sub_arrays = np.delete(sub_arrays,i)
    train = pd.concat(sub_arrays)
    return train, test

#distance methods
def euclidean_distance(x,y):
    return np.linalg.norm(x - y)

def manhatan_distance(x,y):
    return np.sum(np.abs(x-y))

def cosine_distance(x,y):
    return 1-np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

#sorts data of matrix 'data' based on their distance from point 'x', distance is calculated with 'distance_method'
def get_neighbors_sorted(data,x,distance_method):
    global m
    z = np.zeros((len(data),1), dtype='int64')
    data = np.append(data, z, axis=1)
    for row in range(len(data)):
        data[row,m+1] = distance_method(x,data[row,:m])
    return data[data[:,-1].argsort()] 

#gets train and test set;number of nearest neighbors as 'k' and 'distance method'
#predicts a label for each data of test-set and add it to a new colomn of test-set
#then returns test-set
def knn(train,test,distance_method,k):
    global m
    #convert data_frame to numpy matrix to speed-up calculations
    train = train.values
    test = test.values
    #add new column for predicted data to test-set
    z = np.zeros((len(test),1), dtype='int64')
    test = np.append(test, z, axis=1)
    #predict label for each point in test-set and return it
    for row in range(len(test)):
        x = test[row][:m]
        nearest_neighbors = get_neighbors_sorted(train,x,distance_method)[:k]
        test[row][m+1] = mode1(nearest_neighbors[:,m])
    return test


#returns confusion matrix
def get_confusion_matrix(actual, predicted):
    #find number of classes
    labels = np.unique(actual)
    #create labels x labels dataframe 
    matrix = pd.DataFrame(np.zeros((len(labels), len(labels))))
    #(i,j) element of matrix is where actual lable is 'i' but predicted label is 'j'
    for i in range(len(labels)):
        for j in range(len(labels)):
            matrix.iloc[i, j] = np.sum((actual == labels[i]) & (predicted == labels[j]))
    return matrix

#returns accuracy
def get_accuracy(actual, predicted):
    return (actual == predicted).sum() / float(len(actual))

#main function
#gets data,number of folds for k-fold cross validation, distance method and number of nearest neighbors
#return sum of confusion matrixes and mean of accuracy
def cross_validation(data,num_fold,distance_method,k):
    #initial confusion_matrix
    labels = np.unique(data['Severity']) 
    confusion_matrix = pd.DataFrame(np.zeros((len(labels), len(labels))))
    #initial accuracy
    accuracy = 0
    for i in range(num_fold):
        print('fold:',i)
        #define train and test set
        train,test = train_test_data(df,num_fold,i)
        #normalize column 'age' train and test set with train-set mean and std
        mean,std = train['Age'].mean(),train['Age'].std()
        train = normalize(train,'Age',mean,std)
        test = normalize(test,'Age',mean,std)
        #call knn for each fold as test-set
        test = knn(train,test,distance_method,k)
        accuracy += get_accuracy(test[:,m],test[:,m+1])
        confusion_matrix += get_confusion_matrix(test[:,m],test[:,m+1])
    accuracy /= num_fold
    return confusion_matrix,accuracy

#call main function
#cross_validation(data,num_fold,distance_method,k)
num_fold=10
print(cross_validation(df,num_fold,manhatan_distance,15))
        
print("--- %s seconds ---" % (time.time() - start_time))