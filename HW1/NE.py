import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math 

#import shuffled data
data = pd.read_csv('Dataset/a_data.csv')

#returns training and test set
def train_test_data(data,frac):
    n = math.floor(frac*data.shape[0])
    train_data = data.iloc[:n,:]
    test_data = data.iloc[n:,:]
    return train_data,test_data
    
#normalize column'x' of input matrix and returns it, also returns normalization parametrs
def normalize(mat,mean,std):
    x = mat['x']
    x_norm = (x - mean) / (std)
    mat['x'] = x_norm
    return mat, mean,std

#add columns to input matrix according to n and return it
def extend_data(n,x):
    for i in range(2,n+1):
        temp_arr = []
        for j in range(x.shape[0]):
            temp = pow(x.iloc[j,0],i)
            temp_arr.append(temp)
        x.insert(i-1,f'x{i}',temp_arr)
    x.insert(0,'x0',[1]*x.shape[0])

    return x

#main function of this program
def normal_equation(n,data):
    #splite data to training and test set 
    train_data, test_data = train_test_data(data,0.7)
    #parameters for normalization
    mean = train_data['x'].mean()
    std = np.std(train_data['x'])
    
    #normlize train data(x) and separate x and y,
    #also create design matrix X_train with 'extend_data' function
    X_train,mean,std = normalize(train_data.iloc[:,:1],mean,std)
    X_train = extend_data(n,X_train)
    Y_train = train_data['y']

    #do the same thing for test data
    #note that we normalize test data with 'train data parameters', as it should be
    X_test,mean,std = normalize(test_data.iloc[:,:1],mean,std)
    X_test = extend_data(n,X_test)
    Y_test = test_data['y']

    #normal equation
    newtheta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T.dot(Y_train))
    print('theta =',newtheta)

    #final hypothesis vector 
    h = X_train.sort_values(['x']).dot(newtheta)

    #calculate error for both train and test set
    predicted = X_test.dot(newtheta)
    def error(h,y):
        loss = np.subtract(h,y)
        return np.sum(np.power(loss,2))/len(h)

    error_test = "{:.5f}".format(error(predicted,Y_test))
    error_train = "{:.5f}".format(error(h,Y_train))

    #normalize all data to show them
    norm_data,mean,std = normalize(data,data['x'].mean(),np.std(data['x']))
    #plot
    plt.plot(norm_data['x'],norm_data['y'], 'ro',markersize=2)
    plt.plot(np.sort(X_train['x']),h)
    plt.xlabel('Normalized X')
    plt.ylabel('Y')
    plt.title(f'plot for degree={n}')
    plt.table(cellText=[[error_train],[error_test]],rowLabels=['error train','error test'],loc='best', colWidths=[0.25,0.25])

    plt.show()


#call main function
normal_equation(10,data)