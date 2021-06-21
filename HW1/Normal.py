import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math 

#import shuffled data
data = pd.read_csv('Dataset/s_data.csv')

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

    return h,error_train,error_test,X_train



#call main function
h4,error_train4,error_test4,X_train = normal_equation(4,data)
h8,error_train8,error_test8,X_train = normal_equation(8,data)
h12,error_train12,error_test12,X_train = normal_equation(12,data)


norm_data,mean,std = normalize(data,data['x'].mean(),np.std(data['x']))
#plot
plt.plot(norm_data['x'],norm_data['y'], 'ro',markersize=2)
plt.plot(np.sort(X_train['x']),h4,color="#008B8B", label="degree: 4")
plt.plot(np.sort(X_train['x']),h8,color="#FF4500", label="degree: 8")
plt.plot(np.sort(X_train['x']),h12,color="#4B0082", label="degree: 12")
plt.xlabel('Normalized X')
plt.ylabel('Y')
plt.title(f'Normal Equation')
plt.legend(loc='lower right')
plt.show()

errors = {
    'degree':[4,8,12],
    'Error test':[error_test4,error_test8,error_test12],
    'Error train':[error_train4,error_train8,error_train12]
}
error_table = pd.DataFrame(errors,columns=['degree','Error test','Error train'])
print(error_table)
 

