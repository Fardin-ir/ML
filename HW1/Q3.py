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


#calculate error function according to theta 
def cost(X,Y,theta):
    j = np.sum((X.dot(theta)-Y)**2)/(2*X.shape[0])
    return j

#gradient descent
def gradient_descent(X, Y, theta, alpha, landa,iterations):
    cost_history = [0] * iterations
    m=X.shape[0]
    #convert pandas dataframe to numpy matrix, it increase computing speed
    X_np = X.values
    Y_np = Y.values
    #main loop, we use matrix operations 
    for iteration in range(iterations):
        h = X_np.dot(theta)
        gradient = X_np.T.dot(h-Y_np)*1/(m)
        theta = theta - alpha*(gradient + landa/m*theta)
        cost_history[iteration] = cost(X,Y,theta)
        print(theta)
        #exit loop when convergence
        if cost_history[iteration] - cost_history[iteration-1] == 0:
            break
    return theta, cost_history


#main function of this program
def linear_regression(alpha, landa,iterations,n,data):
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

    theta = np.zeros(X_train.shape[1])
    newtheta, cost_history = gradient_descent(X_train, Y_train, theta, alpha, landa,iterations)
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

    return h,error_test,error_train,X_train,newtheta

iteration = 2000
#call main function for each degree
h1,error_test1,error_train1,X_train,newtheta1 = linear_regression( 1,0,iteration,1,data)
h2,error_test2,error_train2,X_train,newtheta2 = linear_regression( 0.7,0,iteration,2,data)
h4,error_test4,error_train4,X_train,newtheta4 = linear_regression( 0.1,0,iteration,4,data)
h6,error_test6,error_train6,X_train,newtheta6 = linear_regression( 0.028,0,iteration,6,data)
h8,error_test8,error_train8,X_train,newtheta8 = linear_regression( 0.0041,0,iteration,8,data)
#normalize all data to show them as points
norm_data,mean,std = normalize(data,data['x'].mean(),np.std(data['x']))
#plot
plt.plot(norm_data['x'],norm_data['y'], 'ro',markersize=2)
plt.plot(np.sort(X_train['x']),h1,color="#006400", label="degree:1, alpha:1")
plt.plot(np.sort(X_train['x']),h2,color="#008B8B", label="degree:2, alpha:0.7")
plt.plot(np.sort(X_train['x']),h4,color="#FF4500", label="degree:4, alpha:0.1")
plt.plot(np.sort(X_train['x']),h6,color="#4B0082", label="degree:6, alpha:0.028")
plt.plot(np.sort(X_train['x']),h8,color="#000000", label="degree:8, alpha:0.0041")
plt.legend(loc='lower right')
plt.xlabel('Normalized X')
plt.ylabel('Y')
plt.title(f'{iteration} iterations')
plt.show()

#create a table for error and save it as .csv
errors = {
    'degree':[1,2,4,6,8],
    'Error test':[error_test1,error_test2,error_test4,error_test6,error_test8],
    'Error train':[error_train1,error_train2,error_train4,error_train6,error_train8]
}
error_table = pd.DataFrame(errors,columns=['degree','Error test','Error train'])
print(error_table)
error_table.to_csv('error_table.csv') 
