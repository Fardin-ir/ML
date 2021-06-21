import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
data = pd.read_csv('Dataset/Dataset2.csv')
'''
plt.plot(data['x'],data['y'], 'ro', markersize=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('')
plt.show()
'''
#show that dataset1 need shuffling

def train_test_data(data,frac):
    n = math.floor(frac*data.shape[0])
    train_data = data.iloc[:n,:]
    test_data = data.iloc[n:,:]
    test_data.to_csv('Dataset/test.csv') 
    return train_data,test_data

train, test = train_test_data(data,0.7)

plt.plot(train['x'],train['y'], 'ro', markersize=2)
plt.xlabel('X')
plt.ylabel('Y')

plt.plot(test['x'],test['y'], 'bo', markersize=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()