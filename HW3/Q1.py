import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
data = pd.read_csv('dataset/s_car.csv')
labels1 = data['values'].unique()
#colomn number of labels
m=7


#splite data to train and test set
def train_test_data(data,frac):
    n = math.floor(frac*data.shape[0])
    train_data = data.iloc[:n,:]
    test_data = data.iloc[n:,:]
    return train_data,test_data

#returns confusion matrix
def get_confusion_matrix(actual, predicted,labels):
    #create labels x labels dataframe 
    matrix = pd.DataFrame(np.zeros((len(labels), len(labels))),index=labels,columns=labels)
    #(i,j) element of matrix is where actual lable is 'i' but predicted label is 'j'
    for i in labels:
        for j in labels:
            matrix.loc[i,j] = np.sum((actual == i) & (predicted == j))
    return matrix

#calculate metrics (one vs all)
#((labels)) is list of all posible labels 
def metrics(actual, predicted,labels):
    metrics = []
    for label in labels:
        new_predicted = predicted.copy()
        new_actual = actual.copy()
        new_predicted[new_predicted!=label] = 'other'
        new_actual[new_actual!=label] = 'other'
        conf_matrix = get_confusion_matrix(new_actual, new_predicted,[label,'other'])
        fp = conf_matrix.loc['other',label]
        fn = conf_matrix.loc[label,'other']
        sensitivity = conf_matrix.loc[label,label]/(conf_matrix.loc[label,label]+fn)
        specificity = conf_matrix.loc['other','other']/(conf_matrix.loc['other','other']+fp)
        metrics.append([sensitivity,specificity,fn,fp])
    return pd.DataFrame(metrics, columns = ['sensitivity','specificity','fn','fp'], index=labels) 

#return set of fpr and tpr for plot roc curves
def roc(outputs,probs):
    fp_arr=[]
    tp_arr=[]
    s_probs = np.sort(probs)
    for prob in s_probs:
        pred = []
        for j in range(len(outputs)):
            if probs[j] >= prob:
                pred.append(1)
            else:
                pred.append(0)
        conf_matrix = get_confusion_matrix(outputs.values, np.asarray(pred),[1,0])

        fp = conf_matrix.loc[0,1]/(conf_matrix.loc[0,1]+conf_matrix.loc[0,0])
        tp = 1-conf_matrix.loc[1,0]/(conf_matrix.loc[1,0]+conf_matrix.loc[1,1])
        fp_arr.append(fp)
        tp_arr.append(tp)
    return fp_arr,tp_arr


#calculate likelihood p(feature=value|y=output) given dataset (train_set)
def get_likelihood(train_set,feature,value,output,laplace):
    k = len(train_set[feature].unique())
    try:
        count = train_set.loc[train_set['values'] == output][feature].value_counts()[value] 
    except KeyError:
        count = 0
    likelihood = (count+int(laplace))/(train_set['values'].value_counts()[output]+int(laplace)*k)
    return likelihood

#calculate posterior (y=output|data)
def posterior(train_set,data,output,laplace):
    prior = train_set['values'].value_counts()[output]/len(train_set)
    likelihood=1
    for col, value in data.items():
        #print(col)
        likelihood *= get_likelihood(train_set,col,data[col],output,laplace)
    return likelihood*prior

#classifay data with npc
#returns posteriors for each label and label which maximize posterior
def nbc(train_set,data,laplace):
    labels = train_set['values'].unique()
    posteriors = []
    for label in labels:
        #print(label,' dsfdfssddfs')
        posteriors.append(posterior(train_set,data,label,laplace))
    pred_output = labels[posteriors.index(max(posteriors))]
    posteriors.append(pred_output)
    return posteriors

#main function
#classifay all data in dataset and returns a pandas data_frame
#each row of data_frame contains posteriors and pred_label for a data of test_set
def nbc_set(train,test,laplace):
    pred = []
    labels = train['values'].unique()
    temp = test.drop('values',axis=1)
    for index, row in temp.iterrows():
        pred.append(nbc(train,row,laplace))
    pred = pd.DataFrame(pred, columns = np.append(labels,'output')) 
    return pred

#splite dataset
train,test = train_test_data(data,0.7)
#call main function
predicted = nbc_set(train,train,False)
print(get_confusion_matrix(train['values'].values,predicted['output'].values,labels1))
print('metrics fo train')
print(metrics(train['values'].values,predicted['output'].values,labels1))
#plot roc curves for each label(output)

predicted = nbc_set(train,test,False)
#print confusion_matrix and metrics
print(get_confusion_matrix(test['values'].values,predicted['output'].values,labels1))
print('metrics for test')
print(metrics(test['values'].values,predicted['output'].values,labels1))
print('***************************************8')



#part3

fig, axs = plt.subplots(2, 2)
fig.tight_layout()
labels1 = data['values'].unique()
for i, label in enumerate(labels1):
    output = test['values'].copy()
    output[output==label]=1
    output[output!=1]=0
    probs = predicted[label]
    fp,tp = roc(output,probs)
    axs[int(i/2),i%2].plot(fp,tp)
    axs[int(i/2),i%2].set_title(f'{label} vs others')

plt.show()
