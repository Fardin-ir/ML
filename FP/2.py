import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix

def purity_score(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true,y_pred)
    return np.sum(np.amax(conf_matrix, axis=0)) / np.sum(conf_matrix) 

colormap = plt.cm.gist_ncar 

#db_scan for 2-d datasets
def db_scan(eps,min_samples,dataset):
    d1 = np.loadtxt(f'Dataset/2/{dataset}.txt')
    X = d1[:,:2]
    y = d1[:,2]
    unique_labels = np.unique(y)
    #maps each label to a color
    colors = [colormap(i) for i in np.linspace(0, 1,len(unique_labels)+1)]  
    #plot data set with real labels
    labels = y
    fig, ax = plt.subplots()
    for i in range(len(unique_labels)):
        idxs = np.where(labels == unique_labels[i])
        ax.scatter(X[idxs,0], X[idxs,1], c=colors[i], label=unique_labels[i], s=10)
    plt.title(f'Compound.txt real labels ')
    ax.legend()
    plt.show()

    #plot clusters
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    unique_labels = np.unique(clustering.labels_)
    colors = [colormap(i) for i in np.linspace(0, 1,len(unique_labels)+1)] 
    print(unique_labels)
    labels = clustering.labels_
    fig, ax = plt.subplots()
    for i in range(len(unique_labels)):
        idxs = np.where(labels == unique_labels[i])
        ax.scatter(X[idxs,0], X[idxs,1], c=colors[i], label=unique_labels[i], s=10)
    plt.title(f'{dataset} Clustering eps={eps} min_samples={min_samples}, purity={purity_score(y,labels):.4f}')
    ax.legend()
    plt.show()


#call main function for each 2-d dataset
db_scan(1.4,4,'Compound')

#db_scan(0.46,5,'D31')

#db_scan(2.5,20,'pathbased')

#db_scan(3,3,'spiral')


#db_scan for 3-d datasets
def db_scan_3d(eps,min_samples,dataset):
    d1 = np.loadtxt(f'Dataset/2/{dataset}.txt')
    X = d1[:,1:4]
    y = d1[:,0]
    unique_labels = np.unique(y)
    colors = [colormap(i) for i in np.linspace(0, 1,len(unique_labels)+1)]  
    print(len(colors))
    print(unique_labels)
    labels = y
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(unique_labels)):
        idxs = np.where(labels == unique_labels[i])
        ax.scatter(X[idxs,0], X[idxs,1], X[idxs,2], c=colors[i], label=unique_labels[i], s=10)
    plt.title(f'Compound.txt Labels ')
    ax.legend()
    plt.show()

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    unique_labels = np.unique(clustering.labels_)
    colors = [colormap(i) for i in np.linspace(0, 1,len(unique_labels)+1)] 
    print(unique_labels)
    labels = clustering.labels_
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(unique_labels)):
        idxs = np.where(labels == unique_labels[i])
        ax.scatter(X[idxs,0], X[idxs,1], X[idxs,2], c=colors[i], label=unique_labels[i], s=10)
    plt.title(f'{dataset} Clustering eps={eps} min_samples={min_samples} , purity={purity_score(y,labels):.4f}')
    ax.legend()
    plt.show()

#db_scan_3d(5,3,'rings')