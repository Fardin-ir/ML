import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

#part_b
data = pd.read_csv('Dataset/1/1-2/Shill Bidding Dataset.csv')
X = data.drop(['Record_ID','Class','Auction_ID','Bidder_ID'],axis=1)


sse = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, random_state=0,init='random').fit(X)
    sse.append(kmeans.inertia_)

plt.plot(range(1,11),sse)
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.xticks(range(1,11))
plt.show()

#part-c
def purity_score(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true,y_pred)
    return np.sum(np.amax(conf_matrix, axis=0)) / np.sum(conf_matrix) 

kmeans = KMeans(n_clusters=2, random_state=0,init='random').fit(X)
print('purity:',purity_score(data['Class'],kmeans.labels_))
