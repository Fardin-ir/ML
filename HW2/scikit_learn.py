import numpy as np
import pandas as pd
import math
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

start_time = time.time()

df = pd.read_csv('dataset\knn\mammographic_masses.csv', na_values = '?')

for column in df.columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

age=df['Age']
df['Age']=(age-age.mean())/age.std()

X = df.drop(columns=['Severity'])
y = df['Severity'].values


knn = KNeighborsClassifier(n_neighbors =30)


y_pred = cross_val_predict(knn, X, y, cv=10)
conf_mat = confusion_matrix(y,y_pred)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores.mean())
print(conf_mat)

print("--- %s seconds ---" % (time.time() - start_time))