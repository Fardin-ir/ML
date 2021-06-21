import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


data = pd.read_csv('Dataset/4/SeoulBikeData.csv')
y = data['Rented Bike Count']
X = data.drop('Rented Bike Count', axis=1)
feature_names = X.columns

le = preprocessing.LabelEncoder()
X['Seasons'] = le.fit_transform(X['Seasons'].values)
X['Holiday'] = le.fit_transform(X['Holiday'].values)
X['Functioning Day'] = le.fit_transform(X['Functioning Day'].values)
X['Date'] = le.fit_transform(X['Date'].values)

scaler = StandardScaler()
X = scaler.fit(X).transform(X)

#part B

X = pd.DataFrame(X,columns=feature_names)
#X.corr().to_excel("correlation.xlsx")  
print('features with correlation>0.55:')
high_corr = X.corr().where( X.corr()>0.5)
high_corr_features = high_corr[high_corr < 1].stack().index.tolist()
print(high_corr_features)


clf = LinearRegression().fit(X, y)
deleted_feature = ['None']
score = [clf.score(X, y)]
for x in high_corr_features:
    if x[0] in deleted_feature:
        continue
    deleted_feature.append(x[0])
    clf = LinearRegression().fit(X.drop(x[0], axis=1), y)
    score.append(clf.score(X.drop(x[0], axis=1), y))

print('Score of model after removing features')
print(pd.Series(data=score, index=deleted_feature))

r_X = X.copy()
X = X.drop('Dew point temperature', axis=1)
feature_names = X.columns

#remove more features by linear regression coefficients
clf = LinearRegression().fit(X, y)
importance = np.abs(clf.coef_)
feature_names = feature_names[importance.argsort()[::-1]]
importance = importance[importance.argsort()[::-1]]

#show Feature importances
plt.barh( feature_names,importance)
plt.title("Feature importances via coefficients (linear regression)")
plt.show()

#select k top features and calculate score
clf = LinearRegression().fit(X, y)
features = []
score = []
for i in range(len(feature_names)):
    feature_set = feature_names[:i+1]
    features.append(len(feature_set))
    clf = LinearRegression().fit(X[feature_set], y)
    score.append(clf.score(X[feature_set], y))
plt.plot(features,score)
plt.xticks(features)
plt.xlabel('k features with highest coefficients (linear regression)')
plt.ylabel('Score')
plt.show()

#print selected features
k=7
print('k selected features:', feature_names[:k])

#part C
print('part c ***************************')
X = r_X.copy()
feature_names = X.columns

score = []
for alpha in range(1,100):
    clf = Lasso(alpha=alpha).fit(X, y)
    score.append(clf.score(X, y))

plt.plot(range(1,100),score)
plt.xlabel('alpha')
plt.ylabel('Score')
plt.show()

clf = Lasso(alpha=30).fit(X, y)
importance = np.abs(clf.coef_)
plt.barh( feature_names,importance)
plt.title("Feature importances via coefficients (lasso)")
plt.show()
