import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import  VotingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

data = pd.read_csv('Dataset/5/heart_failure_clinical_records_dataset.csv', na_values='?')
y = data['DEATH_EVENT']
X = data.drop('DEATH_EVENT', axis=1)
columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)
X_train = pd.DataFrame(X_train, columns=columns)
X_test = pd.DataFrame(X_test, columns=columns)



continuous_features = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium']
discrete_variable = ['anaemia','diabetes','high_blood_pressure','sex','smoking','time']

imp = SimpleImputer(strategy="most_frequent")
X_train[discrete_variable] = imp.fit_transform(X_train[discrete_variable])
X_test[discrete_variable] = imp.transform(X_test[discrete_variable])

imp = SimpleImputer(strategy="mean")
X_train[continuous_features] = imp.fit_transform(X_train[continuous_features])
X_test[continuous_features] = imp.transform(X_test[continuous_features])

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
model = SelectFromModel(lsvc, prefit=True)
print(columns[model.get_support()])

X_train = model.transform(X_train)
X_test = model.transform(X_test)

clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('gnb', clf3)], voting='hard')
eclf.fit(X_train,y_train)
print('Accuracy of voting model 1:', eclf.score(X_test,y_test))


clf1 = KNeighborsClassifier(n_neighbors=5)
clf2 = RandomForestClassifier()
clf3 = AdaBoostClassifier(random_state=0)
eclf = VotingClassifier(estimators=[('knn', clf1), ('rf', clf2), ('ab', clf3)], voting='soft')
eclf.fit(X_train,y_train)
print('Accuracy of voting model 2:', eclf.score(X_test,y_test))

clf1 = KNeighborsClassifier(n_neighbors=5)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = KNeighborsClassifier(n_neighbors=9)
eclf = VotingClassifier(estimators=[('3nn', clf1), ('5nn', clf2), ('7nn', clf3)], voting='hard')
eclf.fit(X_train,y_train)
print('Accuracy of voting model 3:', eclf.score(X_test,y_test))

print('**********************************')
clf1 = KNeighborsClassifier(n_neighbors=5)
clf2 = RandomForestClassifier()
clf3 = AdaBoostClassifier(n_estimators=100, random_state=0)
print('Accuracy of KNeighborsClassifier:', clf1.fit(X_train,y_train).score(X_test,y_test))
print('Accuracy of RandomForestClassifier:', clf2.fit(X_train,y_train).score(X_test,y_test))
print('Accuracy of AdaBoostClassifier:', clf3.fit(X_train,y_train).score(X_test,y_test))
