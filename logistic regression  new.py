import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ds=pd.read_csv('Social_Network_Ads.csv')
x=ds.iloc[:,[2,3]].values
y=ds.iloc[:,4].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
xtrain=scx.fit_transform(xtrain)
xtest=scx.transform(xtest)

#logistic regression
from sklearn.linear_model import LogisticRegression
c= LogisticRegression(random_state=0)
c.fit(xtrain,ytrain)
ypred=c.predict(xtest)
#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(xtrain, ytrain)
ypred1=knn.predict(xtest)

#naive bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(xtrain,ytrain)
ypred2=nb.predict(xtest)

#decision trees
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy',random_state=0)
dtc.fit(xtrain,ytrain)
ydtc=dtc.predict(xtest)

#random forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
rfc.fit(xtrain,ytrain)
yrfc=rfc.predict(xtest)

#svm
from sklearn.svm import SVC
sv=SVC(kernel='linear',random_state=0)
sv.fit(xtrain,ytrain)
ysc=sc.predict(xtest)

#confusion mtrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
cm1=confusion_matrix(ytest,ypred1)
cm2=confusion_matrix(ytest,ypred2)


#graph
 