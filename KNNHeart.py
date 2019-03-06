import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#Loading data
data=pd.read_csv('D:\heart.csv')
print(data.head(9))
print(data.columns)

#Preparing data sets
X=data[['age','sex','trestbps','chol','thalach','oldpeak']]

#Preparing dependent data set
Y=data[['thal']]

#Normalizing the data
X=preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])

#Creating data sets for training and testing
from sklearn.model_selection import train_test_split
XTrain,XTest,YTrain,YTest=train_test_split(X,Y,test_size=0.2,random_state=4)
print('Testing set :', XTrain.shape,YTrain.shape)
print('Testing set: ',XTest.shape,YTest.shape)

#Creating model
from sklearn.neighbors import KNeighborsClassifier
neigh=KNeighborsClassifier(n_neighbors=4).fit(XTrain,YTrain)

#Predicting values
result=neigh.predict(XTest)

# Evaluating predicted values

from sklearn.metrics import accuracy_score
print('Training test accuracy: ',accuracy_score(YTrain,neigh.predict(XTrain)))
print('Testing set accuracy: ',accuracy_score(YTest,result))

#Plotting predictions
plt.plot(YTest,'ro')
plt.plot(result,'bo')
plt.show()




