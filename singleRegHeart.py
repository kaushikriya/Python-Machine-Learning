import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn

data=pd.read_csv('D:\heart.csv')
col=data[['age','chol']]
# Plotting data
plt.scatter(col.age,col.chol,color='red')
plt.xlabel('Age')
plt.ylabel('Chol')
plt.show()

#Creating datasets for training and testing
mask=np.random.rand(len(data))<0.8
train=col[mask]
test=col[~mask]

#Creating model for linear regression
from sklearn import linear_model
reg=linear_model.LinearRegression()
train_x=np.asanyarray(train[['age']])
train_y=np.asanyarray(train[['chol']])
reg.fit(train_x,train_y)
print(reg.coef_)
print(reg.intercept_)

#Plotting data with our model
plt.scatter(col.age,col.chol,color='blue')
plt.plot(train_x,train_x*reg.coef_[0][0]+reg.intercept_[0],'-r')
plt.xlabel('age')
plt.ylabel('chol')
plt.show()

#Testing the model
from sklearn.metrics import r2_score
test_x=np.asanyarray(test[['age']])
test_y=np.asanyarray(test[['chol']])
result=reg.predict(test_x)

# calculating metrics
from sklearn.metrics import r2_score
print('r2_score score of testing set: ',r2_score(test_y,result))
print('r2_score similarity score of training set: ',r2_score(train_y,reg.predict(train_x)))