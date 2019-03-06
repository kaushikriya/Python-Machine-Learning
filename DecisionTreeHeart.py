import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#Loading Data
data=pd.read_csv('D:heart.csv')
print(data.columns)

#Preparing feature and target data sets
X=data[['age','sex','cp','trestbps','chol','thalach']].values
Y=data[['target']].values


#Preparing Testing and Training data sets
from sklearn.model_selection import train_test_split
XTrain,XTest,YTrain,YTest=train_test_split(X,Y,test_size=0.1,random_state=4)
print('Shape of training set: ',XTrain.shape,YTrain.shape)
print('Shape of testing set: ',XTest.shape,YTest.shape)

#Preparing Model
tree=DecisionTreeClassifier(criterion='entropy',max_depth=6)
tree.fit(XTrain,YTrain)

#Predicting targets using the model
result=tree.predict(XTest)

#Evaluating accuracy of the model
from sklearn.metrics import accuracy_score
print('Accuracy of the testing set: ',accuracy_score(YTest,result))
print('Accuracy of the training set: ',accuracy_score(YTrain,tree.predict(XTrain)))