import pandas as pd

#Loading data
data=pd.read_csv('D:\heart.csv')

#Ceating feature and target data set
X=data[['age','sex','cp','trestbps','chol','thalach']].values
Y=data[['target']].values

print(data.dtypes)

#Creating training and testing data sets
from sklearn.model_selection import train_test_split
XTrain,XTest,YTrain,YTest=train_test_split(X,Y,test_size=0.2,random_state=4)
print('Shape of training set: ',XTrain.shape,YTrain.shape)
print('Shape of testing set: ',XTest.shape,YTest.shape)

#Preparing model
from sklearn import svm
Model=svm.SVC(kernel='rbf')
Model.fit(XTrain,YTrain)

#Using model for prediction
result=Model.predict(XTest)

#Evaluation of accuracy
from sklearn.metrics import jaccard_similarity_score
print('Jaccard similaity score of training set: ',jaccard_similarity_score(YTrain,Model.predict(XTrain)))
print('Jaccard similarity score of testing set: ',jaccard_similarity_score(YTest,result))

from sklearn.metrics import f1_score
print('F1 score of training set: ',f1_score(YTrain,Model.predict(XTrain),average='weighted'))
print('F1 score of testing set: ',f1_score(YTest,result,average='weighted'))