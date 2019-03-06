import pandas as pd

#Loading data
data=pd.read_csv('D:\heart.csv')
print(data.columns)

#Preparing feature and target sets
X=data[['age','sex','cp','trestbps','chol','thalach']].values
Y=data[['target']].values

#Normalizing feature set
from sklearn import preprocessing
X=preprocessing.StandardScaler().fit(X).transform(X)

#Creating training and testing sets
from sklearn.model_selection import train_test_split
XTrain,XTest,YTrain,YTest=train_test_split(X,Y,test_size=0.2,random_state=4)
print('Size of training set: ',XTrain.shape,YTrain.shape)
print('Size of testing set: ',XTest.shape,YTest.shape)

#Creating Model
from sklearn.linear_model import LogisticRegression
Reg=LogisticRegression(C=0.01,solver='liblinear').fit(XTrain,YTrain)

#Using model for prediction
result=Reg.predict(XTest)
result_prob=Reg.predict_proba(XTest)
print(result_prob[0:5])

#Evaluating accuracy of the model
from sklearn.metrics import jaccard_similarity_score
print('Jaccard similarity score of testing set: ',jaccard_similarity_score(YTest,result))
print('Jaccard similarity score of training set: ',jaccard_similarity_score(YTrain,Reg.predict(XTrain)))