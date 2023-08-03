# Wine_Quality_Prediction
This model check the quality of wine on the basis of their ingredients like alcohol
# Wine Quality Prediction

#Importing required packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#loading the dataset
df = pd.read_csv('WineQT.csv')

#Let's check how the data is distributed


df.sample(5)

# check the dataset shape


df.shape

# decrive value of dataset 

df.describe()

##Information about the dtype columns

df.info()

# Check the dataset null values

df.isnull().sum()

# quality c description
df.describe()['quality']

df.drop(columns='Id',inplace = True)

## data columns quality correlation with the  other columns
df.corr()['quality']

#graph all the data set - just for looking

df.plot(figsize=(10,6))

sns.distplot(df['quality'])

#Here we see that fixed acidity does not give any specification to classify the quality.
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = df)

#Here we see that Volatile acidity does not give any specification to classify the quality.
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = df)

#Here we see that citric  acidity does not give any specification to classify the quality.
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = df)

df.head()

#Here we see that citric  acidity does not give any specification to classify the quality.
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = df)

# using graph interactive the show the effect free and total - sulfur dioxide in the quality

px.scatter(df, x="free sulfur dioxide", y="total sulfur dioxide",animation_frame="quality")

# Analysis result
We have 5 types of quality in DataSet - 3 to 8¶
The Best quality 8
The less quality 3 ### The elements highest effect on the quality of wine:
1 - Alcohol
2 - Free sulfur dioxide
3 - Total sulfur dioxide

# Buliding Machine learning model

#Importing the basic librarires for building model


from sklearn.linear_model import LinearRegression ,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error ,mean_squared_error, median_absolute_error,confusion_matrix,accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC ,SVR

#Defined X value and y value , and split the data train

X = df.drop(columns="quality")           
y = df["quality"]    # y = quality

X_train ,X_test,y_train , y_test = train_test_split(X ,y , test_size=0.25 , random_state = 42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

#Linear Model

## Linear Regression
linear = LinearRegression()
linear.fit(X_train,y_train)

print("the score of X_train with y_train :",linear.score(X_train , y_train))
print("the score of X_test with y_test :",linear.score(X_test , y_test))

# Expected value Y using X test
y_pred = linear.predict(X_test)

# Model evalution
print("predict the Mean absolute error:" , mean_absolute_error(y_test , y_pred))
print("predict the Mean Sqaured  error:" , mean_squared_error(y_test , y_pred))
print("predict the Median absolute error:" , median_absolute_error(y_test , y_pred))

## logistic Regression

logistic = LogisticRegression()
logistic.fit(X_train , y_train)

print("the score of X_train with y_train :",logistic.score(X_train , y_train))
print("the score of X_test with y_test :",logistic.score(X_test , y_test))

y_pred = logistic.predict(X_test)

# Model Evaluation
print( " Model Evaluation Logistic R : mean absolute error is ", mean_absolute_error(y_test,y_pred))
print(" Model Evaluation Logistic R : mean squared  error is " , mean_squared_error(y_test,y_pred))
print(" Model Evaluation Logistic R : median absolute error is " ,median_absolute_error(y_test,y_pred)) 

print(" Model Evaluation Logistic R : accuracy score " , accuracy_score(y_test,y_pred))

## decision Tree
Tree_model = DecisionTreeClassifier()

Tree_model.fit(X_train , y_train)

print("Score the X-train with Y-train is : ", Tree_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", Tree_model.score(X_test,y_test))

print("Tree model class",Tree_model.classes_)
y_pred = Tree_model.predict(X_test)

print(" Model Evaluation Decision Tree : accuracy score " , accuracy_score(y_test,y_pred))

## using model Random Forest Classifiers
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

#Let's see how our model performed
print(classification_report(y_test, pred_rfc))

# Stochastic Gradient Decent Classifier

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)

print(classification_report(y_test, pred_sgd))

# Support Vector Classifier

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)

print(classification_report(y_test, pred_svc))
