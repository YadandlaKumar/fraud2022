import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#Loading the Data set
dataset = pd.read_excel("C:\\Deployment_model\\New folder\\OnlineBetting.xlsx")

dataset.head()
dataset.info()

import dtale

dtale.show(dataset)
#from sklearn.preprocessing import LabelEncoder

#dataset["Country"] = LabelEncoder().fit_transform(dataset["Country"])
#dataset = pd.get_dummies(dataset, columns = ["Country"], drop_first = True)
#dataset["GENDER"] = LabelEncoder().fit_transform(dataset["GENDER"])
dataset.drop("Customer_ID", inplace = True, axis = 1)
dataset.drop("odds1", inplace = True, axis = 1)
#dataset = pd.get_dummies(dataset, columns = ["Mode_Of_Payment"], drop_first = True)
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.3, variables = ["Amount_Spent"])
dataset["Amount_Spent"] = winsor.fit_transform(dataset[["Amount_Spent"]])
X = dataset.drop(["Fraud"], axis = 1)
X.head(), X.shape
y = dataset.Fraud
y.value_counts()

# Accumulate all the column names under one variable




from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_test.shape, X_train.shape

from sklearn.ensemble import RandomForestClassifier
Rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
Rfc.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = Rfc.predict(X_test)
accuracy_score(y_test, y_pred)
y_pred_tr = Rfc.predict(X_train)
accuracy_score(y_train, y_pred_tr)
confusion_matrix(y_test, y_pred)
hyper = RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=20,criterion='gini')
hyper.fit(X_train, y_train)

print('Train accuracy: {}'.format(hyper.score(X_train, y_train)))
print('Test accuracy: {}'.format(hyper.score(X_test, y_test)))

import pickle

pickle.dump(hyper,open('model1.pkl' , 'wb'))
model1 = pickle.load(open('model1.pkl','rb'))
import xgboost as xgb
xgbost_model = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)
xgbost_model.fit(X_train, y_train)
accuracy_score(y_test, xgbost_model.predict(X_test))
accuracy_score(y_train, xgbost_model.predict(X_train))
import pickle

pickle.dump(xgbost_model,open('xgbost_model.pkl' , 'wb'))
model1 = pickle.load(open('xgbost_model.pkl','rb'))
result = model1.score(X_test, y_test)
print(result)
