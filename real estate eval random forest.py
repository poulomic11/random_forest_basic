import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"F:\python new\real estate eval\Real estate valuation data set.csv")
#dataset = pd.read_csv(r"F:\python new\teacher assistant evaluation details\teacher eval data.csv")

#print(dataset.head(5))

import sklearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler


#real estate
X = dataset.iloc[:, [1,2,3,4,5,6]].values
Y = dataset.iloc[:, [7]].values

#teacher eval
#X = dataset.iloc[:, [0,1,2,3,4]].values
#Y = dataset.iloc[:, [5]].values



X_train , X_test , Y_train , Y_test = train_test_split(dataset , Y , test_size = 0.3 )


print(Y_test.head())

#Y_train = np.reshape(Y_train , (17,17))
#Y_test = np.reshape(Y_test , (25,5))
#X_train = np.reshape(X_train , (289,8))
#X_test = np.reshape(X_test , (125,8))


sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)



lab_enc = preprocessing.LabelEncoder()
Y_train = lab_enc.fit_transform(Y_train)
Y_test = lab_enc.fit_transform(Y_test)



RF = RandomForestRegressor()
RF.fit(X_train,Y_train)

Y_pred_test = RF.predict(X_test)
Y_pred_train = RF.predict(X_train)

Y_pred_test = np.reshape(Y_pred_test , (25,5))
Y_pred_train = np.reshape(Y_pred_train , (17,17))




#print(Y_test[0:5])

#print("Training accuracy: " , RF.score(Y_train,Y_pred_train))
#print("\nTesting accuracy: " , RF.score(Y_test,Y_pred_test))

#print(confusion_matrix(Y_train,Y_pred_train))

#print(metrics.accuracy_score(X_test,Y_test))