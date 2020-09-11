import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\Poulomi\Desktop\python new\real estate eval")
#dataset = pd.read_csv(r"F:\python new\teacher assistant evaluation details\teacher eval data.csv")




import sklearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
#from skmultilearn.model_selection import iterative_train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import classification_report,accuracy_score, roc_auc_score 
from sklearn.preprocessing import StandardScaler
import statistics 



#X = dataset.iloc [:, [1,2,3,4,5,6]]
#Y = dataset.iloc [:, [7]]

X = dataset.iloc[:, [0,1,2,3,4]].values
Y = dataset.iloc[:, [5]].values

#X = pd.read_csv(r"F:\python new\real estate eval\Real estate valuation data set.csv")
#Y = X.pop("Y_house_price_of_unit_area")



#y_oob = RF.oob_prediction_
#print(y_oob)

#print(RF.feature_importances_)
#f_imp = pd.Series(RF.feature_importances_ , index = X.columns )
#f_imp = f_imp.sort_values()
#f_imp.plot(kind = 'bar' , figsize = (7,6))
#plt.show()

X_train , X_test , Y_train , Y_test = train_test_split(X , Y ,  test_size = 0.2 )
X_train , X_val , Y_train , Y_val = train_test_split(X_train , Y_train , test_size = 0.2)

print("X_train shape: " , X_train.shape)
print("X_test shape: " , X_test.shape)                   
print("Y_train shape: " , Y_train.shape)                
print("Y_test shape: " , Y_test.shape)                   
print("Y_val shape: " , Y_val.shape)

lab_enc = preprocessing.LabelEncoder()
Y_train = lab_enc.fit_transform(Y_train)
Y_test = lab_enc.fit_transform(Y_test)
Y_val = lab_enc.fit_transform(Y_val)

RF = RandomForestClassifier(n_estimators=300 , oob_score= True)
RF.fit(X_train , Y_train)

pred1 = RF.predict(X_test)
pred2 = RF.predict(X_val)
#prob = RF.predict_proba(X_test)[:, 1]
print("Pred1 shape: " , pred1.shape)
print("Pred2 shape: " , pred2.shape)

print("First test data:")

for i in range(0, 5):
        print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(Y_test)[i], pred1[i]))
#print(RF.oob_score_)


print ("Train Accuracy :: ", accuracy_score(Y_train, RF.predict(X_train))*100,"%")
print ("Test Accuracy  :: ", accuracy_score(Y_test, pred1)*100,"%")
#print (" Confusion matrix ", confusion_matrix(Y_test, pred))

#rav = roc_auc_score(Y_test , prob)
#print(rav)

print("Second test data")

for i in range(0, 5):
        print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(Y_val)[i], pred2[i]))
#print(RF.oob_score_)


print ("Train Accuracy :: ", accuracy_score(Y_train, RF.predict(X_train))*100,"%")
print ("Test Accuracy  :: ", accuracy_score(Y_val, pred2)*100,"%")

 



