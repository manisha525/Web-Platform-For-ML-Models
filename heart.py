import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
import joblib
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import svm
warnings.filterwarnings("ignore")

#Read Dataset
data = pd.read_csv("heart.csv")
data = np.array(data)  #Change into numpy array

#Values of X & Y
X = data[0:, 0:-1]
y = data[0:, -1]

y = y.astype('int')
X = X.astype('float')

print(X)
print(y)


#Splitting into Training and Testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

#Using different models
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
pred1 = log_reg.predict(X_test)
print("predictions= ",pred1)
print('The accuracy of the Logistic Regression is = ',accuracy_score(pred1,y_test))

rf=RandomForestClassifier()
rf.fit(X_train,y_train)
pred2 = rf.predict(X_test)
print("predictions= ",pred2)
print('The accuracy of the Random Forest is = ',accuracy_score(pred2,y_test))

supvec=svm.SVC(kernel='linear')
supvec.fit(X_train,y_train)
pred3=supvec.predict(X_test)
print("predictions= ",pred3)
print('The accuracy of the svm is',accuracy_score(pred3, y_test))

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
pred4=dt.predict(X_test)
print("predictions= ",pred4)
print('The accuracy of the Decision Tree is',accuracy_score(pred4,y_test))

knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
pred5=knn.predict(X_test)
print("predictions= ",pred5)
print('The accuracy of the KNeighbors is',accuracy_score(pred5,y_test))


joblib.dump(log_reg,"heart")
# pickle.dump(supvec,open('forest.pkl','wb'))