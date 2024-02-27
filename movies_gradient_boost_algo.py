# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:07:03 2024

@author: urmii
"""

import pandas as pd
df=pd.read_csv('C:/10-ML/Gradient_Boosting_Algo/movies_classification.csv')
df.head()
df.info()
#n-1 dummy variables will be created for n categories
df=pd.get_dummies(df,columns=['3D_available','Genre'],drop_first=True)
df.head()
#input and output split

predictors=df.loc[:,df.columns!='Start_Tech_Oscar']
type(predictors)

target=df['Start_Tech_Oscar']
type(target)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(predictors,target,test_size=0.2,random_state=0)

from sklearn.ensemble import GradientBoostingClassifier

boost_clf=GradientBoostingClassifier()
boost_clf.fit(x_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix

confusion_matrix(y_test,boost_clf.predict(x_test))
accuracy_score(y_test,boost_clf.predict(x_test))

#Hperparameters
boost_clf2=GradientBoostingClassifier(learning_rate=0.02,n_estimators=1000,max_depth=1)
boost_clf2.fit(x_train,y_train)

#evaluation on testing set
from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,boost_clf2.predict(x_test))
accuracy_score(y_test,boost_clf2.predict(x_test))

#evaluation on training set
accuracy_score(y_train,boost_clf2.predict(x_train))





