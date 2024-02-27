# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:56:53 2024

@author: urmii
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
iris=datasets.load_iris()
X,y=iris.data[:,1:3],iris.target
clf1=LogisticRegression()
clf2=RandomForestClassifier(random_state=1)
clf3=GaussianNB()

print("After five fold creoss validation")
labels=['Logisitc regression','random Forest model','Naive Bayes model']
for clf,label in zip([clf1,clf2,clf3],labels):
    score=model_selection.cross_val_score(clf,X,y,cv=5,scoring='accuracy')
    print('Accuracy:',score.mean(),'for ',label)

voting_clf_hard=VotingClassifier(estimators=[(labels[0],clf1),
                                             (labels[1],clf2),
                                             (labels[2],clf3)],
                                             voting='hard'
                                             )

voting_clf_soft=VotingClassifier(estimators=[(labels[0],clf1),
                                             (labels[1],clf2),
                                             (labels[2],clf3)],
                                             voting='soft'
                                             )
labels_new=['Logisitc Regression','Random Forest model','Naive Bayes Model','Voting Hard','Voting Soft']
for clf,label in zip([clf1,clf2,clf3,voting_clf_hard,voting_clf_soft],labels_new):
    score=model_selection.cross_val_score(clf,X,y,cv=5,scoring='accuracy')
    print('Accuracy:',score.mean(),'for ',label)



