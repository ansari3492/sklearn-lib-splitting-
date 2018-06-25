# -*- coding: utf-8 -*-
"""
Created on Tue May 29 12:01:48 2018

@author: Lenovo
"""

import pandas as pd
data=pd.read_csv("Loan.csv")
data=data.drop('Loan_ID',axis=1)

features=data.drop("Target",axis=1)
labels = data["Target"]

for i in features:
    if features[i].dtype==object:
        features[i] = features[i].astype('category')
        features[i] = features[i].cat.codes
    
labels = labels.astype('category')
labels = labels.cat.codes


features = pd.get_dummies(features, columns=["Property_Area"])
    
features = features.values
labels = labels.values

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.2,random_state=0)
