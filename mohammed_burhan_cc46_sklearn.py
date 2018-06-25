# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:13:21 2018

@author: Lenovo
"""
import pandas as pd
data=pd.read_csv("Loan.csv")
features=data.iloc[:,:12].values

label=data.iloc[:,-1].values


from sklearn.model_selection import train_test_split
features_train,features_test,label_train,label_test = train_test_split(features,label,test_size=0.2,random_state=0)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
features[:,1]=labelencoder.fit_transform(features[:,1])

labelencoder=LabelEncoder()
features[:,2]=labelencoder.fit_transform(features[:,2])

labelencoder=LabelEncoder()
features[:,3]=labelencoder.fit_transform(features[:,3])

labelencoder=LabelEncoder()
features[:,4]=labelencoder.fit_transform(features[:,4])

labelencoder=LabelEncoder()
features[:,5]=labelencoder.fit_transform(features[:,5])

labelencoder=LabelEncoder()
features[:,11]=labelencoder.fit_transform(features[:,11])

labelencoder=LabelEncoder()
features[:,0]=labelencoder.fit_transform(features[:,0])


features_view=pd.DataFrame(features)

onehotencoder=OneHotEncoder(categorical_features=[11])
features=onehotencoder.fit_transform(features).toarray()

label=labelencoder.fit_transform(label)

