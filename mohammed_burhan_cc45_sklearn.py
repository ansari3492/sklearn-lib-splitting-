# -*- coding: utf-8 -*-
"""
Created on Tue May 29 13:25:16 2018

@author: Lenovo
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder ,  Imputer
data=pd.read_csv("Automobile.csv")
data=pd.DataFrame(data)
features=data.drop("price",axis=1).values
labels = data["price"]






imputer=Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
imputer= imputer.fit(features[:,1:2])
features[:,1:2]=imputer.transform(features[:,1:2])





labelencoder=LabelEncoder()
features[:,7]=labelencoder.fit_transform(features[:,7])



new_data=pd.get_dummies(data, columns=[ "drive_wheels","body_style"])



