# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:57:27 2021

@author: admin
"""
import csv
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('naivedata.csv')

x = np.array(data.iloc[:,0:-1])
y = np.array(data.iloc[:,-1])  

print(data.head())

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(x,y)

#Predict Output
predicted= model.predict([[7,148,78,35,0,34,0.625,54]])
print("Predicted Value:", predicted)