# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:26:34 2020

@author: sarav
"""


import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

#import dataset
dataset = pd.read_csv("height_weight.csv")

x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,1].values


#missing value in array
np.isnan(x).sum()
np.isnan(y).sum()

#impot imputer and fill the null values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN" , strategy = "mean" ,axis =0)
x = imputer.fit_transform(x)

#check if the issue is fixed
np.isnan(x).sum

#update : fixing missing vales
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan , strategy = "mean")
x = imp.fit_transform(x)

#cheack
np.isnan(x).sum()

#split dataset as train and test
from sklearn.model_selection import train_test_split
x_train , x_test , y_train  , y_test = train_test_split(x,y,test_size=.30 ,random_state = 0)

#building model
from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(x_train , y_train)

#predict

y_pred = regress.predict(x_test)

#visulise train data

plt.title("Height and Weight Train data")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.scatter(x_train , y_train ,color = "red")
plt.plot(x_train , regress.predict(x_train) , color = "blue")
plt.show()

#visulise test data

plt.title("Height and Weigh  Test data")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.scatter(x_test , y_test ,color = "red")
plt.plot(x_test, regress.predict(x_test) , color = "blue")
plt.show()


my_ht = [[172]]
my_wt_pred = regress.predict(my_ht)

print ("Score = ",regress.score(x_train, y_train))

