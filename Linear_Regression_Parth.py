#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 00:45:50 2021

@author: Parth Patel
"""
import numpy as np
import pandas as pd
import os
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)


# load data 
#--------------------------------------

path = "/Users/sambhu/iCloud Drive (Archive) - 1/Desktop/centennial college /sem-1/intoduction to AI/assignment-3"
filename = 'Ecom Expense.csv'
fullpath = os.path.join(path,filename)
ecom_exp_Parth  = pd.read_csv(fullpath)

# data analysis 
#--------------------------------------
 
print(ecom_exp_Parth.head(3))
print("------------------------------------------")
print(ecom_exp_Parth.shape)
print("------------------------------------------")
print(ecom_exp_Parth.columns.values)
print("------------------------------------------")
print(ecom_exp_Parth.dtypes)
print("------------------------------------------")
#print(ecom_exp_Parth.isna())
col=ecom_exp_Parth.columns.values.tolist()
l=len(col)
for i in range (l):
    missing_val=ecom_exp_Parth[col[i]].isna().sum()
    if (missing_val>0):
        print ("column name:"+col[i]+"missing values:"+missing_val)
    else :
        print ("0")
        
# Data Cleaning
#--------------------------------------

# i : make dummies and join with dataframe 
col1=['Gender','City Tier']
for var in col1:
    cat_list='var'+'_'+var
    print(cat_list)
    cat_list = pd.get_dummies(ecom_exp_Parth[var], prefix=var)
# ii. 	Attach the newly created variables 
    ecom_exp_Parth1=ecom_exp_Parth.join(cat_list)
    ecom_exp_Parth=ecom_exp_Parth1
    
    
# iii & iv	: Remove the original categorical variables columns. 
ecom_exp_Parth_f=ecom_exp_Parth1.drop(columns=['Transaction ID', 'Gender', 'City Tier'])   
print(ecom_exp_Parth_f.shape)
    
# v : normalization function 
def normalization(df):
       df = df.copy()
   
       for column in df.columns:
           df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        
       return df

# vi :	Call the new function and pass as an argument 
ecom_exp_Parth_norm=normalization(ecom_exp_Parth_f)
ecom_exp_Parth_norm.columns.values

# vii : vii.	Display (print) the first two records.
print(ecom_exp_Parth_norm.head(2))

# viii : generate a plot showing all the variables histograms
#plt.hist(ecom_exp_Parth_norm)
ecom_exp_Parth_norm.hist(figsize=(9,10))


# ix : generate a plot illustrating the relationships between :   'Age ','Monthly Income','Transaction Time','Total Spend'
cat_vars=['Age ','Monthly Income','Transaction Time','Total Spend']
ecom_exp_Parth_norm_vars=ecom_exp_Parth_norm.columns.values.tolist()
to_keep=[i for i in ecom_exp_Parth_norm_vars if i in cat_vars]
ecom_exp_Parth_norm_final=ecom_exp_Parth_norm[to_keep]
ecom_exp_Parth_norm_final.columns.values

pd.plotting.scatter_matrix(ecom_exp_Parth_norm_final,alpha=0.4,figsize=(13,15))




# req: d. Build a model
#--------------------------------------

# i: linear relationship between the output variable and the predictor variables  

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

feature_cols1 = ['Monthly Income', 'Transaction Time','Gender_Female', 'Gender_Male', 'City Tier_Tier 1',
       'City Tier_Tier 2', 'City Tier_Tier 3']
X = ecom_exp_Parth_norm[feature_cols1]
Y = ecom_exp_Parth_norm['Total Spend']
print(Y.head())
print(X.head())
print(X.shape)

# ii & iv : “train test split” 
x_train_Parth,x_test_Parth,y_train_Parth,y_test_Parth = train_test_split(X,Y, test_size = 0.35)

# iii.	Set the seed 
np.random.seed(43)

# v.	 Using sklearn fit a linear regression model to the training data.
lm1 = LinearRegression()
lm1.fit(x_train_Parth,y_train_Parth)

# vi. the coefficients 
print (lm1.coef_)
# vii. the model score 
print (lm1.score(x_train_Parth,y_train_Parth))

print(lm1.predict(x_test_Parth))
print(y_test_Parth)


# viii. add  the feature ‘Record’ to the list of predictors 
feature_cols2 = ['Monthly Income', 'Transaction Time','Record','Gender_Female', 'Gender_Male', 'City Tier_Tier 1',
       'City Tier_Tier 2', 'City Tier_Tier 3']
X = ecom_exp_Parth_norm[feature_cols2]
Y = ecom_exp_Parth_norm['Total Spend']
print(Y.head())
print(X.head())

#  “train test split” 
x_train_Parth,x_test_Parth,y_train_Parth,y_test_Parth = train_test_split(X,Y, test_size = 0.35)

# 	Set the seed 
np.random.seed(43)

# 	 Using sklearn fit a linear regression model to the training data.
lm2 = LinearRegression()
lm2.fit(x_train_Parth,y_train_Parth)

# ix the coefficients 
print (lm2.coef_)
# x the model score 
print (lm2.score(x_train_Parth,y_train_Parth))

print(lm2.predict(x_test_Parth))
print(y_test_Parth)

 
































