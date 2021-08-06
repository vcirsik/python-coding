# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 20:34:20 2021

@author: virsik
"""
# change directory
import os
path = ".\Documents\GitHub\python-coding\linear_regression"
os.chdir(path)

# import relavent libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt

# load data
birth = pd.read_csv('./data/Birthweight_reduced_R.csv')
birth.columns

# reduce to exclude irrelavent or redundant columns
# lowbwt and LowBirthWeight are transformations of the Birthweight column
birth = birth[['headcircumference', 'length', 'Birthweight', 'Gestation',
       'smoker', 'motherage', 'mnocig', 'mheight', 'mppwt', 'fage', 'fedyrs',
       'fnocig', 'fheight', 'mage35']]

# =============================================================================
# # normalize values
# def normalize(df):
#     result = df.copy()
#     for feature_name in df.columns:
#         max_value = df[feature_name].max()
#         min_value = df[feature_name].min()
#         result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
#     return result
# 
# birthNorm = normalize(birth)
# =============================================================================

# get X labels
predictorLabel = birth.columns.drop('Birthweight',1)

# get y 
y = birth["Birthweight"]

# get correlation plots
for ii in range(len(predictorLabel)):
    plt.figure()
    ax1 = plt.subplot(1,2,1)
    ax1.scatter(birth[predictorLabel[ii]],y)
    ax1.set_xlabel(predictorLabel[ii])
    ax1.set_ylabel('Birthweight')
    ax1 = None
    
 
# get correlations to isolate useful variables
corrMatrix = birth.corr()
corrPred   = corrMatrix['Birthweight'].drop('Birthweight')

# drop worst predictors
idxw = abs(corrPred)<.1
predictorLabel = predictorLabel[~idxw]

# determine best possible predictors
idxb = corrPred[predictorLabel]>0.5
vars_to_use = predictorLabel[idxb]

# pregenerate variables for storing results
record_of_columns = np.array([])
record_of_Rsq = np.array([])
record_of_cv = np.array([])

# remove used variables 
remainingVar = predictorLabel.drop(vars_to_use)

for ii in range(len(predictorLabel[~idxb])+1):
    
    
    # record your choice of columns (". "join(x) puts ", " in between each element of x and makes it a single string)
    record_of_columns = np.append(record_of_columns, ", ".join(vars_to_use))
    X = birth[vars_to_use]


    # Calculate adjusted R squared value 
    model1 = sm.OLS(y, sm.add_constant(X)).fit()
    Rsq = model1.rsquared_adj 
    record_of_Rsq = np.append(record_of_Rsq, Rsq)
    print("R squared =", Rsq)

    # Calculate CV score 
    LR_setup = LinearRegression()
    cv_scores = cross_val_score(LR_setup, X, y, cv = 5)
    cv = np.mean(cv_scores)
    record_of_cv = np.append(record_of_cv, cv)
    print("Cross validation =", cv)
    if ii < 6:
        vars_to_use = vars_to_use.append(pd.Index([remainingVar[ii]]))
# =============================================================================

# get summary
Sum = pd.DataFrame({"columns": record_of_columns, "Rsq": record_of_Rsq, "cv": record_of_cv})
for ii in range(len(Sum)):
    print(Sum['columns'])
