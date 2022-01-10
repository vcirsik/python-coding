# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:20:49 2021

@author: virsik
"""


import os
os.chdir('d:\\Classification\\Week5')

import numpy as np
import pandas as pd


# load data as dataframe
loans = pd.read_csv("lending-club-data.csv") 

# recode loan code in column ['bad loans']; prev 1 bad 0 good
# safe_loans =  1 => safe  # safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)

# get rid of old unneeded column with prev loan coding
loans = loans.drop('bad_loans',1)

target = 'safe_loans'
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
             'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]
# get target and feature columns but drop nas
loans = loans[[target] + features].dropna()



import json

# get indices for training and validation set for classifier
with open("module-8-assignment-1-train-idx.json", "r") as read_file:
    train_idx = json.load(read_file)

with open("module-8-assignment-1-validation-idx.json", "r") as read_file:
    validation_idx = json.load(read_file)
    

# =============================================================================
#  one hot encoding w/ pandas
# =============================================================================

# limit to categorical data using df.select_dtypes()
tmpDat = loans.select_dtypes(include=[object])
tmpDat.head(3)

# get one hot encoding via get_dummies
oneH_tmpDat = pd.get_dummies(tmpDat, prefix=tmpDat.columns)

# remove nas replace with 0s
#oneH_tmpDat = oneH_tmpDat.fillna(0)

# get rid of old columns
loans = loans.drop(tmpDat.columns,1)

# append one hot coded data

loan_data = pd.concat([loans,oneH_tmpDat], axis=1)

# =============================================================================
# separate into training and validation data; convert to np 
# =============================================================================

train_data = loan_data.iloc[train_idx]
validation_data = loan_data.iloc[validation_idx]    

features= train_data.columns     
features = features.drop(target)  

# separate X and Y train dat & convert to np
train_X = train_data.drop(target,1).to_numpy()
train_y = train_data[target].to_numpy()

# separate X and Y train dat & convert to np
validation_X = validation_data.drop(target,1).to_numpy()
validation_y = validation_data[target].to_numpy()

import sklearn
from sklearn.ensemble import GradientBoostingClassifier

model_5 = GradientBoostingClassifier(n_estimators=5, max_depth=6).fit(train_X, train_y)


validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data

tPred = model_5.predict(sample_validation_data.drop(target,1))
#model_5.score(sample_validation_data.drop(target,1),sample_validation_data[target])

model_5.predict_proba(sample_validation_data.drop(target,1))

model_5.score(validation_X, validation_y)
# get number of false positives
rPred = model_5.predict(validation_risky_loans.drop(target,1))
falsePos = rPred != validation_risky_loans[target]
falsePos.sum()

# could also do this
# =============================================================================
# false_positives = ((predictions==1) * (validation_Y==-1)).sum()
# =============================================================================

predictions = model_5.predict(validation_X)
falseNeg = ((predictions == -1)*(validation_y == 1)).sum()
cost = 10000 * falseNeg.sum()  + 20000 * falsePos.sum() 
misttakes = predictions != validation_y
misttakes.sum()
misttakes.sum()/len(predictions)

# =============================================================================
# we will find the loans that are most likely to be predicted safe. We can do this in a few steps:
# =============================================================================


# =============================================================================
# Step 1: Use the model_5 (the model with 5 trees) and make probability predictions for all the loans in validation_data.
# =============================================================================
probPred = model_5.predict_proba(validation_X)

# =============================================================================
# Step 2: Similar to what we did in the very first assignment, add the probability predictions as a column called predictions into validation_data.
# =============================================================================

validation_data['probabilityofPos'] = probPred[:,1].reshape(len(probPred),1)

# =============================================================================
# Step 3: Sort the data (in descreasing order) by the probability predictions.
# =============================================================================
validation_data.sort_values('probabilityofPos', ascending = False)

# =============================================================================
# making more models w/ more trees
# =============================================================================
model_10 = GradientBoostingClassifier(n_estimators=10, max_depth=6).fit(train_X, train_y)
model_50 = GradientBoostingClassifier(n_estimators=50, max_depth=6).fit(train_X, train_y)
model_100 = GradientBoostingClassifier(n_estimators=100, max_depth=6).fit(train_X, train_y)
model_200 = GradientBoostingClassifier(n_estimators=200, max_depth=6).fit(train_X, train_y)
model_500 = GradientBoostingClassifier(n_estimators=500, max_depth=6).fit(train_X, train_y)

train_err_10  = 1 - model_10.score(train_X, train_y)
train_err_50  = 1 - model_50.score(train_X, train_y)
train_err_100 = 1 - model_100.score(train_X, train_y)
train_err_200  = 1 - model_200.score(train_X, train_y)
train_err_500 = 1 - model_500.score(train_X, train_y)

validation_err_10  = 1 - model_10.score(validation_X, validation_y)
validation_err_50  = 1 - model_50.score(validation_X, validation_y)
validation_err_100 = 1 - model_100.score(validation_X, validation_y)
validation_err_200  = 1 - model_200.score(validation_X, validation_y)
validation_err_500 = 1 - model_500.score(validation_X, validation_y)

import matplotlib.pyplot as plt
%matplotlib inline
def make_figure(dim, title, xlabel, ylabel, legend):
    plt.rcParams['figure.figsize'] = dim
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(loc=legend, prop={'size':15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    
training_errors = [train_err_10, train_err_50, train_err_100, train_err_200, train_err_500]   
validation_errors = [validation_err_10, validation_err_50, validation_err_100, validation_err_200, validation_err_500]   
# get training errors as a function of tree size
plt.plot([10, 50, 100, 200, 500], training_errors, linewidth=4.0, label='Training error')

# get validation errors as a function of tree size
plt.plot([10, 50, 100, 200, 500], validation_errors, linewidth=4.0, label='Validation error')

make_figure(dim=(10,5), title='Error vs number of trees',
            xlabel='Number of trees',
            ylabel='Classification error',
            legend='best')