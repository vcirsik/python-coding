# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 21:50:23 2021

@author: virsik

"""
# =============================================================================
# 
# # this script fits three decision tree models which predict 
# # loan safety and compares performance between models.
# # script created as part of assignment from classification course.
# # this code will be optimized in DecTree_LoanPred_optimized.py" 
# 
# =============================================================================



import numpy as np
import pandas as pd


# load data as dataframe
loans = pd.read_csv("data\\lending-club-data.csv") 

# recode loan code in column ['bad loans']; prev 1 bad 0 good
# safe_loans =  1 => safe  # safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)

# get rid of  column with prev loan coding
loans = loans.drop('bad_loans',1)

loanDesc = loans['safe_loans'].value_counts()/len(loans)
print('The proportion of safe loans:' + str(loanDesc.iloc[0].round(2)))
print('The proportion of unsafe loans:' + str(loanDesc.iloc[1].round(2)))


# extract features we will use
features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]


import json

# get indices for training and validation set for classifier
with open("data\\module-5-assignment-1-train-idx.json", "r") as read_file:
    train_idx = json.load(read_file)

with open("data\\module-5-assignment-1-validation-idx.json", "r") as read_file:
    validation_idx = json.load(read_file)


# =============================================================================
#  one hot encoding w/ pandas
# =============================================================================

# limit to categorical data using df.select_dtypes()
X = loans.select_dtypes(include=[object])
X.head(3)

# get one hot encoding via get_dummies
oneH_X = pd.get_dummies(X, prefix=X.columns)

# remove nas replace with 0s
oneH_X = oneH_X.fillna(0)

# get rid of old columns
loans = loans.drop(X.columns,1)

# append one hot coded data

loan_data = pd.concat([loans,oneH_X], axis=1)

# =============================================================================
# separate into training and validation data; convert to np 
# =============================================================================

train_data = loan_data.iloc[train_idx]
validation_data = loan_data.iloc[validation_idx]


# separate X and Y train dat & convert to np
train_X = train_data.drop(target,1).to_numpy()
train_y = train_data[target].to_numpy()


# separate X and y data for validation set & convert to np
val_X = validation_data.drop(target,1).to_numpy()
val_y = validation_data[target].to_numpy()


# =============================================================================
#  build and run decision tree classifier
# =============================================================================


from sklearn.tree import DecisionTreeClassifier

# build base dec tree model
DecTreeModel = DecisionTreeClassifier(max_depth=6)
DecTreeModel = DecTreeModel.fit(train_X, train_y)

# build small dec tree model for comparison
small_model = DecisionTreeClassifier(max_depth=2)
small_model = small_model.fit(train_X, train_y)

# get mean accuracy for both models: TRAIN DATA
# =============================================================================
DecTreeModel.score(train_X, train_y)
small_model.score(train_X, train_y)

# =============================================================================
# use models to get predictions with validation data
# =============================================================================
pred_DTM = DecTreeModel.predict(val_X)

pred_smMod = small_model.predict(val_X)

# =============================================================================
# # get reports
# =============================================================================

# getting pred accuracy by hand:
Pred_acc =  (val_y == pred_DTM).sum()/len(val_y)
print('Prediction accuracy for the larger model is :' + str(Pred_acc))

Pred_acc_sm =  (val_y == pred_smMod).sum()/len(val_y)
print('Prediction accuracy for the small model is :' + str(Pred_acc_sm))



# getting detailed reports using sklearn: Base Model
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(val_y,pred_DTM))
print(confusion_matrix(val_y, pred_DTM))

# getting detailed reports using sklearn: Smaller Model
print(classification_report(val_y,pred_smMod))
print(confusion_matrix(val_y, pred_smMod))


# =============================================================================
# generate very large/complex model 
# =============================================================================
Max_model = DecisionTreeClassifier(max_depth=10)
Max_model = Max_model.fit(train_X, train_y)


# get pred accuracy of training set
Max_model.score(train_X, train_y)

# use model to get predictions of validation set 
pred_MaxMod = Max_model.predict(val_X)

# getting pred accuracy by hand:
Pred_acc_MM =  (val_y == pred_MaxMod).sum()/len(val_y)
print('Prediction accuracy for the validation set & MAX model is :' + str(Pred_acc_MM))
# =============================================================================

