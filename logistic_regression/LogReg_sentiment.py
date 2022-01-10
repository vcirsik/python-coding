# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:49:01 2021

@author: virsik
"""
# =============================================================================
# This script builds two logistic regression models (one simple, one complex)
# and compares accuracy. Script created as part of assignment from classification course.
# =============================================================================

# import os
# os.chdir('C:\\Users\\virsik\\Documents\\GitHub\\python-coding\\logistic_regression')

import numpy as np
import pandas as pd
products = pd.read_csv("data\\amazon_baby.csv") 

# define function to remove punctuation
def remove_punctuation(text):
    import string
    return text.translate(str.maketrans('','',string.punctuation))

# convert to str and remove punctuation
products['review'] = products['review'].astype(str)
products['review_clean'] = products['review'].apply(remove_punctuation)

# get rid of N/As
products = products.fillna({'review':''})  # fill in N/A's in the review column

# ignore all reviews with rating = 3, since they tend to have a neutral sentiment.
products = products[products['rating'] != 3]

# recode ratings to +1 -1
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

# reset index [only doing this because of weirdness with matching up with assigned test/training set]
products = products.reset_index()

import json
# get indices for train and test 
with open("data\\module-2-assignment-test-idx.json", "r") as read_file:
    test_vals = json.load(read_file)

with open("data\\module-2-assignment-train-idx.json", "r") as read_file:
    train_vals = json.load(read_file)

# separate data for train and test
test_data  = products.drop(labels = train_vals, axis = 0)
train_data = products.drop(labels = test_vals, axis = 0)

# =============================================================================
# logistic regression: complex model
# =============================================================================
from sklearn.feature_extraction.text import CountVectorizer

 # Use this token pattern to keep single-letter words
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    
# assign columns and count presence of unique word in training set
# then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'])

# =============================================================================
# # this section plots the word count results
# =============================================================================
# vectorizer.fit(train_data['review_clean'])
# Printing the identified Unique words along with their indices
# print("Vocabulary: ", vectorizer.vocabulary_)
# =============================================================================

# convert test data into a sparse matrix using same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])

from sklearn.linear_model import LogisticRegression

# set model parameters
sentiment_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

# do logistic reg
sentiment_model.fit(train_matrix, train_data['sentiment'])

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true=train_data['sentiment'].to_numpy(), y_pred=sentiment_model.predict(train_matrix))
print("Complex Model Train Accuracy: %s" % accuracy)

accuracy = accuracy_score(y_true=test_data['sentiment'].to_numpy(), y_pred=sentiment_model.predict(test_matrix))
print("Complex Test Accuracy: %s" % accuracy)






# =============================================================================
# # get predicted scores from full test set: manual
# FullTscores = sentiment_model.decision_function(test_matrix)
# 
# # get predictons 
# print(sentiment_model.predict(test_matrix))
# 
# # calculate probabilities
# tSetprob = 1./(1+np.exp(-FullTscores))
# print(tSetprob)
# print(tSetprob> .5)*1
# 
# get predictions of test data using trained model [0 or 1]
# print(sentiment_model.predict(test_matrix))

# get probability predictions for full test set
# fullModel = sentiment_model.predict_proba(test_matrix)
# test_data['probabilites'] = fullModel[:,1]
# =============================================================================

# =============================================================================
# logistic regression: simple model
# =============================================================================

# define words for use
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']

# get sparse matrix of unique word count
vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])



simple_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

simple_model.fit(train_matrix_word_subset, train_data['sentiment'])


simple_model_coef_table = pd.DataFrame({'word':significant_words,
                                         'coefficient':simple_model.coef_.flatten()})
# plot words with corresponding coeff
simple_model_coef_table.sort_values('coefficient', ascending  = 0)


complex_Tr_accuracy = accuracy_score(y_true=train_data['sentiment'].to_numpy(), y_pred=sentiment_model.predict(train_matrix))
print("Complex Model Train Accuracy: %s" % complex_Tr_accuracy)

simpMod_Tr_accuracy = accuracy_score(y_true=train_data['sentiment'].to_numpy(), y_pred=simple_model.predict(train_matrix_word_subset))
print("Simple Model Train Accuracy: %s" % simpMod_Tr_accuracy)

complex_Test_accuracy = accuracy_score(y_true=test_data['sentiment'].to_numpy(), y_pred=sentiment_model.predict(test_matrix))
print("Complex Model Test Accuracy: %s" % complex_Test_accuracy)

simpMod_Test_accuracy = accuracy_score(y_true=test_data['sentiment'].to_numpy(), y_pred=simple_model.predict(test_matrix_word_subset))
print("Simple Model Test Accuracy: %s" % simpMod_Test_accuracy)

# compare models with majority class classifier
positive_label = len(test_data[test_data['sentiment']>0])
negative_label = len(test_data[test_data['sentiment']<0])
print("# positive cases is {}, # negative cases is {}".format(positive_label, negative_label))


majorC_Classifer = positive_label/(positive_label + negative_label)



if majorC_Classifer > complex_Test_accuracy:
    print("Bad news: Model is less accurate than majority classifier")
else:
    print('Complex Model is successful: accuracy = {}'.format(complex_Test_accuracy))
        

if majorC_Classifer > simpMod_Test_accuracy:
    print("Bad news: Simple model is less accurate than majority classifier")
else:
    print('Simple Model is successful: accuracy = {}'.format(simpMod_Test_accuracy))
        