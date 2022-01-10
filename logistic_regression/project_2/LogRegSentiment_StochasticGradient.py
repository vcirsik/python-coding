# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 15:29:59 2021

@author: virsik
"""

# =============================================================================
# 
# # this script examines the efficacy of stochastic gradient descent.
# # script created as part of assignment from classification course.
#  
# =============================================================================



# import relevant libraries
import os
os.chdir('C:\\Users\\virsik\\Documents\\GitHub\\python-coding\\logistic_regression\\project_2')
import numpy as np
import pandas as pd
import json

# load data
products = pd.read_csv("data//amazon_baby_subset.csv") 

# get rid of NAs
products = products.fillna({'review':''})  # fill in N/A's in the review column

# define function to remove punctuation
def remove_punctuation(text):
    import string
    return text.translate(str.maketrans('','',string.punctuation))


products['review'] = products['review'].astype(str)
products['review_clean'] = products['review'].apply(remove_punctuation)


# get words for classifier
with open("data//important_words.json", "r") as read_file:
    important_words = json.load(read_file)
# count number of important words in poduct reviews

for word in important_words:
 products[word] = products['review_clean'].apply(lambda s : s.split().count(word))
 


# get indices for training and validation set for classifier
with open("data//module-10-assignment-train-idx.json", "r") as read_file:
    train_idx = json.load(read_file)

with open("data//module-10-assignment-validation-idx.json", "r") as read_file:
    validation_idx = json.load(read_file)

# separate into training and validation data
train_data = products.iloc[train_idx]
validation_data = products.iloc[validation_idx]

# reset index to avoid setting with copy warning [i added this; not assigned by course]
train_data = train_data.reset_index()
validation_data = validation_data.reset_index()

def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    feature_matrix = features_frame.to_numpy()
    label_sarray = dataframe[label]
    label_array = label_sarray.to_numpy()
    return(feature_matrix, label_array)

feature_matrix_train, sentiment_train = get_numpy_data(train_data, important_words, 'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(validation_data, important_words, 'sentiment')


def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients  
    score = np.dot(feature_matrix,coefficients)
    
    # Compute P(y_i = +1 | x_i, w) using the link function
    predictions = 1./(1 + np.exp(-score))
    
    # return predictions
    return(predictions)


def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    derivative = np.dot(feature,errors)
    
    # Return the derivative
    return derivative

def compute_avg_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]   
    
    lp = np.sum((indicator-1)*scores - logexp)/len(feature_matrix)   
    
    return lp


def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores)))
    return lp

# get gradient for one data point [stochastic]
j = 1                        # Feature number
i = 10                       # Data point number
coefficients = np.zeros(194) # A point w at which we are computing the gradient.

predictions = predict_probability(feature_matrix_train[i:i+1,:], coefficients)
indicator = (sentiment_train[i:i+1]==+1)

errors = indicator - predictions
gradient_single_data_point = feature_derivative(errors, feature_matrix_train[i:i+1,j])
print("Gradient single data point: %s" % gradient_single_data_point)
print("           --> Should print 0.0")


# get gradient using batch of data points [stochastic]
j = 1                        # Feature number
i = 10                       # Data point start
B = 10                       # Mini-batch size
coefficients = np.zeros(194) # A point w at which we are computing the gradient.

predictions = predict_probability(feature_matrix_train[i:i+B,:], coefficients)
indicator = (sentiment_train[i:i+B]==+1)

errors = indicator - predictions
gradient_mini_batch = feature_derivative(errors, feature_matrix_train[i:i+B,j])
print("Gradient mini-batch data points: %s" % gradient_mini_batch)
print("                --> Should print 1.0")



def logistic_regression_SG(feature_matrix, sentiment, initial_coefficients, step_size, batch_size, max_iter):
    log_likelihood_all = []

    # make sure it's a numpy array
    coefficients = np.array(initial_coefficients)
    
    # set seed=1 to produce consistent results
    np.random.seed(seed=1)
    
    # Shuffle the data before starting
    permutation = np.random.permutation(len(feature_matrix))
    feature_matrix = feature_matrix[permutation,:]
    sentiment = sentiment[permutation]

    i = 0 # index of current batch
    # Do a linear scan over data
    for itr in range(max_iter):
        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        # Make sure to slice the i-th row of feature_matrix with [i:i+batch_size,:]
        ### YOUR CODE HERE
        predictions =  predict_probability(feature_matrix[i:i+batch_size,:], coefficients)
        
        # Compute indicator value for (y_i = +1)
        # Make sure to slice the i-th entry with [i:i+batch_size]
        ### YOUR CODE HERE
        indicator = (sentiment[i:i+batch_size]==+1)

        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in range(len(coefficients)): # loop over each coefficient
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j]
            # Compute the derivative for coefficients[j] and save it to derivative.
            # Make sure to slice the i-th row of feature_matrix with [i:i+batch_size,j]
            ### YOUR CODE HERE
            derivative = feature_derivative(errors, feature_matrix[i:i+batch_size,j])
                  # Compute the product of the step size, the derivative, and
            # the **normalization constant** (1./batch_size)
            ### YOUR CODE HERE
            coefficients[j] += step_size*derivative*1./batch_size

        # Checking whether log likelihood is increasing
        # Print the log likelihood over the *current batch*
        lp = compute_avg_log_likelihood(feature_matrix[i:i+batch_size,:], sentiment[i:i+batch_size],
                                        coefficients)
        log_likelihood_all.append(lp)
        if itr <= 15 or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) \
         or itr % 10000 == 0 or itr == max_iter-1:
            data_size = len(feature_matrix)
            print('Iteration %*d: Average log likelihood (of data points  [%0*d:%0*d]) = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, \
                 int(np.ceil(np.log10(data_size))), i, \
                 int(np.ceil(np.log10(data_size))), i+batch_size, lp))  

        # if we made a complete pass over data, shuffle and restart
        i += batch_size
        if i+batch_size > len(feature_matrix):
            permutation = np.random.permutation(len(feature_matrix))
            feature_matrix = feature_matrix[permutation,:]
            sentiment = sentiment[permutation]
            i = 0                

    # We return the list of log likelihoods for plotting purposes.
    return(coefficients, log_likelihood_all)

# =============================================================================
# stochastic example
# =============================================================================
initial_coefficients = np.zeros(194)
step_size = 5e-1
batch_size = 1
max_iter = 10
coefficients, log_likelihood_all = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients, step_size, batch_size, max_iter)
plt.plot(log_likelihood_all)

# =============================================================================
# reg gradient descent
# =============================================================================
initial_coefficients = np.zeros(194)
step_size = 5e-1
batch_size = feature_matrix_train.shape[0]
max_iter = 200
coefficients, log_likelihood_all = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients, step_size, batch_size, max_iter)
plt.plot(log_likelihood_all)

import matplotlib.pyplot as plt
plt.plot(log_likelihood_all)


import matplotlib.pyplot as plt
%matplotlib inline

def make_plot(log_likelihood_all, len_data, batch_size, smoothing_window=1, label=''):
    plt.rcParams.update({'figure.figsize': (9,5)})
    log_likelihood_all_ma = np.convolve(np.array(log_likelihood_all), \
                                        np.ones((smoothing_window,))/smoothing_window, mode='valid')

    plt.plot(np.array(range(smoothing_window-1, len(log_likelihood_all)))*float(batch_size)/len_data,
             log_likelihood_all_ma, linewidth=4.0, label=label)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.xlabel('# of passes over data')
    plt.ylabel('Average log likelihood per data point')
    plt.legend(loc='lower right', prop={'size':14})
    
# =============================================================================
#     stochastic
# =============================================================================
step_size= 1e-1
batch_size= 100
initial_coefficients = np.zeros(194)    
max_iter = 200

coefficients, log_likelihood_all_sto = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients, step_size, batch_size, max_iter)
make_plot(log_likelihood_all, feature_matrix_train[i:i+batch_size,:].shape[0], batch_size, smoothing_window=1, label='Stochastic')

# =============================================================================
#     reg gradient ascent
# =============================================================================
initial_coefficients = np.zeros(194)
step_size = 5e-1
batch_size = feature_matrix_train.shape[0]
max_iter = 200

coefficients, log_likelihood_all_reg = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients, step_size, batch_size, max_iter)
make_plot(log_likelihood_all,batch_size , batch_size, smoothing_window=1, label='Stochastic')
# =============================================================================
# 
# =============================================================================

def make_plot(log_likelihood_all, len_data, batch_size, smoothing_window=1, label=''):
    
log_likelihood_all = log_likelihood_all_sto   
len_data =  feature_matrix_train[i:i+batch_size,:].shape[0]
batch_size = batch_size
label='Stochastic'


log_likelihood_all_ma = np.convolve(np.array(log_likelihood_all), \
                                        np.ones((smoothing_window,))/smoothing_window, mode='valid')

plt.plot(np.array(range(smoothing_window-1, len(log_likelihood_all)))*float(batch_size)/len_data,
             log_likelihood_all_ma, linewidth=4.0, label=label)

    
log_likelihood_all = log_likelihood_all_reg
len_data =  feature_matrix_train.shape[0]
batch_size = feature_matrix_train.shape[0]

log_likelihood_all_mt = np.convolve(np.array(log_likelihood_all), \
                                        np.ones((smoothing_window,))/smoothing_window, mode='valid')

plt.plot(np.array(range(smoothing_window-1, len(log_likelihood_all)))*float(batch_size)/len_data,
             log_likelihood_all_ma,)
plt.show()
plt.plot(tt, log_likelihood_all_ma,tt,log_likelihood_all_mt)



# =============================================================================
# 
# =============================================================================
step_size = 1e-4

initial_coefficients=np.zeros(194)
batch_size=100
max_iter = 10
smoothing_window=30

coefficients, log_likelihood_all_1e4 = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients, step_size, batch_size, max_iter)

step_size =1e-3
coefficients, log_likelihood_all_1e3 = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients, step_size, batch_size, max_iter)

step_size =1e-2
coefficients, log_likelihood_all_1e2 = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients, step_size, batch_size, max_iter)

step_size =1e-1
coefficients, log_likelihood_all_1e1 = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients, step_size, batch_size, max_iter)

step_size =1e0
coefficients, log_likelihood_all_1e0 = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients, step_size, batch_size, max_iter)

step_size = 1e1
coefficients, log_likelihood_all_1ep1 = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients, step_size, batch_size, max_iter)

step_size =1e2
coefficients, log_likelihood_all_1ep2= logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients, step_size, batch_size, max_iter)




    make_plot(log_likelihood_all_1e4, len_data=len(feature_matrix_train), batch_size=100,
              smoothing_window=30, label='step_size=%.1e'%step_size)