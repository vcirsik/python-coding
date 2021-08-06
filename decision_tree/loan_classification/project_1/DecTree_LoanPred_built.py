# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:39:35 2021

@author: virsik
"""
# =============================================================================
# 
# # this script fits a decision tree model to predict loan safety
# # via hand built functions instead of sklearn
# # script created as part of assignment from classification course.
#  
# =============================================================================


import numpy as np
import pandas as pd


# load data as dataframe
loans = pd.read_csv("data\\lending-club-data.csv") 

# recode loan code in column ['bad loans']; prev 1 bad 0 good
# safe_loans =  1 => safe  # safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)

# get rid of old unneeded column with prev loan coding
loans = loans.drop('bad_loans',1)

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'

# Extract the feature columns and target column
loans = loans[features + [target]]


import json

# get indices for training and validation set for classifier
with open("data\\module-5-assignment-2-train-idx.json", "r") as read_file:
    train_idx = json.load(read_file)

with open("data\\module-5-assignment-2-test-idx.json", "r") as read_file:
    test_idx = json.load(read_file)
    

# =============================================================================
#  one hot encoding w/ pandas
# =============================================================================

# limit to categorical data using df.select_dtypes()
tmpDat = loans.select_dtypes(include=[object])
tmpDat.head(3)

# get one hot encoding via get_dummies
oneH_tmpDat = pd.get_dummies(tmpDat, prefix=tmpDat.columns)

# remove nas replace with 0s
oneH_tmpDat = oneH_tmpDat.fillna(0)

# get rid of old columns
loans = loans.drop(tmpDat.columns,1)

# append one hot coded data

loan_data = pd.concat([loans,oneH_tmpDat], axis=1)

# =============================================================================
# separate into training and validation data; convert to np 
# =============================================================================

train_data = loan_data.iloc[train_idx]
test_data = loan_data.iloc[test_idx]    

# =============================================================================
# write function to determine error counts when doing node majority
# calculations
# =============================================================================
def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0    
    if type(labels_in_node) is np.ndarray:
        labels_in_node = labels_in_node
    else:
        labels_in_node = labels_in_node.to_numpy()
    # Count the number of 1's (safe loans)
    safeCount = (labels_in_node == 1).sum()
    
    # Count the number of -1's (risky loans)
    riskyCount  = (labels_in_node == -1).sum()    

  
    # Return the number of mistakes that the majority classifier makes.
    mistakeCount = min(riskyCount, safeCount)
    return(mistakeCount)
# =============================================================================
 
# test out function    
# =============================================================================
# example_labels = pd.DataFrame([-1, -1, 1, 1, 1])
# if intermediate_node_num_mistakes(example_labels) == 2:
#     print('Test passed!')
# else:
#     print('Test 1 failed... try again!')   
#     
# example_labels = pd.DataFrame([-1, -1, 1, 1, 1, 1, 1])
# if intermediate_node_num_mistakes(example_labels) == 2:
#     print('Test passed!')
# else:
#     print('Test 2 failed... try again!')       
#     
# example_labels = pd.DataFrame([-1, -1, -1, -1, -1, 1, 1])
# if intermediate_node_num_mistakes(example_labels) == 2:
#     print('Test passed!')
# else:
#     print('Test 3 failed... try again!')       
# =============================================================================

def best_splitting_feature(data, features, target):
    
    # target_values = data[target]
    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        
        # The left split will have all data points where the feature value is 0
        # i.e., uses logical indexing to group based on features[feature]==0
        left_split = data[data[feature] == 0]
            
        # The right split will have all data points where the feature value is 1
        right_split =  data[data[feature] == 1]
                
        # Calculate the number of misclassified examples in the left split.
        # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
        left_mistakes = intermediate_node_num_mistakes(left_split[target])           
    
        # Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split[target]) 
        
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        error = (left_mistakes + right_mistakes)/num_data_points
    
        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        if error < best_error:
            best_feature = feature
            best_error   = error
            
    return(best_feature) # Return the best feature we found

# =============================================================================
# { 
#    'is_leaf'            : True/False.
#    'prediction'         : Prediction at the leaf node.
#    'left'               : (dictionary corresponding to the left tree).
#    'right'              : (dictionary corresponding to the right tree).
#    'splitting_feature'  : The feature that this node splits on
# }
# =============================================================================


def create_leaf(target_values):    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True  }   ## YOUR CODE HERE 
   
    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])    

    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = 1  ## YOUR CODE HERE
    else:
        leaf['prediction'] = -1   ## YOUR CODE HERE        

    # Return the leaf node
    return(leaf) 


def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print("--------------------------------------------------------------------")
    print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))
        

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    if intermediate_node_num_mistakes(target_values) == 0:  ## YOUR CODE HERE
        print("Stopping condition 1 reached.")     
        # If not mistakes at current node, make current node a leaf node
        return(create_leaf(target_values))
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features.empty:   ## YOUR CODE HERE
        print("Stopping condition 2 reached.")   
        # If there are no remaining features to consider, make current node a leaf node
        return(create_leaf(target_values))    
    
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:  ## YOUR CODE HERE
        print("Reached maximum depth. Stopping for now.")
        # If the max tree depth has been reached, make current node a leaf node
        return(create_leaf(target_values))

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    ## YOUR CODE HERE
    splitting_feature = best_splitting_feature(data, remaining_features, target)    
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split =  data[data[splitting_feature] == 1]     ## YOUR CODE HERE
    remaining_features = remaining_features.drop(splitting_feature,1)
    print("Split on feature: %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split)))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print("Creating leaf node.")
        return(create_leaf(left_split[target]))
    if len(right_split) == len(data):
        print("Creating leaf node.")
        return(create_leaf(right_split[target]))

        
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)        
    
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth)        

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}



data = train_data
features = train_data.columns     
features = features.drop(target)  

my_tree = decision_tree_create(data, features, target, current_depth = 0, max_depth = 6)


def classify(tree, x, annotate = False):
       # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
             print("At leaf, predicting %s" % tree['prediction'])
        return tree['prediction']
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
             print("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)


print(test_data.iloc[0])
print('Predicted class: %s ' % classify(my_tree, test_data.iloc[0]))


classify(my_tree, test_data.iloc[0], annotate=True)


def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x),axis = 1)
    
    # Once you've made the predictions, calculate the classification error and return it
    corrPred = ((prediction != data[target]).sum())/len(prediction)
    return(corrPred)

evaluate_classification_error(my_tree, test_data)


def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print("(leaf, label: %s)" % tree['prediction'])
        return(None)
    #split_feature, split_value = split_name.split('.')
    print('                       %s' % name)
    print('         |---------------|----------------|')
    print('         |                                |')
    print('         |                                |')
    print('         |                                |')
    print('  [{0} == 0]               [{0} == 1]    '.format(split_name))
    print('         |                                |')
    print('         |                                |')
    print('         |                                |')
    print('    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree')))
        
        
print_stump(my_tree)
print_stump(my_tree['left'], my_tree['splitting_feature'])
print_stump(my_tree['left']['left'], my_tree['left']['splitting_feature'])

print_stump(my_tree)
print_stump(my_tree['right'], my_tree['splitting_feature'])
print_stump(my_tree['right']['right'], my_tree['right']['splitting_feature'])
