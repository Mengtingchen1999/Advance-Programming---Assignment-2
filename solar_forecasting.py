#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 12:39:26 2021

@author: javi.fong
"""

import numpy as np
import pandas as pd


from sklearn import neighbors, metrics, svm, tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, PredefinedSplit
from sklearn.utils.fixes import loguniform

#from sklearn.model_selection import train_test_split 


my_nia = 100466870
np.random.seed(my_nia)

train = pd.read_pickle("traintestdata_pickle/trainst1ns16.pkl")
test = pd.read_pickle("traintestdata_pickle/testst1ns16.pkl")

#Choose closest point variables
train_x = train.iloc[:,:75]
train_y = train.energy

test_x = test.iloc[:,:75]
test_y = test.energy

#Split Train set into Train_train (first 10y and last 2y)
train_train_x = train_x[:3650]
train_train_y = train_y[:3650]

train_validation_x = train_x[3650:]
train_validation_y = train_y[3650:]

#Scaling Data 
scaler = StandardScaler()
scaler.fit(train_train_x)

scaled_all_train_x = scaler.transform(train_x)
scaled_train_x = scaler.transform(train_train_x)
scaled_test_x = scaler.transform(test_x)
scaled_validation_x = scaler.transform(train_validation_x)

#KNN with default HyperParameters
knn = neighbors.KNeighborsRegressor()
knn.fit(scaled_train_x, train_train_y)
pred_knn_test = knn.predict(scaled_validation_x)
print(metrics.mean_absolute_error(train_validation_y, pred_knn_test))
#2588455.571506849

#SVM with default parameters
svm = svm.SVC() 
svm.fit(scaled_train_x, train_train_y)
pred_svm_test = svm.predict(scaled_validation_x)
print(metrics.mean_absolute_error(train_validation_y, pred_svm_test))
#6920271.25479452

#Decission Tree with default parameters
tree_reg = tree.DecisionTreeRegressor()
tree_reg.fit(scaled_train_x, train_train_y)
pred_tree_test = tree_reg.predict(scaled_validation_x)
print(metrics.mean_absolute_error(train_validation_y, pred_tree_test ))
#MAE = 3063507.1232876712

#Validation set 
validation_indices = np.zeros(train_x.shape[0])
validation_indices[:3650] = -1
validation_partition = PredefinedSplit(validation_indices)

#KNN with parameter tunning
knn_param_grid = {
    'n_neighbors': list(range(1,16,1))
    }

knn_tunned = GridSearchCV(
    knn
    , knn_param_grid
    , scoring='neg_mean_absolute_error'
    , cv = validation_partition
)

knn_tunned.fit(scaled_all_train_x, train_y)
#Params = 14 
#Best score = -2394762.7526418786
pred_knn_tunned_test = knn_tunned.predict(scaled_validation_x)
print(metrics.mean_absolute_error(train_validation_y,pred_knn_tunned_test))


#SVM with Parameter Tunning 
svm_param_grid = {
    'C': loguniform(1e0, 1e3)
    , 'kernel': ['linear', 'rbf']
    , 'gamma': loguniform(1e-4, 1e-3)
    , 'class_weight':['balanced', None]
    }

svm_tunned = RandomizedSearchCV(
    svm
    , svm_param_grid
    , scoring='neg_mean_absolute_error'
    , cv=validation_partition
    , n_iter = 20
    )

svm_tunned.fit(scaled_train_x, train_y)
#pred_svm_tunned_test = svm_tunned.predict(test_x)
#print(metrics.mean_absolute_error(test_y,pred_svm_tunned_test))



param_grid_rt = {"min_samples_split": list(range(2,16,2)),
              "max_depth": list(range(2,16,2)),
              "min_samples_leaf": list(range(2,16,2)),
              "max_leaf_nodes": list(range(2,16,2))}

tree_reg_tunned = RandomizedSearchCV(
    tree_reg
    , param_grid_rt
    , scoring='neg_mean_absolute_error'
    , cv=validation_partition
    , n_iter = 20
    )


svm_tunned.fit(scaled_all_train_x, train_y)






