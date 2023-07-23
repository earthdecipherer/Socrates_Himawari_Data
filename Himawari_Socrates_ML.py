#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 21:04:35 2023

@author: clerance
"""

#import Statements 

#%%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#import netcdf4 
import csv 
import xarray as xr 

import shap
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#%%

soc_data = pd.read_csv('SOCRATES_10s_data_v3.csv') #reads csv file and indexes column by time
#%%
#Creates Xarray of csv file
soc_dataset = xr.Dataset.from_dataframe(soc_data)
soc_dataset = soc_dataset.set_coords(['lon', 'lat'])

him_data = xr.open_dataset('NC_H08_20180119_0200_R21_FLDK.02401_02401.nc')

#filtered dataset for points of interests with 2911 attributes 
him_data_filtered = him_data.sel(latitude=soc_dataset.lat, longitude=soc_dataset.lon, method="Nearest")

#%%
#Attributes 

'''
albedo_01 = him_data_filtered.albedo_01
albedo_02 = him_data_filtered.albedo_02 
sd_albedo_03 = him_data_filtered.sd_albedo_03
albedo_04 = him_data_filtered.albedo_04
albedo_05 = him_data_filtered.albedo_05
albedo_06 = him_data_filtered.albedo_06 
tbb_07 = him_data_filtered.tbb_07
tbb_08 = him_data_filtered.tbb_08
tbb_09 = him_data_filtered.tbb_09
tbb_10 = him_data_filtered.tbb_10
tbb_11 = him_data_filtered.tbb_11
tbb_12 = him_data_filtered.tbb_12
tbb_13 = him_data_filtered.tbb_13
tbb_14 = him_data_filtered.tbb_14
tbb_15 = him_data_filtered.tbb_15   
tbb_16 = him_data_filtered.tbb_16 
SAZ = him_data_filtered.SAZ 
SAA = him_data_filtered.SAA 
SOZ = him_data.SOZ 
SOA = him_data.SOA 
'''




X = him_data_filtered[['albedo_01', 'albedo_02', 'sd_albedo_03', 'albedo_04', 'albedo_05', 'albedo_06']] # define X to be the satellite observations from different channels


X = X.to_dataframe() # convert to a pandas data frame

y = soc_dataset.LWC 


#y = soc_dataset.Ntot_ice
 


#Begin Machine Learning Script
#Splitting Test and Training Data in 20:80 parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# assign the function to a variable name and train the model by calling the method model.fit() 
gbrt_model = GradientBoostingRegressor() # by not defining them here, we keep all GradientBoostingRegressor hyperparameters at the default values 
gbrt_model.fit(X_train, y_train)
gbrt_score = gbrt_model.score(X_test, y_test) #evaluates the score of the model constructed in the previous syntax



n_iter = 100 # number of random iterations

#Default parameters 
grid_params = {'learning_rate': [0.01, 0.03, 0.1, 0.3],
        'max_depth': [2,4,6,8],
        'n_estimators': [100,200,300,400,500],
        'subsample':[0.6,0.7,0.8] ,
        'min_samples_split': [3,10,30] 
       }

searchcv = GridSearchCV(gbrt_model, grid_params, n_jobs = -1, verbose = 1, cv = 5)

# here, we define the way the values for the RandomizedSearchCV are drawn for each specific hyperparameter that we want to tune
random_params = {'learning_rate': np.random.uniform(0,1,size = n_iter),
        'max_depth': np.random.randint(3, 10, n_iter),
        'n_estimators': np.random.randint(100, 500, n_iter),
        'subsample':np.random.uniform(0,1,size = n_iter),
        'min_samples_split': np.random.randint(3, 30, n_iter)
         }

# assign the function RandomizedSearchCV to a variable name and train the models in using hyperparameters set in the variable random by calling the method model.fit() 
searchcv = RandomizedSearchCV(gbrt_model, random_params, n_iter = n_iter, n_jobs = -1, verbose = 1, cv = 5) 
searchcv.fit(X_train, y_train) 

# one attribute of RandomizedSearchCV is "best_estimator_", which is the model instance with the combination of hyperparameter values that produced the best results in the cross validation 
best_gbrt_model = searchcv.best_estimator_

# store the mean test scores of the cross validation in the variable "scores" 
scores = searchcv.cv_results_["mean_test_score"]

# store the maximum depth used in each iteration cross validation in the variable "max_depth"
max_depth = searchcv.cv_results_["param_max_depth"]

# store the number of estimators used in each iteration of the cross validation in the variable "n_estimators"
n_estimators = searchcv.cv_results_["param_n_estimators"]

sub_sample = searchcv.cv_results_["param_subsample"]

# you can visualize how a hyperparameter setting is related to the score
plt.scatter(sub_sample,scores)
#plt.scatter(n_estimators,scores)
plt.xlabel("sub_sample")
#plt.xlabel("param_n_estimators")
plt.ylabel("mean_test_score")
plt.ylim(-1,1)
plt.grid()
plt.show()

# you can also visualize how two hyperparameter settings are related to the score at the same time
plt.scatter(n_estimators, max_depth, c=scores, vmin=0.1, vmax=0.9)
plt.xlabel('n_estimators'); plt.ylabel('max_depth')
plt.colorbar(label='Mean test score in cross validation')
plt.grid()
plt.show()

learning_rate = searchcv.cv_results_["param_learning_rate"]
subsample = searchcv.cv_results_["param_subsample"]

y_hat_gbrt = best_gbrt_model.predict(X_test) # using the optimized GBRT model

from sklearn.metrics import mean_squared_error, r2_score
print('GBRT mean squared error: %.2f' % mean_squared_error(y_test, y_hat_gbrt)) 
print('GBRT coefficient of determination: %.2f' % r2_score(y_test, y_hat_gbrt))
print('GBRT coefficient of determination with the built-in method "score": %.2f' % best_gbrt_model.score(X_test,y_test))


plt.scatter(y_test, y_hat_gbrt)
#plt.scatter(y_test, y_pred)
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.plot(y_test, y_test, color = 'k')
plt.title("Comparing Test Datasets")
plt.grid()
plt.show()


y_hat_gbrt_train = best_gbrt_model.predict(X_train)
plt.scatter(y_train, y_hat_gbrt_train)

plt.plot(y_train, y_train, color = 'k')
plt.title('Comparing Training Datasets')
plt.grid()
plt.show()


# Compute test score (train score is stored in the model output as the attribute "train_score_")
test_score = np.zeros([best_gbrt_model.n_estimators])
for i, y_hat in enumerate(best_gbrt_model.staged_predict(X_test)):
    test_score[i] = best_gbrt_model.loss_(y_test, y_hat)

# plot train and test scores
plt.plot(best_gbrt_model.train_score_, label='Training')
plt.plot(test_score, label='Testing')
plt.ylabel('Score (loss)')
plt.xlabel('Number of estimators')
plt.grid()
plt.legend()
plt.show()

# Here we print the impurity-based feature importance which is saved within the model object.
print(best_gbrt_model.feature_importances_)

pi = permutation_importance(best_gbrt_model, X_test, y_test, random_state=0)
print(pi.importances_mean)




    

