#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:21:35 2023

@author: clerance
"""
import xarray as xr
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


soc_data = pd.read_csv('SOCRATES_10s_data_v3.csv') #reads csv file and indexes column by time

#Creates Xarray of csv file
soc_dataset = xr.Dataset.from_dataframe(soc_data)
soc_dataset = soc_dataset.set_coords(['lon', 'lat'])

him_data = xr.open_dataset('NC_H08_20180119_0200_R21_FLDK.02401_02401.nc')

#filtered dataset for points of interests with 2911 attributes 
him_data_filtered = him_data.sel(latitude=soc_dataset.lat, longitude=soc_dataset.lon, method="Nearest")


# Split the dataset into features and target variable
X = him_data_filtered[['albedo_01', 'albedo_02', 'sd_albedo_03', 'albedo_04', 'albedo_05', 'albedo_06']] # define X to be the satellite observations from different channels
X = X.to_dataframe() # convert to a pandas data frame

y = soc_dataset.LWC

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set the parameters for XGBoost
params = {
    'objective': 'reg:squarederror',  # Specify the objective for regression
    'max_depth': 3,                   # Maximum depth of each tree
    'eta': 0.1,                       # Learning rate
    'eval_metric': 'rmse'             # Evaluation metric
}

# Train the XGBoost model
num_rounds = 100  # Number of boosting rounds
model = xgb.train(params, dtrain, num_rounds)

# Make predictions on the test set
predictions = model.predict(dtest)


plt.scatter(X_test['albedo_02'], predictions)  #catter(n_estimators, max_depth, c=scores, vmin=0.1, vmax=0.9)
plt.grid()
plt.show()




'''

# Create an XGBoost classifier
model = xgb.XGBClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-validation scores: {cv_scores}')

'''

