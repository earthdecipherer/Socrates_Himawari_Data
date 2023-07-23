#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:49:47 2023

@author: clerance
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#import netcdf4 
import csv 
import xarray as xr  
import datetime

soc_data_m = pd.read_csv('SOCRATES_10s_data_v3.csv') #reads csv file and indexes column by time


def matlab_time_converter(mt): 
    python_dt = pd.to_datetime(mt-719529,unit='d').round('s')
    #python_dt = datetime.datetime.fromordinal(int(mt)) + datetime.timedelta(days=mt % 1) - datetime.timedelta(days=366)
    return python_dt


matlab_datenum = soc_data_m["time"].to_numpy() 
python_time = [] 

for i in matlab_datenum:
    converted = matlab_time_converter(i)
    python_time.append(converted) 
    
dates = pd.to_datetime(python_time).date 
times = pd.to_datetime(python_time).time 

Time = pd.DataFrame(python_time, columns = ['P_time'])

soc_data = Time.join(soc_data_m)

#Creates Xarray of csv file
soc_dataset = xr.Dataset.from_dataframe(soc_data)
soc_dataset = soc_dataset.set_coords(['lon', 'lat'])


# Assuming you have an xarray dataset named 'ds' with a variable named 'var'
data_variable = soc_dataset['flight']
#unique_values = data_variable.unique()
unique_values = np.unique(data_variable.values)

print(unique_values)