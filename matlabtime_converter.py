#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:26:32 2023

@author: clerance
"""

import datetime
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#import netcdf4 
import csv 
import xarray as xr 

def matlab_time_converter(mt): 
    python_dt = datetime.datetime.fromordinal(int(mt)) + datetime.timedelta(days=mt % 1) - datetime.timedelta(days=366)
    return python_dt


data = pd.read_csv('SOCRATES_10s_data_v3.csv')

python_time = []

'''
for i in data.time:
    converted = matlab_time_converter(i)
    python_time = list.append(converted)
    
'''

matlab_datenum = data["time"].to_numpy()

for i in matlab_datenum:
    converted = matlab_time_converter(i)
    python_time.append(converted) 



