#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 21:04:35 2023

@author: clerance
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#import netcdf4 
import csv 
import xarray as xr  
import datetime

#import netcdf4 

soc_data = pd.read_csv('SOCRATES_10s_data_v3.csv') #reads csv file and indexes column by time

#Creates Xarray of csv file
soc_dataset = xr.Dataset.from_dataframe(soc_data)
soc_dataset = soc_dataset.set_coords(['lon', 'lat'])

#him_data = xr.open_dataset('NC_H08_20180119_0200_R21_FLDK.02401_02401.nc') #6001(2 Km)

him_data = xr.open_dataset('NC_H08_20180119_1220_R21_FLDK.06001_06001.nc')


#filtered dataset for points of interests with 2911 attributes 
him_data_filtered = him_data.sel(latitude=soc_dataset.lat, longitude=soc_dataset.lon, method="Nearest")

#converting matlab time to python time
def matlab_time_converter(mt): 
    python_dt = datetime.datetime.fromordinal(int(mt)) + datetime.timedelta(days=mt % 1) - datetime.timedelta(days=366)
    return python_dt

python_time = []

matlab_datenum = soc_data["time"].to_numpy()

for i in matlab_datenum:
    converted = matlab_time_converter(i)
    python_time.append(converted) 
    
py_time = pd.DataFrame(python_time, columns=['Python_Time'])


#Extracting Values
SOA = him_data_filtered.SOA.values
tbb_07 = him_data_filtered.tbb_07.values
SAZ = him_data_filtered.SAZ.values
H_lat = him_data_filtered.lat.values
H_lon = him_data_filtered.lon.values

#converting to dataframe 
SOA = pd.DataFrame(SOA)
tbb_07 = pd.DataFrame(tbb_07)
SAZ = pd.DataFrame(SAZ)
H_lat = pd.DataFrame(H_lat)
H_lon = pd.DataFrame(H_lon)

him_col_names = ['py_time','H_lat', 'H_lon', 'tbb_07', 'SOA', 'SAZ']

Him8_extracted_values = pd.concat([py_time, H_lat, H_lon, SOA, SAZ, tbb_07], axis=1)
Him8_extracted_values.columns = him_col_names


Soc_Him8_clipped = pd.concat([soc_data, Him8_extracted_values], axis = 1) 

Soc_Him8_clipped.to_excel('test123.xlsx', index=False)





