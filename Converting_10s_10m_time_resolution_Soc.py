#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:47:17 2023

@author: clerance
"""
import pandas as pd 
import numpy as np
import datetime
import xarray 

filename = "SOCRATES_10s_data_v3.csv"
data = pd.read_csv(filename)


def matlab_time_converter(mt): 
    python_dt = datetime.datetime.fromordinal(int(mt)) + datetime.timedelta(days=mt % 1) - datetime.timedelta(days=366)
    return python_dt

matlab_datenum = data["time"].to_numpy() 
python_time = [] 

for i in matlab_datenum:
    converted = matlab_time_converter(i)
    python_time.append(converted) 
    
Time = pd.DataFrame(python_time, columns = ['P_time'])


Socrates_10s_data = Time.join(data)


round_interval = pd.offsets.Minute(10)

# Round the 'Time' column to the nearest interval
Socrates_10s_data['Rounded_Time'] = (Socrates_10s_data['P_time'] + round_interval/2).dt.floor(round_interval)


#Reassembling Columns 

Socrates_10s_data = Socrates_10s_data[['flight', 'P_time', 'Rounded_Time', 'lat', 'lon', 
                                         'T', 'P', 'Ntot_ice', 'Sice', 'w', 'height', 'LWC']]

#calculating average for matching the 10 minute resolution of Himawari 
Socrates_10m_data_averaged = Socrates_10s_data.groupby('Rounded_Time').mean().reset_index()




#Exporting Data to an Excel File for Comparison 
Socrates_10m_data_averaged.to_excel('soc_10m_avg.xlsx', sheet_name = 'soc10m', index=True)

Socrates_10s_data.to_excel('soc_10s_avg.xlsx', sheet_name = 'soc10s', index=True)











