#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 13:55:12 2023

@author: clerance
    
"""

import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import datetime as dt 
from datetime import datetime


#########################################################################################
#####Read File





data_stratiform = pd.read_csv('SOCRATES_stratiform_profiles.csv')
data_soc_10s_r = pd.read_csv('SOCRATES_10s_data_v3.csv')






#########################################################################################
###Define the function matlab time converter 





def matlab_time_converter(mt): 
    python_dt = pd.to_datetime(mt-719529,unit='d').round('s')
    #python_dt = datetime.datetime.fromordinal(int(mt)) + datetime.timedelta(days=mt % 1) - datetime.timedelta(days=366)
    return python_dt

matlab_datenum = data_soc_10s_r["time"].to_numpy() 

python_time = [] #defining empty list to append

for i in matlab_datenum:
    converted = matlab_time_converter(i)
    python_time.append(converted) 
    
Time = pd.DataFrame(python_time, columns = ['TIME']) #converts lists into dataframe
data_soc_10s = Time.join(data_soc_10s_r) #joins calculated python time in the dataframe




#######################################################################################


flights = data_stratiform['Flight'].unique() #returns unique flights 
dates = data_stratiform['Date'].unique() #returns unique dates 
##when flight name changes the date also changes 



#definging empty lists to append values in the analysis 
flight_list = [] #returns a list after the loop of all the unique flights
ntot_ice_calculated = [] #returns the calculated average for ntot_ice
count_list = [] #returns the count of the numbers of values following the condition  
dates_list = [] #gives a list of all the unique dates in the flight campaign





for i in flights:
    flight_list.append(i)
    
for d in dates:
    original_string = str(d)
    year = '2018'
    d = original_string + year

    date_obj = datetime.strptime(d, '%d.%m.%Y')
    formatted_string = date_obj.strftime('%Y-%m-%d')
    dates_list.append(formatted_string)
    
value_to_delete = '2018-01-22' #issues with RF03 having two dates(deleting one of them)
#the issue has to be fixed hence, two columns in the out returns Nan values 

dates_list = [value for value in dates_list if value != value_to_delete] #deletes the specified date 

#there is an issue with the date where one flight campaign runs for two dates 
#specifically RF03 where two dates 22.1 and 23.1 is given.
#I deleted 22.1 from the list since there were only two rows for it just to check if the code runs

n_ice = data_stratiform['N_ice'] #calls N_ice from stratiform 




#######################################################################################
#Calculating averegae by creating a filter by selecitng the flight, date, and then creating a comparison for start and end time




for ii, dd in enumerate(dates_list): 
    f = flight_list[ii] #indexes the flight in the list for every iteration
    #condition that specifies to filter the flight name from 10s and stratiform 
    
    flight_mask = data_soc_10s['flight'] == str(f) 
    flight_mask_stratiform = data_stratiform['Flight'] == str(f)
    
    #apply the flight mask to filter
    flight_filtered_10s = data_soc_10s[flight_mask]
    flight_stratiform = data_stratiform[flight_mask_stratiform]
    
    #formatting time from 10s data to datetime object 
    flight_filtered_10s.loc[:, 'TIME'] = pd.to_datetime(flight_filtered_10s['TIME'], format='%Y-%m-%d %H:%M:%S')
    date = str(dd) #ensuring that the date is a string to further call it in the code and reformat it
    
    for i, val in enumerate(flight_stratiform.StartTime):
        
        val = date + ' ' + str(val) #makes the date time string readable for comparison
        #this is done because that stratiform data start time and date are in different columns
        
        start_time = pd.to_datetime(str(val)) 
        
        et = flight_stratiform['EndTime'].iloc[i] #locates the corresponding end time
        et = date + ' ' + str(et) #makes it readable 
        end_time = pd.to_datetime(str(et))
        
        #filteres the dataframe based on the given condition
        filtered_df = flight_filtered_10s[(flight_filtered_10s['TIME'] >= start_time) & (flight_filtered_10s['TIME'] <= end_time)]
        #returns number of values which satisfies the condition
        count = len(filtered_df['Ntot_ice']) 
        count_list.append(count)
        
        # Calculate the average of the filtered values
        average_value = filtered_df['Ntot_ice'].mean()
        ntot_ice_calculated.append(average_value)
        
        
        
        
        
        
#######################################################################################  
#Arraning and Extracting Value

#converting appended lists to dataframe 
cal_nice = pd.DataFrame(ntot_ice_calculated, columns = ['cal_nice'])
count = pd.DataFrame(count_list, columns = ['Count'])

StartTime = data_stratiform['StartTime']
EndTime = data_stratiform['EndTime']

#Makes a new dataframe where all the pandas columns are joined whihc are of interest here
average_discrepancy = cal_nice.join(n_ice).join(StartTime).join(EndTime).join(count)
average_discrepancy['difference'] = abs(average_discrepancy['N_ice'] - average_discrepancy['cal_nice'])

path = '/Users/clerance/HiWi_Yasmin/' 


#extracts a csv file
average_discrepancy.to_csv(path + 'Average_discrepancy.csv', index=False)


#######################################################################################




