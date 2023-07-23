#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  29 08:00:55 2023

@author: clerance
"""


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#import netcdf4 
import csv 
import xarray as xr 

him_data = xr.open_dataset('NC_H08_20180119_0200_R21_FLDK.02401_02401.nc')

'''
Temperature = him_data.tbb_07.values 

height_07 = []

for t in Temperature:
    H = ((8.31446261815324*t)/(0.029*9.8))/1000   #Based on the formula H = RT/mg https://en.wikipedia.org/wiki/Scale_height
    height_07.append(H)
    
#plt.plot(height, Temperature)
        
        '''
        
def calc_height_from_temperature(BAND):
    band = str(BAND)
    Temperature = him_data[band].values
    height = []
    for t in Temperature:
        H = ((8.31446261815324*t)/(0.029*9.8))/1000   #Based on the formula H = RT/mg https://en.wikipedia.org/wiki/Scale_height
        height.append(H)
    #h = np.reshape(height, [him_data[BAND].shape[0],him_data[BAND].shape[0]])  
    return np.average(height)


l = []
name = []
for i in range(7, 17):
    if i < 10:
        a = 'tbb_0' + str(i)
        #h = calc_height_from_temperature(BAND = 'tbb_0' + '{}'.format(i))
        h = calc_height_from_temperature(BAND = a)
    else:
        a = 'tbb_' + str(i)
        h = calc_height_from_temperature(BAND = a)
    l.append(h)
    name.append(a)
    
Height_deduced = pd.Series(l, index = name)
#dc = dict(a = [1,2,3])

#Height_deduced = pd.DataFrame(dict(H7 = ''))

