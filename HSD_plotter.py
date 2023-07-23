#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:41:50 2023

@author: clerance
"""

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import cartopy.crs as ccrs


import cartopy.feature as cfeature


path = '/Users/clerance/Himawari/19/NC_H08_20180119_0400_R21_FLDK.06001_06001.nc'
him_8 = xr.open_dataset(path)


# Load the CSV file
csv_path = 'SOCRATES_stratiform_profiles.csv'
soc = pd.read_csv(csv_path)



def soc_on_hsd_plotter(flight, latmin, latmax, lonmin, lonmax, point_size):
    
    mask = soc['Flight'] == str(flight)
    soc1 = soc[mask]
    lat_range = (latmin, latmax)
    lon_range = (lonmin, lonmax)
    roi = him_8.sel(latitude=slice(lat_range[0], lat_range[1]), longitude=slice(lon_range[0], lon_range[1]))
    flight_lat = []
    flight_lon = []
    for index, row in soc1.iterrows():
        lat = row['lat']
        lon = row['lon']
       # print(lat, lon)
        # Check if the coordinates lie within the specified range
        if lat_range[1] <= lat <= lat_range[0] and lon_range[0] <= lon <= lon_range[1]:
            #print(lat, lon)
            flight_lat.append(lat)
            flight_lon.append(lon)
    
    #defining variables 
    flight_points = pd.DataFrame({'lat': flight_lat, 'lon': flight_lon})
    latitude = roi['latitude']
    longitude = roi['longitude']
    hour = roi['Hour']
    albedo_03 = roi['albedo_03']

    # Plotting the albedo data
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    #ax.coastlines()
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)


    latitude_grid = np.arange(latmax, latmin, 0.02)
    longitude_grid = np.arange(lonmin, lonmax, 0.02)
    

    gridline_color = 'black'

    ax.gridlines(color=gridline_color, linewidth=0.5, xlocs=longitude_grid, ylocs=latitude_grid, draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Plot the albedo data
    cax = ax.contourf(longitude, latitude, albedo_03, transform=ccrs.PlateCarree(), cmap='viridis')

    #condition = (soc['lat'] >= -50) & (soc['lat'] <= -52) #bounds based on nc bounds 
    #soc = soc[condition]
    soc_latitude = flight_points['lat']
    soc_longitude = flight_points['lon']

    # Plot the latitude and longitude points on the map
    ax.scatter(soc_longitude, soc_latitude, color = 'white', s=point_size, transform=ccrs.PlateCarree())


    # Show the plot
    plt.show()
    
    return flight_points
    
    
soc_on_hsd_plotter(flight = 'RF02', latmin = -45, latmax = -46, lonmin = 140, lonmax = 141, point_size=10)





