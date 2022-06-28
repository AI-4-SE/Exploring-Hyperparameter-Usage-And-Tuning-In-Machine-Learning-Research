#!/usr/bin/env python
# coding: utf-8

# ### Introduction

# Climate change is undoubtedly one of the biggest challenges of our times. The increase of average temperature in the last centuries has already caused many harmful problems, like intense drought, storms, heat waves or rising sea levels. The primary objective of this exploratory data analysis is to uncover if the global trends are also visible in London’s weather.

# ### Data description

# This analyses is based on dataset from Kaggle and was originally created by reconciling measurements from requests of individual weather attributes provided by the European Climate Assessment (ECA). The measurements of this particular dataset were recorded by a weather station near Heathrow airport.
# 
# 
# The dataset conteins 10 variables:
# 1.	date - recorded date of measurement
# 2.	cloud_cover - cloud cover measurement in oktas
# 3.	sunshine - sunshine measurement in hours (hrs)
# 4.	global_radiation - irradiance measurement in Watt per square meter (W/m2)
# 5.	max_temp - maximum temperature recorded in degrees Celsius (°C) 
# 6.	mean_temp - mean temperature in degrees Celsius (°C) 
# 7.	min_temp - minimum temperature recorded in degrees Celsius (°C) 
# 8.	precipitation - precipitation measurement in millimeters (mm)
# 9.	pressure - pressure measurement in Pascals (Pa)
# 10.	snow_depth - snow depth measurement in centimeters (cm)
# 
# It covers measurements from 1st January 1979 to 31st December 2020.

# ### Preparation

# ##### Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ##### Loading dataset

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


d = pd.read_csv('/kaggle/input/london-weather-data/london_weather.csv')


# ##### Data overview

# In[ ]:


# checking first rows
d.head()


# In[ ]:


# checking last rows
d.tail()


# In[ ]:


d.count()


# In[ ]:


# checking names of columns
d.columns


# In[ ]:


# checking types of variables
d.dtypes


# In[ ]:


# changing the date format
d['date'] = pd.to_datetime(d['date'], format='%Y%m%d')
# adding columns with year/month/day
d['year'] = pd.to_datetime(d['date']).dt.year
d['month'] = pd.to_datetime(d['date']).dt.month
d['day'] = pd.to_datetime(d['date']).dt.day


# In[ ]:


# adding a new column with month name
data = {'month': [1,2,3,4,5,6,7,8,9,10,11,12],\
        'month_name': ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]}
month_name = pd.DataFrame(data)
d = d.merge(month_name,on = 'month',how = 'left')


# In[ ]:


d.dtypes


# ### Exploratoration

# ##### Temperature

# In[ ]:


d.groupby('year').mean().mean_temp

plt.plot(d.groupby('year').mean().mean_temp.index, d.groupby('year').mean().mean_temp.values)
plt.show()


# Max temperatures in last 40 years

# In[ ]:


d.groupby('year').mean().max_temp

plt.plot(d.groupby('year').mean().max_temp.index, d.groupby('year').mean().max_temp.values)
plt.show()


# Minimal temperatures in last 40 years

# In[ ]:


d.groupby('year').mean().min_temp

plt.plot(d.groupby('year').mean().min_temp.index, d.groupby('year').mean().min_temp.values)
plt.show()


# Top 5 hottest years

# In[ ]:


# Top 5 hottest years

d[['year', 'max_temp']].groupby('year').mean().sort_values(by='max_temp', ascending=False).head(5)


# Hotwaves

# In[ ]:


d[['date', 'max_temp']].sort_values(['max_temp'],ascending=False).head(5)


# Top 10 lowest temp

# In[ ]:


# Top 10 lowest temp
d[['date', 'max_temp']].sort_values(['max_temp'],ascending=True).head(10)


# Precipitation in last 40 years

# In[ ]:


d.groupby('year').mean().precipitation
plt.plot(d.groupby('year').mean().precipitation.index, d.groupby('year').mean().precipitation.values)
plt.show()


# Number of days a year with downpours

# In[ ]:


d[d['precipitation'] > 10].groupby('year').count().precipitation.sort_values(ascending=False).head(5).to_frame()


# Number of days a year without rain

# In[ ]:


d[d['precipitation'] == 0].groupby('year').count().precipitation.sort_values(ascending=False).head(10).to_frame()


# ##### Snow depth - change between 1979 and 2020

# In[ ]:


d.groupby('year').mean().snow_depth

plt.plot(d.groupby('year').mean().snow_depth.index, d.groupby('year').mean().snow_depth.values)
plt.show()


# ### Summary

# The main aim of this short exploratory analysis was to check main trend in the London weather in last 50 years.
# Here are concusions:
# 
# - there is a significant increasing trend in average mean, min and max temperature
# - 5 hottest years was after 2010
# - 4 of 5 hottest days in last 40 years was after 2015 (!)
# - top 10 lowest temperatures was before 2000
# - precipitation are more extreme
# - top 5 years with the highest number of days with strong precipitation was after 2000
