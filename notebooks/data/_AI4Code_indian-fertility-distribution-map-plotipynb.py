#!/usr/bin/env python
# coding: utf-8

# # India Fertility Rate Across States and UT
# 
# ### India Fertility Rate by National family Health Survey by State, UT Populalation
# 
# 
# <a id='section-0'></a>
# # Table Of Contents:
# 
# ---
# 
# 1. [Summary](#section-1)
# 1. [Importing Necessary Libraries](#section-2)
# 1. [Dataset Loading and Pre-Processing](#section-3)
# 1. [Exploratory Data Analysis](#section-4)
#     1. [Top 10 States Based On Population](#section-4.1)
#     1. [Population Distribution](#section-4.2)
#     1. [NFHS-4 Distribution](#section-4.3)
#     1. [NFHS-5 Distribution](#section-4.4)
# 1. [Thank You](#section-99)
# 
# ---

# <a id='section-0'></a>
# # Summary
# ---
# 
# ## About Dataset
# 
# It is a Dataset of Total Fertility Rate of all India States and Union Territory in NFHS (National Health Family Survey 2015-2016) and NFHS 2019-2021.
# 
# This Dataset contain the population of All States and Union Territory latest updates to 2021.
# 
# ---

# <a id='section-2'></a>
# # Importing Necessary Libraries
# 
# * [Back To Content Table](#section-0)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import geopandas as gpd


# <a id='section-3'></a>
# # Data Loading and Pre-Processing
# 
# * [Back To Content Table](#section-0)
# 
# ---

# ### Data Loading

# In[ ]:


df = pd.read_csv('../input/india-fertility-rate-across-states-and-ut/India_state_ut_fertility_rate.csv')
df.head()


# ### Data Pre-Processing

# In[ ]:


df.info()


# Our data is no null values hence not much preprocessing is required.
# 
# Also except for the State column, all other have numeric datatype.

# In[ ]:


df.describe()


# <a id='section-4'></a>
# # Exploratory Data Analysis
# 
# * [Back To Content Table](#section-0)
# 
# ---

# In[ ]:


top10 = df.copy()
top10 = top10.sort_values(by='Population',ascending = False)
top10 = top10[:10]


# <a id='section-4.1'></a>
# ## Top 10 Countries Based On Population
# 
# * [Back To Content Table](#section-0)

# In[ ]:


plt.figure(figsize=(20,10));
sns.barplot(x='State',y='Population',data=top10,palette='icefire_r');
plt.xlabel('State Name',fontdict={'fontsize': '15', 'fontweight' : '3'});
plt.xticks(rotation=-90)
plt.ylabel('Population Count',fontdict={'fontsize': '15', 'fontweight' : '3'});
plt.title('Top 10 Countries with Highest Population',fontdict={'fontsize': '20', 'fontweight' : '3'});


# I have used the Indian GIS data to plot Indian Map Distribution of Columns.
# 
# Here is a resource to understand the codes used below : [Plot Map](https://www.kaggle.com/code/nehaprabhavalkar/how-to-plot-map-of-india-using-python)

# In[ ]:


shp_gdf = gpd.read_file('../input/india-gis-data/India States/Indian_states.shp')
shp_gdf


# In[ ]:


merged = shp_gdf.set_index('st_nm').join(df.set_index('State'))
merged.head()


# In[ ]:


merged = merged.fillna(0)


# ## Created a function to plot all the columns

# In[ ]:


def map_plot(title,column):
    '''
    Gives a Country Map Plot of India.
    Values to be passed:
    1. The Title of the Graph
    2. The column to be plotted
    '''
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.axis('off')
    ax.set_title(title,
                 fontdict={'fontsize': '15', 'fontweight' : '3'})
    fig = merged.plot(column=column, cmap='icefire_r', linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)


# <a id='section-4.2'></a>
# ## Population Distribution
# 
# * [Back To Content Table](#section-0)

# In[ ]:


map_plot('Population Distribution of Each State','Population')


# <a id='section-4.3'></a>
# ## NFHS-4 Distribution
# 
# * [Back To Content Table](#section-0)

# In[ ]:


map_plot('National Health Family Survey 2015-2016 Distribution of Each State','NFHS-4')


# <a id='section-4.4'></a>
# ## NFHS-5 Distribution
# 
# * [Back To Content Table](#section-0)

# In[ ]:


map_plot('National Health Family Survey 2019-2021 Distribution of Each State','NFHS-5')


# ---
# <a id='section-99'></a>
# # Thank You For Viewing!!!
# 
# ## If you liked the notebook, do upvote it as it helps me stay motivated.
# 
# * [Back To Top](#section-0)
# 
# ---
