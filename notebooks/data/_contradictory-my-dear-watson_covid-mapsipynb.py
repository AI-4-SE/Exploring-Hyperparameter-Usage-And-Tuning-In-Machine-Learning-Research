#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


covid=pd.read_csv("../input/covid19-coronavirus-pandemic/COVID-19 Coronavirus.csv")


# In[ ]:


covid.head()
covid.columns.tolist()



# In[ ]:


covid.columns=covid.columns.str.replace('//','_').str.replace("/","_").str.replace(" ","_").str.replace("\xa0", "_")


# In[ ]:


covid.head()


# In[ ]:


cases_by_continent=covid.groupby(['Continent'])['Total_Cases'].sum()
deaths_by_continent=covid.groupby(['Continent'])['Total_Deaths'].sum()
print(cases_by_continent.index)

sns.scatterplot(cases_by_continent,deaths_by_continent,hue=cases_by_continent.index)


# In[ ]:


sns.scatterplot(x=covid.Total_Cases,y=covid.Total_Deaths,size=covid.Tot_Deaths_1M_pop,hue=covid.Continent,sizes=(20,200))
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)


# In[ ]:


import geopandas as gpd
map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
map.plot(figsize=(30,30),color="white",edgecolor="darkgrey")
final=map.merge(covid,left_on="name",right_on="Country")


# In[ ]:


final.plot('Total_Deaths',figsize=(20,20),legend=True,
           legend_kwds={'label': "Total Deaths",'orientation': "horizontal"},missing_kwds={'color': 'black'},cmap="plasma")


# In[ ]:


final

