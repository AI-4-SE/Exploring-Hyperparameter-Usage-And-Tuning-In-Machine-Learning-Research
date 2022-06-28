#!/usr/bin/env python
# coding: utf-8

# # Loading libraries and the data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/london-weather-data/london_weather.csv')


# In[ ]:


df


# # Data visualisation
# 
# ## The mean temperature
# 
# Let's first look at the mean temperature over the years.
# 
# First overview.

# In[ ]:


sns.lineplot(x="date", y="mean_temp", data=df)


# The let's check the linear regression to check the evolution of it.

# In[ ]:


sns.lmplot(x="date", y="mean_temp", data=df, scatter_kws={"s": 1})


# As we can see, in 4 decades, the average temperature as risen about 2/3 degrees.
# 
# ## The min temperature
# 
# As we can imagine, it should increase year after year, let's check this out !

# In[ ]:


sns.lineplot(x="date", y="min_temp", data=df)


# In[ ]:


sns.lmplot(x="date", y="min_temp", data=df, scatter_kws={"s": 1})


# As we expected it, the min temperature is rising from 2/3 degrees in 40 years. Is it the same for the max temperature ?
# 
# ## The max temperature

# In[ ]:


sns.lineplot(x="date", y="max_temp", data=df)


# In[ ]:


sns.lmplot(x="date", y="max_temp", data=df, scatter_kws={"s": 1})


# The max temperature is also rising.

# ## The precipitation over the years

# In[ ]:


sns.lineplot(x="date", y="precipitation", data=df)


# The precipitation doesn't seem to evolve throught the years, let's check that with another linear regression.

# In[ ]:


sns.lmplot(x="date", y="precipitation", data=df, scatter_kws={"s": 1})


# As we thought a bit earlier, the precipitation doesn't seems to have changed a lot over the years.
# 
# ## Does snow as melted for 40 years ?

# In[ ]:


sns.lineplot(x="date", y="snow_depth", data=df)


# In[ ]:


sns.lmplot(x="date", y="snow_depth", data=df, scatter_kws={"s": 1})


# The snow doesn't seems to have evolved for 40 years.
# 
# ## The global radiation

# In[ ]:


sns.lineplot(x="date", y="global_radiation", data=df)


# In[ ]:


sns.lmplot(x="date", y="global_radiation", data=df, scatter_kws={"s": 1})


# We can notice a slight increase of the global radiations but nothing very important.
# 
# ## Is there still the same amount of sun and cloud ?
# 
# Is the sun evolving or not ?

# In[ ]:


sns.lineplot(x="date", y="sunshine", data=df)


# In[ ]:


sns.lmplot(x="date", y="sunshine", data=df, scatter_kws={"s": 1})


# The sunshine is not evolving at all. What about the cloud cover ?

# In[ ]:


sns.lineplot(x="date", y="cloud_cover", data=df)


# In[ ]:


sns.lmplot(x="date", y="cloud_cover", data=df, scatter_kws={"s": 1})


# It seems to be less cloud cover year after year.
# 
# ## What about the pressure ?

# In[ ]:


sns.lineplot(x="date", y="pressure", data=df)


# In[ ]:


sns.lmplot(x="date", y="pressure", data=df, scatter_kws={"s": 1})


# No interesting data here.
# 
# # Conclusions
# 
# - The temperature has risen from 2/3 degrees in 40 years.
# - The precipitation didn't evolve 
# - The snow depth seems nto have decreased a bit
# - Less cloud over years
# - Still the same sunshine "amount"
# - The pressure seems to be the same
