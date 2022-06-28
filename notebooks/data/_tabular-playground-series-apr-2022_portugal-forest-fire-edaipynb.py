#!/usr/bin/env python
# coding: utf-8

# **EXPLORATORY DATA ANALYSIS OF PORTUGAL FOREST FIRES**

# **Dataset Information**
# 
# This is a dataset of 517 forest fires in Montesinho park, Portugal from January 2000 to December 2003. It was compiled and made availabe by Cortez and morais,2007.
# 
# Data attributes:
# * X : x-axis coordinates(1 - 9).
# * Y : y-axis coordinates(1 - 9).
# * Month : Months of the year(1 - 12).
# * Day : Days of the week(1 - 7).
# * FFMC : Fine Fuel Moisture Code(18.7 - 96.2).
# * DMC : Duff Moisture Code(1.1 - 291.3).
# * DC : Drought Code(7.9 - 860.6).
# * ISI : Initial Spread Index(0 - 56.1).
# * Temp: Temperature(2.2°C - 33.3°C).
# * RH : Relative Humidity(15% - 100%).
# * Wind : Wind Speed(Km/hr)(0.40 - 9.40).
# * Rain : Rain(mm)(0.0 - 6.4).
# * Burned Area : Total burned area(ha)(0 - 1090.84).

# **Importing the required libraries.**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **Loading the data and checking the first five rows.**

# In[ ]:


data = pd.read_csv('/kaggle/input/forest-forest-dataset/forestfires.csv')
data.head()


# **Checking the data for missing entries.**

# In[ ]:


data.isnull().sum()


# **No missing values found.**
# 
# **Converting the month and day column from categorical values to numerical values and checking the first five rows of the data again.**

# In[ ]:


data['month'] = data['month'].replace({'jan': 1, 'feb':2, 'mar': 3, 'apr':4, 'may': 5, 'jun' :6,  'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov':11, 'dec': 12})
data['day'] = data['day'].replace({'mon':1, 'tue':2, 'wed':3,'thu':4, 'fri':5,'sat':6,'sun':7})
data.head()


# **Histogram distribution of the burnt areas.**

# In[ ]:


ax = sns.histplot(data.area, bins = 10)
sns.despine()
plt.title('burnt area(in hectares)')
plt.ylabel('frequency')
total = float(len(data.area))
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()/2-0.05
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center')


# **Transforming the burnt area with natural logarithm and comparing the plot with the untransformed plot.**

# In[ ]:


fig, axes = plt.subplots(1, 2)
fig.suptitle('Histogram of burnt area(Blue) and logrithm transform(Red)')
sns.despine()
sns.histplot(data.area, bins = 10, color ='b',ax = axes[0])
axes[0].set_title('burnt area(in hectares)')
axes[0].set_ylabel('frequency')
sns.histplot(np.log(data.area +1), color = 'r', ax = axes[1])
axes[1].set_title('Log(burnt area)')
axes[1].set_ylabel('frequency')
plt.show()


# **Plot showing the frequency of forest fires by month.**

# In[ ]:


ax = sns.countplot(x =data.month)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
sns.despine()
total = float(len(data.month))
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()/2-0.05
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center')


# **Plot showing the frequency of forest fires in days of the week.**

# In[ ]:


ax = sns.countplot(x =data.day)
ax.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
sns.despine()
total = float(len(data.day))
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()/2-0.05
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center')


# **Plots showing the distribution of the remaining attributes and their densities.**

# In[ ]:


for x in ['X', 'Y', 'FFMC','DMC', 'DC', 'ISI', 'temp','RH','wind','rain']:
          sns.displot(data[x], kde=True)


# **Creating a correlation matrix using Pearson's method to show how the attributes are correlated.**

# In[ ]:


data_corr = data.corr(method = 'pearson')
data_corr


# **A heatmap of the correlation matrix for better visualization.**

# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(data_corr,cmap='Greens', annot=True, fmt ='.2g')


# **The correlation values of the attributes to the burnt area**

# In[ ]:


data_corr_area = abs(data_corr['area']).sort_values(ascending=False).drop('area', axis=0)
data_corr_area


# **Plot showing the correlation values of the attributes to the burnt area.**

# In[ ]:


data_corr_area.plot.bar()


# **CONCLUSION**
# 
# * The analysis showed that 98.2% of the fires burnt less than 100m² of the forest area.
# * The highest occurrence of forest fires happened during the summer season, particularly in august and September.
# * There was an increased spike in forest fires in march.
# * More fires started on weekends compared to workdays. This could mean that most fires where caused by humans.
# * The two most correlated attributes to burnt area are temperature and relative humidity.

# **References**
# 
# Cortez, P and Morais, A. (2007). A data mining approach to predict forest fires using meteorological data. In: J.Neves, M.Santos,and J.Machado, editors. Proceeding of the Portuguese conference on artificial intelligence (EPIA, 2007). pp.512-523.
