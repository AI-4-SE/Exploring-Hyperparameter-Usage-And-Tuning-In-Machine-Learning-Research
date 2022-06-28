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

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Supress warnings

import warnings
warnings.filterwarnings('ignore')


# # 1. **Inspecting and Cleaning Data**

# In[ ]:


Data_2022 = pd.read_csv('/kaggle/input/world-happiness-report-2022/World Happiness Report 2022.csv')


# In[ ]:


# Shape of dataset

Data_2022.shape


# In[ ]:


Data_2022.info()


# In[ ]:


Data_2022.head(5)


# In[ ]:


Data_2022.sample(5)


# In[ ]:


Data_2022.tail(5)


# In[ ]:


# Setting Column 1 as Index

Data_2022.index = Data_2022['RANK']


# In[ ]:


# Checking for Missing values values

missing = pd.DataFrame(Data_2022.isnull().sum()/len(Data_2022.index))

missing.columns = ['Percentage of Missing Values']

missing


# In[ ]:


Data_2022.nunique()


# In[ ]:


Data_2022.duplicated().value_counts()


# What can we make from the data set till now
# 
# 1. There are 146 rows and 12 columns in this dataset'
# 2. There is no null value present 
# 3. Only RANK column is of int datatype, Country column is of Object datatype while others columns are of float datatype
# 4. There is no Duplicate values Present in this data
# 5. Finland Ranks 1 and Afghanistan Ranks Last in this Dataset

# # Checking Distribution of Data and Outliners

# In[ ]:


# Checking Distribution of Data

sns.distplot(Data_2022['Happiness score'])

plt.axvline(x = Data_2022['Happiness score'].mean(), color = 'red')

plt.axvline(x = Data_2022['Happiness score'].median(), color = 'black')


# In[ ]:


sns.distplot(Data_2022['Whisker-high'])

plt.axvline(x = Data_2022['Whisker-high'].mean(), color = 'red')

plt.axvline(x = Data_2022['Whisker-high'].median(), color = 'black')


# In[ ]:


sns.distplot(Data_2022['Whisker-low'])

plt.axvline(x = Data_2022['Whisker-low'].mean(), color = 'red')

plt.axvline(x = Data_2022['Whisker-low'].median(), color = 'black')


# In[ ]:


sns.distplot(Data_2022['Dystopia (1.83) + residual'])

plt.axvline(x = Data_2022['Dystopia (1.83) + residual'].mean(), color = 'red')

plt.axvline(x = Data_2022['Dystopia (1.83) + residual'].median(), color = 'black')


# In[ ]:


sns.distplot(Data_2022['Explained by: Social support'])

plt.axvline(x = Data_2022['Explained by: Social support'].mean(), color = 'red')

plt.axvline(x = Data_2022['Explained by: Social support'].median(), color = 'black')


# In[ ]:


sns.distplot(Data_2022['Explained by: Healthy life expectancy'])

plt.axvline(x = Data_2022['Explained by: Healthy life expectancy'].mean(), color = 'red')

plt.axvline(x = Data_2022['Explained by: Healthy life expectancy'].median(), color = 'black')


# In[ ]:


sns.distplot(Data_2022['Explained by: Freedom to make life choices'])

plt.axvline(x = Data_2022['Explained by: Freedom to make life choices'].mean(), color = 'red')

plt.axvline(x = Data_2022['Explained by: Freedom to make life choices'].median(), color = 'black')


# In[ ]:


sns.distplot(Data_2022['Explained by: Generosity'])

plt.axvline(x = Data_2022['Explained by: Generosity'].mean(), color = 'red')

plt.axvline(x = Data_2022['Explained by: Generosity'].median(), color = 'black')


# In[ ]:


sns.distplot(Data_2022['Explained by: Perceptions of corruption'])

plt.axvline(x = Data_2022['Explained by: Perceptions of corruption'].mean(), color = 'red')

plt.axvline(x = Data_2022['Explained by: Perceptions of corruption'].median(), color = 'black')


# In[ ]:


# Checking Outliners

Boxplot_data = Data_2022.drop(columns = ['RANK'])

plt.figure(figsize = (20,20))

Boxplot_data.boxplot()

plt.xticks(rotation = 90)


# In[ ]:


# Descriptive Statistics

Data_2022.describe()


# What we can conclude till now
# 
# 1. It looks like that mean is almost equal to mode, so we can say that this data is almost normally distributed
# 2. There are only few outliners which are having a very less effect on the mean

# In[ ]:


# Correlation

plt.figure(figsize = (7,7))

corr = Data_2022.corr()

sns.heatmap(corr, annot = True)


# In[ ]:


# Top 10 Happiest Countries

Top_10 = Data_2022.head(10)


# In[ ]:


Top_10


# In[ ]:


plt.bar(Top_10['Country'], Top_10['Happiness score'])

plt.xticks(rotation = 90)

plt.xlabel('Countries')
plt.ylabel('Happiness Score')

plt.figure(figsize = (10,10))


# In[ ]:


plt.figure(figsize = (10,5))

plt.bar(Top_10['Country'],Top_10['Explained by: GDP per capita'])

plt.xlabel('Countries')
plt.ylabel('Explained by: GDP per capita')

plt.xticks(rotation = 90)


# In[ ]:


plt.figure(figsize = (10,5))

plt.bar(Top_10['Country'],Top_10['Explained by: Social support'])

plt.xlabel('Countries')
plt.ylabel('Explained by: Social support')

plt.xticks(rotation = 90)


# In[ ]:


plt.figure(figsize = (10,5))

plt.bar(Top_10['Country'],Top_10['Explained by: Healthy life expectancy'])

plt.xlabel('Countries')
plt.ylabel('Explained by: Healthy life expectancy')

plt.xticks(rotation = 90)


# In[ ]:


plt.figure(figsize = (10,5))

plt.bar(Top_10['Country'],Top_10['Explained by: Freedom to make life choices'])

plt.xlabel('Countries')
plt.ylabel('Explained by: Freedom to make life choices')

plt.xticks(rotation = 90)


# In[ ]:


plt.figure(figsize = (10,5))

plt.bar(Top_10['Country'],Top_10['Explained by: Generosity'])

plt.xlabel('Countries')
plt.ylabel('Explained by: Generosity')

plt.xticks(rotation = 90)


# In[ ]:


plt.figure(figsize = (10,5))

plt.bar(Top_10['Country'],Top_10['Explained by: Perceptions of corruption'])

plt.xlabel('Countries')
plt.ylabel('Explained by: Perceptions of corruption')

plt.xticks(rotation = 90)


# In[ ]:


# Last 10 Countries

Last_10 = Data_2022.tail(10)

Last_10


# In[ ]:


plt.bar(Last_10['Country'], Last_10['Happiness score'], color = 'red')

plt.xticks(rotation = 90)

plt.xlabel('Countries')
plt.ylabel('Happiness Score')

plt.figure(figsize = (10,10))


# In[ ]:


plt.figure(figsize = (10,5))

plt.bar(Last_10['Country'],Last_10['Explained by: GDP per capita'], color = 'red')

plt.xlabel('Countries')
plt.ylabel('Explained by: GDP per capita')

plt.xticks(rotation = 90)


# In[ ]:


plt.figure(figsize = (10,5))

plt.bar(Last_10['Country'],Last_10['Explained by: Social support'], color = 'red')

plt.xlabel('Countries')
plt.ylabel('Explained by: Social support')

plt.xticks(rotation = 90)


# In[ ]:


plt.figure(figsize = (10,5))

plt.bar(Last_10['Country'],Last_10['Explained by: Healthy life expectancy'], color = 'red')


plt.xlabel('Countries')
plt.ylabel('Explained by: Healthy life expectancy')

plt.xticks(rotation = 90)


# In[ ]:


plt.figure(figsize = (10,5))

plt.bar(Last_10['Country'],Last_10['Explained by: Freedom to make life choices'], color = 'red')

plt.xlabel('Countries')
plt.ylabel('Explained by: Freedom to make life choices')

plt.xticks(rotation = 90)


# In[ ]:


plt.figure(figsize = (10,5))

plt.bar(Last_10['Country'],Last_10['Explained by: Generosity'], color = 'red')

plt.xlabel('Countries')
plt.ylabel('Explained by: Generosity')

plt.xticks(rotation = 90)


# In[ ]:


plt.figure(figsize = (10,5))

plt.bar(Last_10['Country'],Last_10['Explained by: Perceptions of corruption'], color = 'red')

plt.xlabel('Countries')
plt.ylabel('Explained by: Perceptions of corruption')

plt.xticks(rotation = 90)


# In[ ]:


plt.figure(figsize = (1000,1000))

sns.pairplot(Data_2022)


# In[ ]:


fig = px.choropleth(Data_2022, locations = "Country",locationmode = 'country names', color = "Happiness score",
                    scope = 'world', title = "Happiness Ranking World Map", color_continuous_scale= "viridis")


# In[ ]:


fig


# # Conclusion
# 
# **Finland is the most happiest country in the world**
# 
# It has its GDP less then many countries
# It has a good level of Social support and Healthy life Expectancy
# It ranks 1 (among top 10 Countries) in terms of freedom to make speech and Preception of corruption
# It has lowest Generosity (among top 10 Countries)
# 
# **Afghanistan Rank last In terms of Happiness Score**
# 
# It ranks last in terms ofin terms of Freedom to make choices and Preceptions of Corruption
# It ranks 4th in Last 10 countries in terms of Generosity
# It ranks 5th in Last 10 countries in terms of Healthy life expectancy
# It ranks 3rd in Last 10 countries in terms of GDP per capita
# 
# **Out of all the attributes GDP, Freedom to make choices, Social Support, Life Expectancy and Perception of Corruption are the once that contribute most to the happiness score**
# 
# **Rank increases when GDP Decreases but there are some exceptions same is the case with Health Expectancy, Freedom to make choices and Social Support but Generosity and Perception of corruption is on the lower side for majority of the Countries.**
