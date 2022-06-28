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


df = pd.read_csv('/kaggle/input/cancer-trials-in-the-united-statese/fullData.csv')
df.head()
# full data is the file I will clean and visualize, first need to drop the zip codes, index, countycode, county 
# then clean out duplicate rows
df.drop(columns=['index','zipCode','countyCode','studyCount','County','Name'], inplace=True)


# In[ ]:


df = df.drop_duplicates()


# In[ ]:


# df.reset_index(inplace=True)
# df.drop(columns=['level_0','index'],inplace=True)
# recTrend is the cancer mortality recent trend
# recentTrend is cancer incidence recent trend
# looks like this dataset is ripe for seaborn visual analysis


# In[ ]:


df_dict = pd.read_csv('/kaggle/input/cancer-trials-in-the-united-statese/data_dict.csv')
df_dict
# Columns need more descriptive names


# In[ ]:


df.rename(columns={'PovertyEst':'poverty_pop', 'recentTrend':'inc_recent_trend','fiveYearTrend':'inc_five_year_trend','deathRate':'death_per_k',
                  'recTrend':'cx_mort_recent_trend'}, inplace=True)
df.columns


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# seaborn charts I want to try:
# 1) scatter
# 2) displot
# 3) boxplot
# 4) catplot
# 5) violinplot
# 6) histplot
# 7) stripplot
# 8) jointgrid
# 9) relplot
# 10) pairplot

# In[ ]:


df.describe().round(2)
# the describe gives a lot of ideas of data distribution to visualize
# first one is the poverty population IQR per state


# BOXPLOT
# x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, width=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5, ax=None

# In[ ]:


# boxplot for poverty IQR by state
plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.boxplot(x='State', y='povertyPercent', data=df)
# I want to order this by the mean poverty percent by state


# If I created quartiles for which states had the highest avg poverty, I bet you wouldn't be surprised

# 

# In[ ]:


pov_order = df.groupby(by='State').povertyPercent.mean().sort_values().index


# In[ ]:


# boxplot for poverty IQR by state
plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.boxplot(x='State', y='povertyPercent', data=df, order=pov_order)
# I want to add a label for the quartile poverty percent the county is in nationally


# In[ ]:


# how do I add the ntile for a value
# df['Quantile_rank'] = pd.qcut(df['Score'], 4,labels = False)
df['poverty_quartiles'] = pd.qcut(df['povertyPercent'], 4, labels=False)
# this is the county quartile, I need the state quartile


# In[ ]:


state_poverty = pd.qcut(df.groupby(by='State').povertyPercent.mean(), 4, labels=False)


# In[ ]:


state_poverty = pd.DataFrame(state_poverty)
# making the series into a dataframe to rejoin back into data for state poverty ranking


# In[ ]:


# renaming column for rejoining to main dataframe
state_poverty.rename(columns={'povertyPercent':'state_poverty_quartile'}, inplace=True)


# In[ ]:


df = df.merge(state_poverty, how='inner', on='State')
#df.drop(columns='state_poverty_quartile', inplace=True)


# In[ ]:


# boxplot for poverty IQR by state
plt.figure(figsize=(14,6))
plt.xticks(rotation=90)
sns.boxplot(x='State', y='povertyPercent', hue='state_poverty_quartile', 
            data=df[df.state_poverty_quartile == 0])


# In[ ]:


# boxplot for poverty IQR by state
plt.figure(figsize=(14,6))
plt.xticks(rotation=90)
sns.boxplot(x='State', y='povertyPercent', hue='state_poverty_quartile', 
            data=df[df.state_poverty_quartile == 3])


# g = sns.JointGrid(data=penguins, x="bill_length_mm", y="bill_depth_mm")
# g.plot_joint(sns.scatterplot, s=100, alpha=.5)
# g.plot_marginals(sns.histplot, kde=True)

# In[ ]:


# I can do much more with this data set, I added calculations onto it to give extra valu
g = sns.JointGrid(data=df, x='povertyPercent',y='incidenceRate')
g.plot_joint(sns.scatterplot)
g.plot_marginals(sns.histplot, kde=True)


# In[ ]:


df.describe()
sns.jointplot(data=df, x='povertyPercent',y='death_per_k', kind='hex')
# this is simpler


# In[ ]:


# hypothesis list would help guide graph creation
df.describe()
sns.jointplot(data=df, x='povertyPercent', y='incidenceRate', kind='reg')


# In[ ]:


df.describe()
f, ax = plt.subplots(figsize=(12,6))
sns.despine(f)
sns.histplot( data=df[df.incidenceRate < 600], x='incidenceRate', hue='poverty_quartiles',
             multiple='stack', palette='light:m_r',
             edgecolor='.3', linewidth=.75
)


# In[ ]:


# few outliers for incidenceRate
high_cx_inc = df[df.incidenceRate > 600]
high_cx_inc.describe()


# In[ ]:


g = sns.catplot(data=df, kind='bar',
                x='state_poverty_quartile', y='incidenceRate',
                hue='cx_mort_recent_trend', ci='sd', palette='dark',
                alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels('', 'Cancer Incidence Rate')
# no valuable differences between this and everything else


# In[ ]:


df.columns
#need a small set of catagories if I want to do a displot
# also need a histogram category
sns.displot(data=df, x='incidenceRate', col='state_poverty_quartile',
           row='poverty_quartiles', facet_kws=dict(margin_titles=True))
# does this compare the state vs county incidence rate well?
# in the [0,0] quartile, there is a normal distribution because the avg of the counties is in that quartile


# In[ ]:


df.columns


# In[ ]:


# violin plot would be interesting to see the distribution of poverty amongst the state poverty slices
plt.figure(figsize=(12,6))
sns.violinplot(data=df[df.state_poverty_quartile == 3], x='State', y='povertyPercent',
              split=True, inner='quart', linewidth=1)


# f, ax = plt.subplots(figsize=(12,6))
# sns.despine(f)
# sns.histplot( data=df[df.incidenceRate < 600], x='incidenceRate', hue='poverty_quartiles',
#              multiple='stack', palette='light:m_r',
#              edgecolor='.3', linewidth=.75
# )

# In[ ]:


f, ax = plt.subplots(figsize=(12,6))
sns.histplot(data=df[df.avgAnnCount < 2000], x='avgAnnCount',hue='poverty_quartiles',
            multiple='stack')


# histograms are a good way of finding exceptional data to examine, I can ask questions about the counties with annual count > 2000

# In[ ]:


highAvgAnnCount = df[df.avgAnnCount > 2000]


# In[ ]:


highAvgAnnCount.shape[0] / df.shape[0]
#5 percent of counties are in the long right skewed tail


# In[ ]:


highAvgAnnCount.poverty_quartiles.value_counts().plot(kind='bar')


# In[ ]:




