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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#Reading the datta and making dataframe df
df = pd.DataFrame(pd.read_csv('/kaggle/input/inc-5000-companies/INC 5000 Companies 2019.csv'))
df


# In[ ]:


#getting info of df
df.info()


# In[ ]:


#using loop to make a new column of numeric revenue [' new_rev']
A=[]
for i in range(len(df)):
    A.append(float(df['revenue'][i].split()[0]))
df['new_rev'] = A


# In[ ]:


#plot graph of Mean Revenue Among Indusries

plt.figure(figsize=(10,8))
Axes=sns.barplot(x='industry',y='new_rev',data=df,ci=None)
Axes.set_xlabel("Industry",fontsize=20)
Axes.set_ylabel("Revenue",fontsize=20)
Axes.set_title('Mean Revenue Among Industries',fontsize=20)

Axes.set_xticklabels(Axes.get_xticklabels(), rotation=50, horizontalalignment='right');


# In[ ]:


#Q1 - What's the average revenue among companies on the list? Broken down by industry?
# => 31 Million
plt.figure(figsize=(10,8))
df['new_rev'].mean()


# In[ ]:


#Q2 - Which industries are most and least represented in the list?

plt.figure(figsize=(12,8))
plt.xlabel("Industry",fontsize=20)
plt.ylabel("Counts on list",fontsize=20)
plt.title("Counts of Industries on List",fontsize=20)
df['industry'].value_counts().plot.bar();


# In[ ]:


#Q - Which industries saw the largest average growth rate?
# => Logistic and transportation
plt.figure(figsize=(12,8))
Ax=sns.barplot(x='industry',y='growth_%',data=df,ci=None)
Ax.set_xlabel("Industry",fontsize=20)
Ax.set_ylabel("Average Growth",fontsize=20)
Ax.set_title('Average Growth Rate of Industries',fontsize=20)

Ax.set_xticklabels(Ax.get_xticklabels(), rotation=50, horizontalalignment='right');


# In[ ]:


df['New_hires'] = df['workers'] - df['previous_workers']
Sorted_df = df.sort_values(by=['New_hires'], ascending=False)
Sort_df=Sorted_df.head(10)


# In[ ]:


#Q - Which companies had the largest increase in staff/new hires?
# =>Allied Universal
plt.figure(figsize=(12,8))
Axy=sns.barplot(x='name',y='New_hires',data=Sort_df,ci=None)
Axy.set_xlabel("Company",fontsize=20)
Axy.set_ylabel("Staff",fontsize=20)
Axy.set_title('Company wise Increase in Staff',fontsize=20)
Axy.set_xticklabels(Axy.get_xticklabels(), rotation=50, horizontalalignment='right');


# In[ ]:


#Q - Did any companies increase revenue while reducing staff?
# Help me out here...


# In[ ]:




