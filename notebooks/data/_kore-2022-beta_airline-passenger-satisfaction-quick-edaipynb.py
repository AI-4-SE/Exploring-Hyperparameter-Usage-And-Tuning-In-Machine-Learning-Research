#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load data

# In[ ]:


df = pd.read_csv('/kaggle/input/airline-passenger-satisfaction/airline_passenger_satisfaction.csv')


# In[ ]:


df.describe()


# # Data vis
# 
# How old are the passengers ?

# In[ ]:


chart = sns.countplot(x="Age", data=df)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45);


# Let's look at the genders

# In[ ]:


chart = sns.countplot(x="Gender", data=df)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45);


# There are more or less the same amount of male and female in this dataset.
# 
# Type of airline customer (First-time/Returning) :

# In[ ]:


chart = sns.countplot(x="Customer Type", data=df)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45);


# Purpose of the flight ?

# In[ ]:


chart = sns.countplot(x="Type of Travel", data=df)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45);


# In[ ]:


chart = sns.countplot(x="Class", data=df)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45);


# What is the distance traveled ?

# In[ ]:


chart = sns.countplot(x="Flight Distance", data=df)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45);


# In[ ]:


sns.barplot(x="Gender", y="In-flight Wifi Service", data=df)


# In[ ]:


sns.barplot(y="Flight Distance", x="In-flight Wifi Service", data=df)


# In[ ]:


g = sns.jointplot(
    data=df,
    x="Flight Distance", y="In-flight Wifi Service", hue="Gender",
    kind="kde",
)


# In[ ]:


sns.pairplot(df, hue="Gender")


# In[ ]:




