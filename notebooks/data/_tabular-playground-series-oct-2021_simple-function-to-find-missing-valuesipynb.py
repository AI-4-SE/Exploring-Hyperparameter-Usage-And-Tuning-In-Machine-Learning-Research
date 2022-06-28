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


# # **The first step we take after loading the data is to look at the nature of the data and are there missing values, but what will we do if the number of columns is too large?**

# In[ ]:


import pandas as pd


# In[ ]:


df=pd.read_csv("../input/titanic/train.csv")


# In[ ]:


df.head()


# In[ ]:


df.columns


# # Here it is clear that the number of columns is small and we can see all the values that are driven by "isnull":

# In[ ]:


df.isnull().sum()


# #  Let's take data with a larger number of columns and see what it would look like using "isnull":

# In[ ]:


df2=pd.read_csv("../input/tabular-playground-series-oct-2021/train.csv")


# In[ ]:


df2.columns


# # Here the number of columns is 287 and we notice that they do not appear all. We can solve this in several ways, but here I have added a simple code that helps us see only the columns that contain missing values

# In[ ]:


def NF(dataset):
    if max(dataset.isnull().sum())==0:
        print("There are no missing values")
    else:
        for i in dataset.columns:
            if dataset[i].isnull().sum()!=0:
                print("The number of missing values in column",i," :",dataset[i].isnull().sum())


# In[ ]:


NF(df)


# # Let's apply it to the second dataset

# In[ ]:


NF(df2)


# # > Hope this code is useful
