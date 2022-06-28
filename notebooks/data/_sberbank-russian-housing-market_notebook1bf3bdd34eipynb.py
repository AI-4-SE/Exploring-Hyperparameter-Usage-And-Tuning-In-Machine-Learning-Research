#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().system('unzip /kaggle/input/sberbank-russian-housing-market/train.csv.zip')


# In[ ]:


dataset = pd.read_csv(r'./train.csv')
pd.set_option('display.max_columns', 300)
dataset.head()


# In[ ]:


dataset.timestamp


# In[ ]:


dataset.shape


# In[ ]:


get_ipython().system('unzip  /kaggle/input/sberbank-russian-housing-market/macro.csv.zip')


# In[ ]:


micro = pd.read_csv(r'./macro.csv')
pd.set_option('display.max_columns', 300)
micro.head()


# In[ ]:


micro.columns

