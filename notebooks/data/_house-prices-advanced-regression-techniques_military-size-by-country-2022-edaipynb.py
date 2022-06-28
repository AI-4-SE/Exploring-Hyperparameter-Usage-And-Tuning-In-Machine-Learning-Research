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


df = pd.read_csv('/kaggle/input/military-size-by-country-2022/army-total-world.csv')


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.head()


# In[ ]:


df[['country','total']]


# In[ ]:


sns.set_theme()


# In[ ]:


df_sorted = df.sort_values(by=['total'], ascending=False)
plt.figure(figsize = (15,8))

ax = sns.barplot(x="country", y="total", data=df_sorted.head(20))
plt.xticks(rotation=90)


# In[ ]:




