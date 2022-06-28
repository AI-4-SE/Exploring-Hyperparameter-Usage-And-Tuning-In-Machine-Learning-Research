#!/usr/bin/env python
# coding: utf-8

# # **Setting up the environment**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Library for Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # **Import data**

# In[ ]:


pay_history=pd.read_csv("../input/uber-driver-dataset/Driver Payment.csv")
pay_history.head()


# In[ ]:


pay_history.info()


# In[ ]:


pay_history['Local Timestamp'] = pd.to_datetime(pay_history['Local Timestamp'])
pay_history['City Id'] = pay_history['City Id'].astype(str)


# In[ ]:


pay_day = pay_history.groupby([pay_history['Local Timestamp'].dt.date]).sum()
pay_day.head()


# # **Data Visualisation**

# In[ ]:


plt.figure(figsize=(20,10))
plt.title("Total earned per day")
sns.lineplot(data=pay_day,x='Local Timestamp',y='Local Amount')

