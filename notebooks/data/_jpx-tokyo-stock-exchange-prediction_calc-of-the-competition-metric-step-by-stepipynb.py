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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


df  = pd.read_csv('/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv', parse_dates=True)
df.head()


# In[ ]:


df = df.loc[df["Date"]>= "2021-01-01"]
df


# In[ ]:


# getting close prices for all SecuritiesCodes
#prices = df.pivot(index='Date', columns='SecuritiesCode', values='Close')
prices_target = df.pivot(index='Date', columns='SecuritiesCode', values='Target')
prices_target.head()


# In[ ]:


# target calculation with shifting 2 days for all SecuritiesCodes
#prices_target = (prices.shift(-2) - prices.shift(-1)).div(prices.shift(-1))
#prices_target.head()


# In[ ]:


#prices_target.iloc[-2:]


# In[ ]:


# we can directly add last two days from df for because we shift 2 days 
#prices_target[prices_target.index=="2021-12-02"] = df[df["Date"]=="2021-12-02"]["Target"]
#prices_target[prices_target.index=="2021-12-03"] = df[df["Date"]=="2021-12-03"]["Target"]
#prices_target.head()


# In[ ]:


#len(prices_target.index)


# In[ ]:


# first 200 values of SecuritiesCodes for every time step(1202).
first_200 = []
for i in range(0,len(prices_target.index),1):
      first_200.append(-np.sort(-prices_target.iloc[i, :].values)[:200])
prices_target_first_200 = pd.DataFrame(first_200,index=prices_target.index)
prices_target_first_200.head()


# In[ ]:


# last 200 values of SecuritiesCodes for every time step(1202).
last_200 = []
for i in range(0,len(prices_target.index),1):
      last_200.append(np.sort(prices_target.iloc[i, :].values)[0:200])
prices_target_last_200 = pd.DataFrame(last_200,index=prices_target.index)
prices_target_last_200.head()


# In[ ]:


weights = np.linspace(start=2, stop=1, num=200)
weights


# In[ ]:


Sup=((prices_target_first_200 * weights).sum(axis = 1))/np.mean(weights)
Sup


# In[ ]:


Sdown=((prices_target_last_200 * weights).sum(axis = 1))/np.mean(weights)
Sdown


# In[ ]:


(Sup - Sdown)


# In[ ]:


sharpe_ratio = (Sup - Sdown).mean()/(Sup - Sdown).std()
sharpe_ratio


# **How many of each SecuritiesCode(2000) was used during the entire time?**

# In[ ]:


prices_target=prices_target.replace(np.nan, 100) # get rid of NaN for counting


# In[ ]:


count_SecuritiesCode_first_200 = prices_target.isin(prices_target_first_200.values.flatten())
True_first_200 = (count_SecuritiesCode_first_200.apply(pd.Series.value_counts, axis=0).fillna(0).iloc[1:2]).T
True_first_200 = True_first_200.add_prefix('first_200_')


# In[ ]:


count_SecuritiesCode_last_200 = prices_target.isin(prices_target_last_200.values.flatten())
True_last_200 = (count_SecuritiesCode_last_200.apply(pd.Series.value_counts, axis=0).fillna(0).iloc[1:2]).T
True_last_200 = True_last_200.add_prefix('last_200_')


# In[ ]:


Count = pd.concat([True_first_200,True_last_200],axis=1)


# In[ ]:


Count


# In[ ]:


SecuritiesCode_weight = (Count/len(prices_target.index)).sum(axis=1)/2


# In[ ]:


df_weight = pd.DataFrame(SecuritiesCode_weight,columns=["Weight"]).reset_index()


# In[ ]:


df_weight


# In[ ]:


plt.figure(figsize=(10, 6), dpi=80)
plt.scatter(x=SecuritiesCode_weight.index,y=SecuritiesCode_weight)

