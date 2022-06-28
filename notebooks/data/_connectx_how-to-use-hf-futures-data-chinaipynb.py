#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


directory = "/kaggle/input/highfrequency-futures-data-china/minDataSum/"
future_symbol = ['PP', 'Y', 'OI', 'EB', 'NR', 'I', 'RM', 'CF', 'SN', 'TA', 'TF',
       'M', 'J', 'IF', 'AP', 'UR', 'ZC', 'JD', 'CJ', 'P', 'FU', 'PG',
       'CS', 'MA', 'TS', 'SS', 'IH', 'IC', 'AL', 'SC', 'NI', 'RB', 'HC',
       'RU', 'CU', 'V', 'PB', 'SP', 'ZN', 'FG', 'JM', 'CY', 'BU', 'SF',
       'AU', 'RR', 'C', 'FB', 'EG', 'T', 'L', 'B', 'AG', 'A', 'SM', 'SA',
       'SR']
futures_dict = {}
for symbol in future_symbol:
    individual_futures = pd.read_csv(directory + symbol + '.csv')
    futures_dict[symbol] = individual_futures


# In[ ]:


for sym in future_symbol:
    plt.figure()
    plt.rcParams.update({'font.size': 15})
    futures_dict[sym].index = futures_dict[sym]['Time']
    futures_dict[sym]['Close'].plot(title=sym,figsize=(20,8),xlabel='time',ylabel='price',fontsize=10)
    plt.show()

