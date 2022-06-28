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
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session.
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt 

income_statement = pd.read_csv('/kaggle/input/chinese-income-statement-2006-2022/Chinese_Income_Statement.csv')
print(income_statement)
print(income_statement.columns.tolist())
sub_1 = ['Date','net profit growth', 'total operating income growth']
sub_2 = ['Date', 'ticker', 'net profit',  'Total operating income',  'total operating expense-operating expense', 'total operating expense-sales expense', 'total operating expense-management expense', 'total operating expense-financial expense', 'total operating expense-total opearting expense', 'opearting profit', 'total profit']
ticker_mapping = pd.read_csv('/kaggle/input/chinese-income-statement-2006-2022/map_ticker_to_company.csv')
tickers = income_statement['ticker'].to_numpy()
for i in range(0,10):
    figure(figsize=(8, 6))
    first = income_statement[income_statement['ticker']==tickers[i]]
    first = first.fillna(method='ffill').fillna(method='backfill')
    first_pct = first[sub_1]
    first_val = first[sub_2]
    first_pct.index = first_pct['Date']
    first_val.index = first_val['Date']
    first_pct = first_pct.drop('Date', axis=1)
    first_val = first_val.drop('Date', axis=1)
    first_pct.plot(figsize=(10,8),title=tickers[i])
    first_val.plot(figsize=(10,8),title=tickers[i])
    plt.show()

