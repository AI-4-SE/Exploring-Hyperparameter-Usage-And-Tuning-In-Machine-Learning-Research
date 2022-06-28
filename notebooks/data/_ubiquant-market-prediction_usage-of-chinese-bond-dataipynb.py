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


cov_bonds = pd.read_csv('/kaggle/input/fixed-income-bonds-data-china/convertible_bonds.csv')
gen_bonds = pd.read_csv('/kaggle/input/fixed-income-bonds-data-china/general_bond.csv')


# In[ ]:


cov_bonds = cov_bonds[['date','ticker','open','high','low','close','volume']].sort_values('date')
gen_bonds = gen_bonds[['date','ticker','open','high','low','close','volume']].sort_values('date')


# In[ ]:


sym_cov = 'sh110038'
cov_bond_1 = cov_bonds[cov_bonds['ticker']==sym_cov]
cov_bond_1.index = cov_bond_1['date']
cov_bond_1['close'].plot(figsize=(8,6),xlabel='date',ylabel='price',title=sym_cov+" (convertible bond)")


# In[ ]:


sym_gen = 'sh010107'
gen_bond_1 =  gen_bonds[gen_bonds['ticker']==sym_gen]
gen_bond_1.index = gen_bond_1['date']
gen_bond_1['close'].plot(figsize=(8,6),xlabel='date',ylabel='price',title=sym_gen+" (general bond)")

