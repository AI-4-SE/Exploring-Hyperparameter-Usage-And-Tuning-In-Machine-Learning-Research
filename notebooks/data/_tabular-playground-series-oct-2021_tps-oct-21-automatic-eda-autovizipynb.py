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


get_ipython().run_cell_magic('capture', '', '# Autoviz for automatic EDA\n!pip install xlrd\n!pip install autoviz\nfrom autoviz.AutoViz_Class import AutoViz_Class\n')


# In[ ]:


train=pd.read_csv("../input/tabular-playground-series-oct-2021/train.csv")
test=pd.read_csv("../input/tabular-playground-series-oct-2021/test.csv")


# In[ ]:


AV = AutoViz_Class()
df = AV.AutoViz(filename="", sep=',', depVar='target', dfte=train, header=0, verbose=1, lowess=False, 
                chart_format='svg',  max_cols_analyzed=15)


# # If you liked the notebook, please consider giving an upvote. 

# In[ ]:




