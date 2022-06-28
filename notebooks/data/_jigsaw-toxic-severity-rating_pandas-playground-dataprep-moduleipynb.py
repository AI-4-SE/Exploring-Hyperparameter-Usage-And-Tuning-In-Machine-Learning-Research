#!/usr/bin/env python
# coding: utf-8

# # Dataprep Python Module
# 
# Sample use of dateprep Python module. See more at https://dataprep.ai/

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Prerequisities
# 
# We need to install datapre module first.

# In[ ]:


get_ipython().system('pip install dataprep')


# ## Sample datapre data table overview

# In[ ]:


from dataprep.datasets import load_dataset
from dataprep.eda import plot
df = load_dataset("titanic")
plot(df, "Age")


# # Convert GPS to Lat/Lon

# In[ ]:


from dataprep.clean import clean_lat_long
df = pd.DataFrame({"coord": ["""0°25'30"S, 91°7'W""", """27°29'04.2"N   89°19'44.6"E"""]})

df2 = clean_lat_long(df, "coord", split=True)
# print(df2)


# In[ ]:


df2

