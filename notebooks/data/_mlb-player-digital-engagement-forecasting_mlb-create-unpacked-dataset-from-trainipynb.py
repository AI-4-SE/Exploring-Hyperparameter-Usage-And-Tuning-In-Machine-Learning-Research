#!/usr/bin/env python
# coding: utf-8

# Dataset Link: https://www.kaggle.com/mrutyunjaybiswal/mlbunpackedtrainjson

# In[ ]:


import os
import gc
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# paths
inp_folder = "../input/mlb-player-digital-engagement-forecasting"

train_ = os.path.join(inp_folder, "train.csv")
test_ = os.path.join(inp_folder, "example_test.csv")


# In[ ]:


sample_test = pd.read_csv(test_)
sample_test


# In[ ]:


sample_test.info()


# In[ ]:


test_cols_ = sample_test.set_index('date').columns.to_list()
train_cols_ = ['nextDayPlayerEngagement'] + test_cols_


# In[ ]:


def unpack_json(json_str, date):
    res_df = pd.DataFrame()
    
    if pd.isna(json_str):
        return res_df
    else:
        res_df = pd.read_json(json_str)
        
    res_df['pk_date'] = date
    
    return res_df


def unpack_data(data, dfs=None, n_jobs=-1):
    if dfs is not None:
        data = data.loc[:, dfs]
    unnested_dfs = {}
    for name, column in tqdm(data.iteritems(),total=len(data.columns)):
        daily_dfs = Parallel(n_jobs=n_jobs)(
            delayed(unpack_json)(item, date) for date, item in column.iteritems())
        df = pd.concat(daily_dfs)
        unnested_dfs[name] = df.set_index("pk_date")
        del data[name], df
        _ = gc.collect()
        
    return unnested_dfs


# In[ ]:


train = pd.read_csv(train_)
storage_dict = unpack_data(train.set_index('date'), train_cols_)

if not os.path.exists("./unpacked_train"):
    os.makedirs("./unpacked_train")
    

for col in train_cols_:
    storage_dict[col].to_csv(f"./unpacked_train/train_{col}.csv")


# In[ ]:


get_ipython().system('cp -r -v ../input/mlb-player-digital-engagement-forecasting/* ./')

