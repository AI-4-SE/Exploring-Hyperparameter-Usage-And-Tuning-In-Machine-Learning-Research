#!/usr/bin/env python
# coding: utf-8

# **This notebook demonstrates how to simulate the online API in a local environment**

# In[ ]:


import pandas as pd, numpy as np, sys
sys.path.insert(0, '../input/jpx-local-api')
from local_api import local_api
from tqdm.auto import tqdm


# In[ ]:


sup_data = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv')


# **We directly make perfect predictions here for simplicity.**

# In[ ]:


help(local_api)


# In[ ]:


myapi = local_api('../input/jpx-tokyo-stock-exchange-prediction/supplemental_files')
# env = jpx_tokyo_market_prediction.make_env()
env = myapi.make_env()
iter_test = env.iter_test()
for (prices, options, financials, trades, secondary_prices, sample_prediction) in tqdm(iter_test):
    prices = sup_data.loc[sup_data['Date'] == prices['Date'].iloc[0]]
    all_codes = list(prices['SecuritiesCode']) # sample_prediction has mismatched indexes
    ranks = np.argsort(-np.array(prices['Target']))
    ranks = {all_codes[ranks[i]] : i for i, code in enumerate(all_codes)}
    sample_prediction['Rank'] = sample_prediction['SecuritiesCode'].map(ranks)
    env.predict(sample_prediction)
print(env.score())

