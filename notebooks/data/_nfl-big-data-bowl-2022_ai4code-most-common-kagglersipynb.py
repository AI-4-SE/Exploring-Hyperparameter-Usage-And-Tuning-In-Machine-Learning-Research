#!/usr/bin/env python
# coding: utf-8

# - Let's see most common kagglers in AI4Code Dataset
# - Enjoy :)

# In[ ]:


import os
import re
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from collections import Counter


# In[ ]:


def extract_user(file):
    json_open = open(f'../input/AI4Code/train/{file}', 'r')
    json_load = json.load(json_open)
    json_load = '\n'.join(json_load['source'].values())
    res = re.findall(r'www.kaggle.com/+[a-zA-Z0-9_]+/', json_load)
    res = set([r.split('/')[-2] for r in res])
    res = [r for r in res if r not in ['c', 'kernels', 'learn']]
    return res

files = os.listdir('../input/AI4Code/train')
result = Parallel(n_jobs=4, verbose=1)(delayed(extract_user)(file) for file in files)
result = sum(result, [])
count = Counter(result)


# In[ ]:


# TOP30
for i, c in enumerate(count.most_common(30)):
    print(i+1, c)


# - Glad to know I'm 23rd :)
