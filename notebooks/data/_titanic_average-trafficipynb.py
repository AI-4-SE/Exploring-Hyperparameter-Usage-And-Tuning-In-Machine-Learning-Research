#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# In[ ]:


sumdf = pd.read_csv(r'/kaggle/input/fast-lane-prices/summary.csv',index_col=0)
sumdf.head(10)


# In[ ]:


datacols = [c for c in sumdf.columns if c != 'time']


# In[ ]:


smooth = sumdf[datacols].rolling(3).mean().ffill().bfill()
smoothavg = smooth.mean(axis=1)


# In[ ]:


x = list(smoothavg.index)
y = list(smoothavg.values)

fig, ax = plt.subplots(figsize=(30, 6),dpi=80)
ax.set_xlabel("Time of day")
ax.set_ylabel("Price [NIS]")

ax.plot(x,y);

skip = 6
ax.set_xticks(x[::skip])
ax.set_xticklabels(x[::skip], rotation=45)

plt.grid(axis="x")
plt.show()

