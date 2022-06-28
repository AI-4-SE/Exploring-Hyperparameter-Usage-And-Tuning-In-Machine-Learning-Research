#!/usr/bin/env python
# coding: utf-8

# In this notebook, test trips are divided into public & pribvate and visualized.  
# 
# baseline submission reference source. (by @saitodevel01)  
# https://www.kaggle.com/code/saitodevel01/gsdc2-baseline-submission

# In[ ]:


from pathlib import Path
import numpy as np
import pandas as pd
import copy
import plotly.express as px
import plotly.graph_objects as go
import pyproj
import json
import bisect

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import warnings
warnings.simplefilter('ignore')
pd.set_option('display.max_rows',10)
pd.set_option('display.max_columns',None)


# In[ ]:


def visualize_trafic(df, center={"lat":37.5, "lon":-122.1}, zoom=9):
    fig = px.scatter_mapbox(df,
                            
                            # Here, plotly gets, (x,y) coordinates
                            lat="LatitudeDegrees",
                            lon="LongitudeDegrees",
                            
                            #Here, plotly detects color of series
                            color="tripId",
                            labels="tripId",
                            
                            zoom=zoom,
                            center=center,
                            height=500,
                            width=1000)
    fig.update_layout(mapbox_style='stamen-terrain')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()


# In[ ]:


# https://www.kaggle.com/code/saitodevel01/gsdc2-baseline-submission
df_sub = pd.read_csv('../input/gsdc2-baseline-submission/submission.csv')


# In[ ]:


df_sub.head()


# In[ ]:


public_trips = [
    '2021-06-22-US-MTV-1/XiaomiMi8',
    '2021-08-12-US-MTV-1/GooglePixel4',
    '2021-08-17-US-MTV-1/GooglePixel5',
    '2021-08-24-US-SVL-2/GooglePixel5',
    '2021-09-07-US-MTV-1/SamsungGalaxyS20Ultra',
    '2021-09-14-US-MTV-1/GooglePixel5',
    '2021-09-20-US-MTV-1/XiaomiMi8',
    '2021-09-20-US-MTV-2/GooglePixel4',
    '2022-01-04-US-MTV-1/SamsungGalaxyS20Ultra',
    '2022-01-11-US-MTV-1/GooglePixel6Pro',
    '2022-01-26-US-MTV-1/XiaomiMi8',
    '2022-02-01-US-SJC-1/XiaomiMi8',
    '2022-02-08-US-SJC-1/XiaomiMi8',
    '2022-02-15-US-SJC-1/GooglePixel5',
    '2022-02-23-US-LAX-1/GooglePixel5',
    '2022-02-23-US-LAX-3/XiaomiMi8',
    '2022-02-23-US-LAX-5/XiaomiMi8',
    '2022-02-24-US-LAX-1/SamsungGalaxyS20Ultra',
    '2022-02-24-US-LAX-3/XiaomiMi8',
    '2022-02-24-US-LAX-5/SamsungGalaxyS20Ultra',
    '2022-03-14-US-MTV-1/GooglePixel5',
    '2022-03-17-US-SJC-1/GooglePixel5',
    '2022-03-31-US-LAX-3/SamsungGalaxyS20Ultra',
    '2022-04-01-US-LAX-1/SamsungGalaxyS20Ultra',
    '2022-04-01-US-LAX-3/XiaomiMi8',
    '2022-04-22-US-OAK-1/GooglePixel5',
    '2022-04-22-US-OAK-2/XiaomiMi8',
    '2022-04-25-US-OAK-2/GooglePixel4'
]

private_trips = [
    '2021-04-28-US-MTV-2/SamsungGalaxyS20Ultra',
    '2021-09-28-US-MTV-1/GooglePixel5',
    '2021-11-05-US-MTV-1/XiaomiMi8',
    '2021-11-30-US-MTV-1/GooglePixel5',
    '2022-01-18-US-SJC-2/GooglePixel5',
    '2022-03-22-US-MTV-1/SamsungGalaxyS20Ultra',
    '2022-03-31-US-LAX-1/GooglePixel5',
    '2022-04-25-US-OAK-1/GooglePixel5',
]


# In[ ]:


df_public = df_sub[df_sub['tripId'].isin(public_trips)].reset_index(drop = True)
df_private = df_sub[df_sub['tripId'].isin(private_trips)].reset_index(drop = True)


# In[ ]:


# Check private ratio
# This leaderboard is calculated with approximately 80% of the test data. The final results will be based on the other 20%, so the final standings may be different.

print('public ratio :',round(len(df_public)/len(df_sub),3))
print('private ratio :',round(len(df_private)/len(df_sub),3))


# # Visualize Public Trip

# In[ ]:


visualize_trafic(df_public)


# # Visualize Private Trip

# In[ ]:


visualize_trafic(df_private)


# ### Make override submission

# In[ ]:


def override_lad(df,tripId):
    df_override = df.copy()
    df_override.loc[df_override['tripId'] == tripId,'LatitudeDegrees'] = 0
    return df_override


# In[ ]:


df_sub_override = df_sub.copy()

for tripId in private_trips:
    df_sub_override = override_lad(df_sub_override,tripId)
    print(f'override:{tripId}')


# In[ ]:


df_sub_override.head()


# ### Submit override prediction and check LB.

# In[ ]:


df_sub_override.to_csv('df_sub_override',index = False)

