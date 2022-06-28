#!/usr/bin/env python
# coding: utf-8

# # Top Instagram Influencers

# In[ ]:


import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[ ]:


data=pd.read_csv('../input/top-instagram-influencers-data-cleaned/Top Instagram Influencers data (Cleaned).csv')
display(data[0:10])


# In[ ]:


display(data.info())


# In[ ]:


posts=[]
for item in data['Posts ']:
    if 'k' in item:
        posts+=[int(float(item.replace('k',''))*1000)]
    else:
        posts+=[int(item)]
data['Posts ']=posts


# In[ ]:


print(data.columns.tolist())
print(data['Country or Region '].unique().tolist())


# In[ ]:


data['Country or Region ']=data['Country or Region '].apply(lambda x:x.replace(' ','').replace('United','United '))


# In[ ]:


data1a=data[['Channel Info','Followers ']]
data1a=data1a.sort_values('Followers ',ascending=False)
display(data1a[0:5])
fig = px.bar(data1a[:30], x='Channel Info', y='Followers ',title="Followers Ranknig by Channel Info")
fig.show()


# In[ ]:


data['Number']=1
data1b=data[['Country or Region ','Number']]
data1b=data1b.groupby('Country or Region ',as_index=False)['Number'].sum()
data1b=data1b.sort_values('Number',ascending=False)
display(data1b[0:5])
fig = px.bar(data1b[:30], x='Country or Region ', y='Number',title="Numbers by Country or Region")
fig.show()


# In[ ]:


data1b=data[['Country or Region ','Influence score ']]
data1b=data1b.groupby('Country or Region ',as_index=False)['Influence score '].max()
data1b=data1b.sort_values('Influence score ',ascending=False)
display(data1b[0:5])
fig = px.bar(data1b[:30], x='Country or Region ', y='Influence score ',title="Max Influence score by Country or Region")
fig.show()


# In[ ]:


fig,ax = plt.subplots(figsize=(7,7))
ax.set_xlabel('Influence score ',fontsize=20)
ax.set_ylabel('Rank ',fontsize=20)
ax.scatter(data[ 'Influence score '],data['Rank '])
plt.show()


# In[ ]:


fig,ax = plt.subplots(figsize=(7,7))
ax.set_xlabel('Influence score ',fontsize=20)
ax.set_ylabel('Followers ',fontsize=20)
ax.scatter(data[ 'Influence score '],data['Followers '])
plt.show()


# In[ ]:


fig,ax = plt.subplots(figsize=(7,7))
ax.set_xlabel('Influence score ',fontsize=20)
ax.set_ylabel('Posts ',fontsize=20)
ax.scatter(data['Influence score '],data[ 'Posts '])
plt.show()


# In[ ]:


fig,ax = plt.subplots(figsize=(7,7))
ax.set_xlabel('Influence score ',fontsize=20)
ax.set_ylabel('Total Likes',fontsize=20)
ax.scatter(data[ 'Influence score '],data[ 'Total Likes'])
plt.show()


# ## Top 10 Influencers

# In[ ]:


names=data['Channel Info'].tolist()
for i in range(10):
    print('https://www.instagram.com/'+names[i])


# In[ ]:





# In[ ]:





# In[ ]:




