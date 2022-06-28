#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date, timedelta
import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv(
    '../input/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    parse_dates=["date"],
)


# In[ ]:


print( train.shape )
train.tail()


# In[ ]:


item_dates = train[['date','item_nbr']]


# In[ ]:


print( item_dates.shape )
item_dates.tail()


# In[ ]:


item_first_dates = item_dates.groupby('item_nbr').min().sort_values('date')
print( item_first_dates.shape )
item_first_dates.tail()


# In[ ]:


item_dates_7 = item_first_dates['date'] + pd.DateOffset(days=7)
item_dates_7.tail()


# In[ ]:


item_dates_8 = item_first_dates['date'] + pd.DateOffset(days=8)
item_dates_9 = item_first_dates['date'] + pd.DateOffset(days=9)
item_dates_10 = item_first_dates['date'] + pd.DateOffset(days=10)
item_dates_11 = item_first_dates['date'] + pd.DateOffset(days=11)
item_dates_12 = item_first_dates['date'] + pd.DateOffset(days=12)
item_dates_13 = item_first_dates['date'] + pd.DateOffset(days=13)
item_dates_14 = item_first_dates['date'] + pd.DateOffset(days=14)
item_dates_14.shape


# In[ ]:


item_dates_2nd_week = pd.DataFrame( pd.concat([item_dates_7, item_dates_8, item_dates_9, 
                                               item_dates_10, item_dates_11, item_dates_12, 
                                               item_dates_13], axis=0)).reset_index()
print( item_dates_2nd_week.shape )
item_dates_2nd_week.head()


# In[ ]:


train_2nd_week = train.merge( item_dates_2nd_week, on=['date','item_nbr'], how='inner')
train_2nd_week.shape


# In[ ]:


train_2nd_week.head()


# In[ ]:


train_2nd_week.to_csv('train_2nd_week.csv', float_format='%.1f', index=None)


# In[ ]:


item_dates_2nd_week.to_csv('items_2nd_week.csv', index=None)


# In[ ]:


items_train = train[['item_nbr']].drop_duplicates().item_nbr.values
items_test = pd.read_csv( "../input/test.csv", usecols=[3] ).drop_duplicates().item_nbr.values
items_new = set(items_test) - set(items_train)
new_df = pd.DataFrame([i for i in items_new])
new_df.columns=['item_nbr']
print(new_df.shape)
new_df.head()


# In[ ]:


new_df.to_csv('new_items.csv',index=None)


# In[ ]:




