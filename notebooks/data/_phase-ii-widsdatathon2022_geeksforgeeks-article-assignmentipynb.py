#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd


df = pd.read_csv('/kaggle/input/geeksforgeeks-articles/articles.csv')
df.dropna(inplace = True)


# In[ ]:


df


# In[ ]:


df.isnull().sum()


# ## 1) How many authors do we have in the dataset?

# In[ ]:


df['author_id'].describe()


# ## unique             5589
# 

# ## 2) How many easy articles are publised on GeeksforGeeks?

# In[ ]:


len(df[df['category'] == 'easy'])


# ## 9654
# 

# ## 3) Which category is having most number of articles?

# In[ ]:


df.groupby('category').size().sort_values(ascending = False)


# ## medium    10431
# 

# ## 4) What percent of articles are in expert category out of total number of articles?

# In[ ]:


round((len(df[df['category'] == 'expert'])/len(df))*100)


# ## 6%

# ## 5) Who has written most number of articles after GeeksforGeeks?

# In[ ]:


df.groupby('author_id').size().sort_values(ascending = False).head()


# ## ManasChhabra2      317
# 

# ## 6) Who has written most number of articles in Expert Category after GeeksforGeeks?

# In[ ]:


df[df['category'] == 'expert'].groupby('author_id').size().sort_values(ascending = False).head()


# ## mishrapriyank17     36
# 

# In[ ]:


df


# In[ ]:


df['last_updated'][0]


# In[ ]:


for date in df['last_updated']:
    if len(date)<12:
        df.drop(df[df['last_updated'] == date].index, inplace=True)


# In[ ]:


day = []
month = []
year = []
data = df.values
for date in data:
    day.append(date[2].split(" ")[0])
    month.append(date[2].split(" ")[1][:-1])
    year.append(date[2].split(" ")[2])


# In[ ]:


df['day'] = day
df['month'] = month
df['year'] = year


# In[ ]:


df['month']


# ## 7) Which day of the month has most number of articles published?

# In[ ]:


df.head()


# In[ ]:


df.groupby('day').size().sort_values(ascending = False)


# ## 28 day
# 

# ## 8) Which day of July has most number of articles published?

# In[ ]:


df.head()


# In[ ]:


df[df['month'] == 'Jul'].groupby('day').size().sort_values(ascending = False)


# ## 07 day of July
# 

# ## 9) Which month has least number of articles published?

# In[ ]:


df.head()


# In[ ]:


df.groupby('month').size().sort_values()


# ## march

# ## 10) Growth Rate of articles written 2021 as compare to 2020?

# ## Article written in 2020

# In[ ]:


l1 = len(df[df['year'] == '2020'])
l2 = len(df[df['year'] == '2021'])
print("2020::",l1)
print("2021::",l2)


# ## Article written in 2021

# In[ ]:


round((l2/l1)*100)


# 402 % is the growth rate

# In[ ]:




