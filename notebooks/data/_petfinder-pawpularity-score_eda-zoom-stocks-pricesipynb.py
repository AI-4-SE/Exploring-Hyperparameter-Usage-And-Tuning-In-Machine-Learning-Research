#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv(r'/kaggle/input/zoom-stocks/Zoom.csv')


# # Preparation

# Lets start with checking nan values

# In[ ]:


df.isna().sum()


# Checking datatypes

# In[ ]:


df.info()


# Change object type to date for more convenient usage

# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])


# Extract some more data from date for deeper analysis

# In[ ]:


df['year'] = pd.DatetimeIndex(df['Date']).year
df['month'] = pd.DatetimeIndex(df['Date']).month
df['month_year'] = df['Date'].dt.to_period('M')
df.head()


# # Visualization

# (my favourite part)

# First of all lets look at the stocks prices during dates dataset provides

# In[ ]:


df[['Adj Close','Date']].groupby('Date').mean().plot(figsize=(12,8), color = 'teal', grid = True)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stocks Prices each Year')
plt.show()


# Lets check out what are the differences between 'Open','High','Low','Close' prices grouping by year

# In[ ]:


stocks_year = df.groupby(by='year').mean()
stocks_year[['Open','High','Low','Close']].plot(kind = 'bar', figsize = (12,8), 
        grid = True, color = ['teal','aquamarine','lightseagreen','turquoise'])
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Open, High, Low and Close prices each Year')
plt.show()


# Pandemic increased stocks prices significantly. Now lets do the same analysis but grouping by months

# In[ ]:


stocks_month = df.groupby(by='month').mean()
stocks_month[['Open','High','Low','Close']].plot(kind = 'bar', figsize = (12,8), 
        grid = True, color = ['teal','aquamarine','lightseagreen','turquoise'])
plt.xlabel('Month')
plt.ylabel('Price')
plt.title('Average Stocks Price each Month')
plt.show()


# Looks like semptember - october are the most 'valuable' months. I consider this is due to new semesters at schools and universities and the necessity to purchase software. Now lets go a bit deeper and count the difference between open and close price

# In[ ]:


pd.options.mode.chained_assignment = None
open_close = df[['month_year','Open','Close','year']]
open_close['start_end_changes'] = open_close['Open'] - open_close['Close']
open_close.groupby('year').mean()['start_end_changes'].plot(kind = 'bar', grid = True,
    figsize = (12,8), color = 'teal')
plt.xlabel('Year')
plt.ylabel('Price Changes')
plt.title('Open and Close Stocks Prices Changes Annualy')
plt.show()


# Looks like the biggest difference between start and final price for a day were in 2020. Lets go deeper to 2020

# In[ ]:


pd.options.mode.chained_assignment = None
open_close_2020 = df[['month','Open','Close','year']]
open_close_2020['start_end_changes'] = open_close_2020['Open'] - open_close_2020['Close']
open_close_2020[open_close_2020['year']==2020].groupby('month').mean()['start_end_changes'].plot(kind = 'bar', grid = True, figsize = (12,8),
        color = 'teal')
plt.xlabel('Month')
plt.ylabel('Price Changes')
plt.title('Open and Close Stocks Prices Changes in 2020 each Month')
plt.show()


# 2020 was rather unstable. And what do we have with volume...

# In[ ]:


df[['month_year','Volume']].groupby('month_year').mean().plot(figsize = (12,8), color = 'teal', grid = True,)
plt.xlabel('month-year')
plt.ylabel('volume in billions')
plt.title('Volume Changes')
plt.show()


# Now lets check out what is the lowest and highest prices grouped by year

# In[ ]:


df[['year','High','Low']].groupby('year').mean().plot(kind = 'bar', 
        figsize = (12,8), grid = True, color = ['teal','aquamarine'])
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('The Hieghest and Lowest Average Stocks Prices per Year')
plt.show()


# # Conclusion

# As we can see on the plots, pandemic made Zoom issue shares at the very first months, due to which their prices was unstable. Rather obvious conclusion but visually demostrated.
# As usual, thank the author of the dataset. The data is well structured and organized. 
# If you find this notebook helpful, feel free to upvote it, I really aprecite it. Good luck!
