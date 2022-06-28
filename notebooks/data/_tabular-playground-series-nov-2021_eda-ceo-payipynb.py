#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[ ]:


top250 = pd.read_csv("/kaggle/input/highest-paid-ceos-total-compensation/CEO_largestrevenue_highestpaid_2020-21.csv")
top50 = pd.read_csv("/kaggle/input/highest-paid-ceos-total-compensation/CEO_compensation_top50_2020.csv")


# In[ ]:


top250.head


# # Company Names Wordcloud

# In[ ]:


text = " ".join(review for review in top250.company.astype(str))
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400).generate(text)
plt.axis("off")
plt.tight_layout(pad=0)
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()


# In[ ]:


rev_comp = top250[['ceoPay','revenue']]


# In[ ]:


rev_comp


# # Revenue of Company vs CEO Pay

# In[ ]:


rev_comp.plot(x ='revenue', y='ceoPay', kind = 'scatter')
plt.title('Revenue of Company vs CEO Pay')
plt.show()


# In[ ]:


top250.sort_values('ceoPay',ascending=False)


# # Seems like CEO of Palantir is the outlier

# In[ ]:


rev_comp2 = top250.sort_values('ceoPay',ascending=False)

rev_comp3 =rev_comp2[['ceoPay','revenue']]
rev_comp3 


# In[ ]:


rev_comp4 = rev_comp3.iloc[10: , :]
rev_comp4


# # get rid of 10 outliers

# In[ ]:


rev_comp4.plot(x ='revenue', y='ceoPay', kind = 'scatter')
plt.title('Revenue of Company vs CEO Pay')
plt.show()


# ## company revenue doesnt have much affect on CEO's salary, even companies with low revenue have CEOs that much more

# In[ ]:


pay_ratio = top250.sort_values('payRatio',ascending=False)


# In[ ]:


pay_ratio.head(20)


# ## the CEO to employee pay ratio may be skewed due to the lost hours/jobs during the start of covid pandemic in 2020

# In[ ]:


top250.info()


# In[ ]:


top50.info()


# In[ ]:


top250.describe()


# In[ ]:


top50.describe()


# ## Seaborn pairplot - see correlations

# In[ ]:


sns.pairplot(top250)


# # Top 10 Companies by CEO Pay

# In[ ]:


top10company=top250.sort_values(by='ceoPay',ascending=False).head(10)
fig,ax=plt.subplots(figsize=(20,6))
ax=sns.barplot(x='company',y='ceoPay',data=top10company,palette="mako")
ax.set_ylabel('CEO Pay, in Billions USD')
ax.set_title('Top 10 Companies By CEO Pay')


# In[ ]:


top10company.head(10)


# # Highest paid CEO
# 
# ### It is crazy that CEO of Palantir can get paid over 1 Billion in one year! Unfortunately Palantir stock has dropped more than 70% since the highs

# In[ ]:


sns.pairplot(top50)

