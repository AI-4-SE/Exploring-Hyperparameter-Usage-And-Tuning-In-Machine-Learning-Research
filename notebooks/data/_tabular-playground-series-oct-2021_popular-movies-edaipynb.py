#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
import plotly.express as px
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/popularity-based-movie-dataset/movies2.csv')
df = df.sort_values(by=['vote_average'], ascending=False)
df


# There seems to be some replicate rows.

# In[ ]:


df = df.drop_duplicates(subset=['title', 'release_date', 'overview', 'popularity', 'vote_average', 'vote_count'], keep='last')


# # Top 25 of popular film

# In[ ]:


df = df.sort_values(by=['popularity'], ascending=False)
chart = sns.barplot(x="title", y="popularity", data=df.head(25))
chart.set_xticklabels(chart.get_xticklabels(), rotation=90);


# # Top 25 average vote

# In[ ]:


df = df.sort_values(by=['vote_average'], ascending=False)
chart = sns.barplot(x="title", y="vote_average", data=df.head(25))
chart.set_xticklabels(chart.get_xticklabels(), rotation=90);


# # Let's see all the movies sorted by the average vote

# In[ ]:


#df = df.sort_values(by=['popularity'], ascending=False)
fig = px.bar(df, x='title', y='vote_average', color="popularity")
fig.show()


# # Good visualization of the movies and their popularity

# In[ ]:


fig = px.scatter(df, x="title", y="vote_average", size="popularity", color="vote_average", 
                 hover_name="title",
                 #log_x=True,
                 size_max=60)
fig.show()


# In[ ]:


#df = df.sort_values(by=['title'], ascending=False)
fig = px.scatter(df, x="title", y="popularity", size="popularity", color="vote_average", 
                 hover_name="title",
                 #log_x=True,
                 #size_max=60
                )
fig.show()


# # Additional charts

# In[ ]:


g = sns.jointplot(
    data=df,
    x="popularity", y="vote_average",
)


# In[ ]:


g = sns.jointplot(
    data=df,
    x="popularity", y="vote_average",
    kind="kde",
)

