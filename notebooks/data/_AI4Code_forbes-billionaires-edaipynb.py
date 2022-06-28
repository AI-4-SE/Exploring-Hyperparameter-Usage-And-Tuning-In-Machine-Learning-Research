#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
import plotly.express as px
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
#%matplotlib inline
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

#mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
mpl.rcParams['font.size']=12                #10 
mpl.rcParams['savefig.dpi']=100             #72 
mpl.rcParams['figure.subplot.bottom']=.1 

stopwords = set(STOPWORDS)


# In[ ]:


df = pd.read_csv('/kaggle/input/forbes-billionaires-data-preprocessed/Forbes Billionaires.csv')


# In[ ]:


df.head(10)


# In[ ]:


df.tail(10)


# In[ ]:


def to_str(df_col):
    text = ''
    for line in df_col:
        text = text + ' ' + line
    return text
def to_arr(df_col):
    text = []
    for line in df_col:
        text.append(line)
    return text
#to_str(df['Country'])


# # Data vizualisation
# ## Where does the billionaires come from ? 
# We are going to use both seaborn and wordclouds to check that. Let's count them and ordering them by their homeland countries.

# In[ ]:


chart = sns.countplot(x="Country", data=df, order=df['Country'].value_counts().index)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90);


# We can check the same request using wordclouds.

# In[ ]:


wordcloud = WordCloud(
                      background_color='white',
                      stopwords=stopwords,
                      max_words=200,
                      max_font_size=40, 
                      random_state=42
                     ).generate(to_str(df['Country']));

print("\nMost representatives countries ");
fig = plt.figure(1);
plt.imshow(wordcloud);
plt.axis('off');
plt.show();


# In[ ]:


fig = px.treemap(
    names = to_arr(df['Name']),
    parents = to_arr(df['Country'])
)
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()


# ## Where does this money come from ?
# We can have a good idea by using wordclouds once again.

# In[ ]:


wordcloud = WordCloud(
                      background_color='white',
                      stopwords=stopwords,
                      max_words=200,
                      max_font_size=40, 
                      random_state=42
                     ).generate(to_str(df['Source']));

print("\nMost representatives countries ");
fig = plt.figure(1);
plt.imshow(wordcloud);
plt.axis('off');
plt.show();


# ## What kind of industry is efficient to earn a lot of money ?

# In[ ]:


wordcloud = WordCloud(
                      background_color='white',
                      stopwords=stopwords,
                      max_words=200,
                      max_font_size=40, 
                      random_state=42
                     ).generate(to_str(df['Industry']));

print("\nMost representatives countries ");
fig = plt.figure(1);
plt.imshow(wordcloud);
plt.axis('off');
plt.show();


# # Let's gather different datas
# 
# Using plotly we can observe the networth of billionaires sorted by the industry and their age.

# In[ ]:


fig = px.scatter(df, x="Age", y="Networth", size="Networth", color="Industry", 
                 hover_name="Name",
                 log_x=True, size_max=60)
fig.show()


# Another way to see it.

# In[ ]:


g = sns.jointplot(
    data=df,
    x="Age", y="Networth", hue="Industry",
    kind="kde",
    height=15
)


# A different way to see things.

# In[ ]:


g = sns.jointplot(
    data=df,
    x="Age", y="Networth",
    kind="kde",
)


# We can do the same thing but with their countries.

# In[ ]:


fig = px.scatter(df, x="Age", y="Networth", size="Networth", color="Country", 
                 hover_name="Name",
                 log_x=True, size_max=60)
fig.show()


# # Is the country with the most billionaires the one with the most wealth?

# In[ ]:


chart = sns.barplot(x="Country", y="Networth", data=df, order=df['Country'].value_counts().index, estimator=sum,errwidth=0);
chart.set_xticklabels(chart.get_xticklabels(), rotation=90);


# The answer seems to be yes but we can see that France has more money than countries which have more billionaires since it's sorted by the number of billionaires in the chart.

# In[ ]:




