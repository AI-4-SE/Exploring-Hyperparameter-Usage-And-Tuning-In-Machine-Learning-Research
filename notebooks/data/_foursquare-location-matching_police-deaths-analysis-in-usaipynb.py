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


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
#     <p style="padding: 10px; color:white; font-weight: bold; font-size : 22px;">👮 Introduction</p>
# </div>
# 
# For most of the last 35 years, the number of police officers who die on the job in the U.S. declined, but one grim statistic held steady: The most common cause of death was gun homicide. Those numbers grew significantly at 2016 on a Thursday night when five police officers were shot and killed at a demonstration in Dallas that was protesting killings by police officers in other states. Former president Obama called it “a vicious, calculated and despicable attack on law enforcement.”
# 
# <img style = 'align:middle' src="https://i.gifer.com/embedded/download/KYml.gif">

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
#     <p style="padding: 10px; color:white; font-weight: bold; font-size : 22px;">💾 Data Preparation</p>
# </div>

# 💡 Let’s get our environment ready with the required libraries which we will need, and then import the data!

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from wordcloud import WordCloud
warnings.filterwarnings('ignore')
plt.style.use('ggplot')


# Check out the Data

# In[ ]:


df = pd.read_csv('/kaggle/input/police-deaths-in-americae/clean_data.csv')


# In[ ]:


df.head()


# Get the information from data

# In[ ]:


df.info()


# The type of date column is object. we have to convert it to datetime

# In[ ]:


df['date'] = pd.to_datetime(df['date'])
df.info()


# Now let’s create 2 new columns called Month, and Day of Week based on date column

# In[ ]:


df['Month'] = df['date'].apply(lambda time: time.month)
df['Day of Week'] = df['date'].apply(lambda time: time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)


# It's time to create column called type for sorting the death type.

# In[ ]:


df['type'] = np.where(df['canine'] == False, 'Police','Canine')


# Now, Let's see the finalised dataframe!

# In[ ]:


df.head()


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
#     <p style="padding: 10px; color:white; font-weight: bold; font-size : 22px;">🔦 Missing Values</p>
# </div>

# In[ ]:


import missingno as msno
msno.matrix(df)


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
#     <p style="padding: 10px; color:white; font-weight: bold; font-size : 22px;">💬 Word Cloud</p>
# </div>

# Word cloud is a technique for visualising frequent words in a text where the size of the words represents their frequency.

# Let's see the frequency of police officer cause deaths.

# In[ ]:


plt.figure(figsize=(16,10))
data = df[df['type']=='Police']['cause_short'].value_counts().to_dict()
wc = WordCloud(width= 2000, height = 1000, random_state=1,background_color='#191919').generate_from_frequencies(data)
plt.imshow(wc)
plt.title('Police Death Causes')
plt.axis('off')
plt.show()


# Let's see the frequency of canine cause deaths.

# In[ ]:


plt.figure(figsize=(16,10))
data = df[df['type']=='Canine']['cause_short'].value_counts().to_dict()
wc = WordCloud(width= 2000, height = 1000, random_state=1,background_color='#191919', colormap='rainbow').generate_from_frequencies(data)
plt.imshow(wc)
plt.title('Canine Death Causes')
plt.axis('off')
plt.show()


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
#     <p style="padding: 10px; color:white; font-weight: bold; font-size : 22px;">📊 Exploratory Data Analysis</p>
# </div>

# Let's visualise the summary of death

# In[ ]:


fig,ax=plt.subplots(figsize=(15,8))
sns.countplot(x='type',data=df,palette='cividis')

plt.title("Summary of Death")
plt.ylabel("Total Cases")
plt.show()


# Now, It's time to see the number of death causes which has been occured for polices!

# In[ ]:


df[df['type']=='Police']['cause_short'].value_counts()


# It's time to see the number of death for each states which has been occurred for polices!

# In[ ]:


df[df['type']=='Police']['state'].value_counts()


# Now, It's time to see the number of death causes which has been occurred for canines!

# In[ ]:


df[df['type']=='Canine']['cause_short'].value_counts()


# It's time to see the number of death for each states which has been occurred for canines!

# In[ ]:


df[df['type']=='Canine']['state'].value_counts()


# Let's visualise the top 10 states which had the most police death number.

# In[ ]:


plt.figure(figsize=(16,10))
ax = df[df['type']=='Police']['state'].value_counts().iloc[:10].plot(kind="barh", color = 'blue')
ax.invert_yaxis()
ax.title.set_text('Top 10 State For Police Death')


# Let's visualise the top 10 police death causes.

# In[ ]:


plt.figure(figsize=(16,10))
ax = df[df['type']=='Police']['cause_short'].value_counts().iloc[:10].plot(kind="barh")
ax.invert_yaxis()
ax.title.set_text('Top 10 Police Death Causes')


# Let's visualise the top 10 states which had the most canine death number.

# In[ ]:


plt.figure(figsize=(16,10))
ax = df[df['type']=='Canine']['state'].value_counts().iloc[:10].plot(kind="barh", color = 'green')
ax.invert_yaxis()
ax.title.set_text('Top 10 State For Canine Death')


# Let's visualise the top 10 canine death causes.

# In[ ]:


plt.figure(figsize=(16,10))
ax = df[df['type']=='Canine']['cause_short'].value_counts().iloc[:10].plot(kind="barh", color = 'brown')
ax.invert_yaxis()
ax.title.set_text('Top 10 Causes For Canine Death')


# Let's visualise the top 10 police departments which had the most police deaths.

# In[ ]:


plt.figure(figsize=(16,10))
ax = df[df['type']=='Police']['dept_name'].value_counts().iloc[:10].plot(kind="barh", color = 'orange')
ax.invert_yaxis()
ax.title.set_text('Top 10 Department For Police Death')


# Let's visualise the top 10 police departments which had the most canine deaths.

# In[ ]:


plt.figure(figsize=(16,10))
ax = df[df['type']=='Canine']['dept_name'].value_counts().iloc[:10].plot(kind="barh", color = 'purple')
ax.invert_yaxis()
ax.title.set_text('Top 10 Department For Canine Death')


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
#     <p style="padding: 10px; color:white; font-weight: bold; font-size : 22px;">📅 Date & Time Parameters</p>
# </div>

# Let's visualise the number of police deaths in each year!

# In[ ]:


plt.figure(figsize=(16,8))
df[df['type']=='Police'].groupby('year').count()['person'].plot()
plt.tight_layout()
plt.title('Police Death by Year')
plt.ylabel("Total Cases")
plt.xlabel("Date")


# Let's visualise the number of canine deaths in each year!

# In[ ]:


plt.figure(figsize=(16,8))
df[df['type']=='Canine'].groupby('year').count()['person'].plot(color='green')
plt.tight_layout()
plt.title('Canine Death by Date')
plt.ylabel("Total Cases")
plt.xlabel("Date")


# In[ ]:


policebyMonth = df[df['type']=='Police'].groupby('Month').count()
policebyMonth


# In[ ]:


caninebyMonth = df[df['type']=='Canine'].groupby('Month').count()
caninebyMonth


# It's time to see the number of police deaths for each month!

# In[ ]:


plt.figure(figsize=(16,8))
policebyMonth['person'].plot(kind = 'bar')
plt.title('Police Death For Each Month')
plt.ylabel("Total Cases")
plt.xlabel("Month")


# let's the number of canine deaths for each month!

# In[ ]:


plt.figure(figsize=(16,8))
caninebyMonth['person'].plot(kind = 'bar', color = 'brown')
plt.title('Canine Death by Month')
plt.ylabel("Total Cases")
plt.xlabel("Month")


# Let's visualise the number of police deaths by gunfire for each year !

# In[ ]:


plt.figure(figsize=(16,8))
df[df['type']=='Police'][df['cause_short']=='Gunfire'].groupby('year').count()['person'].plot()
plt.title('Police Death by Gunfire')
plt.tight_layout()


# Let's visualise the number of canine deaths by gunfire for each year !

# In[ ]:


plt.figure(figsize=(16,8))
df[df['type']=='Canine'][df['cause_short']=='Gunfire'].groupby('year').count()['person'].plot(color = 'blue')
plt.title('Canine Death by Gunfire')
plt.tight_layout()


# Let's visualise the number of 9/11 related illness for each year !

# In[ ]:


Policedeath_911 =df[(df['cause_short']=='9/11 related illness') & \
(df['type']=='Police')].year.value_counts().sort_index()

fig = plt.figure(figsize=(18, 10))
Policedeath_911.plot(kind='bar', color = 'green')
plt.ylabel("Total Cases")
plt.title('9/11 Related Police Death by Year')
plt.show()


# Now let’s create heatmap for all the cause deaths based on the month and day of the week.

# In[ ]:


PoliceMonthDay = df[df['canine']==False].groupby(by=['Day of Week','Month']).count()['cause_short'].unstack()
PoliceMonthDay.head()


# In[ ]:


CanineMonthDay = df[df['canine']==True].groupby(by=['Day of Week','Month']).count()['cause_short'].unstack()
CanineMonthDay.head()


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
#     <p style="padding: 10px; color:white; font-weight: bold; font-size : 22px;">🗺️ HEATMAP</p>
# </div>

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(PoliceMonthDay,cmap='cividis')


# In[ ]:


plt.figure(figsize=(12,8))
sns.clustermap(PoliceMonthDay,cmap='cividis')


# As we can see in the above heatmap, most of the deaths has been occured on July, August, September and December. Also, It was mostly during the weekend! 

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(CanineMonthDay,cmap='inferno')


# In[ ]:


dft = df.copy()
dft['state'] = df['state'].str.strip()
dft.head()


# In[ ]:


us_states = np.asarray(['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',\
                     'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI',\
                     'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY',\
                     'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT',\
                     'WA', 'WI', 'WV', 'WY'])

police_map = dft[dft['state'].isin(us_states)]
police_map.head()


# Let's visualise the number of police officer deaths per state in USA

# In[ ]:


import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode

police_state = np.asarray(police_map[police_map['type']=='Police'].groupby('state').state.count())


data = [dict(
        type = 'choropleth',
        autocolorscale = False,
        showscale = True,
        locations = us_states,
        z = police_state,
        locationmode = 'USA-states',
        marker = dict(
            line = dict(
                color = 'rgb(255, 255, 255)',
                width = 2)
            ),
        )]

layout = dict(
        title = 'Police Deaths by State in United States',
        geo = dict(
            scope = 'usa',
            projection = dict(type = 'albers usa'),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
            countrycolor = 'rgb(255, 255, 255)')
        )

figure = dict(data = data, layout = layout)
iplot(figure)


# In[ ]:


us_states2 = np.asarray(['AR', 'AL', 'AZ', 'CA', 'AK', 'CO', 'FL', 'GA', 'WY', 'MS', 'MO',\
                     'PA', 'IA', 'ID', 'LA', 'MD', 'KS', 'MN', 'NY', 'NC', 'MI', 'ME', 'NJ',\
                     'KY', 'DC', 'OH', 'OK', 'WA', 'TX', 'SC', 'TN', 'IN', 'UT', 'VA', 'DE',\
                     'WA', 'MT', 'OR', 'HI', 'RI', 'NE', 'SD', 'NH', 'ND', 'NM', 'NV', 'VT',\
                     'CT', 'WI', 'WV', 'IL'])
canine_map = dft[dft['state'].isin(us_states)]
canine_map.head()


# Let's visualise the number of canine deaths per state in USA

# In[ ]:


import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
# police officer deaths per state
police_perstate = np.asarray(canine_map[canine_map['type']=='Canine'].groupby('state').state.count())


data = [dict(
        type = 'choropleth',
        autocolorscale = False,
        showscale = True,
        locations = us_states2,
        z = police_perstate,
        locationmode = 'USA-states',
        marker = dict(
            line = dict(
                color = 'rgb(255, 255, 255)',
                width = 2)
            ),
        )]

layout = dict(
        title = 'Canine Deaths by State in United States',
        geo = dict(
            scope = 'usa',
            projection = dict(type = 'albers usa'),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
            countrycolor = 'rgb(255, 255, 255)')
        )

figure = dict(data = data, layout = layout)
iplot(figure)


# ❤️ Thanks for your time for reading this article and I hope it was useful for you!

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
#     <p style="padding: 10px; color:white; font-weight: bold; font-size : 22px;">👍 Please Upvote if you liked this notebook.
# </div>
# 
# <img src="http://i.imgur.com/oifRqiR.gif">

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
#     <p style="padding: 10px; color:white; font-weight: bold; font-size : 22px;">📚 References</p>
# </div>
# 
# 1. [The Dallas Shooting Was Among The Deadliest For Police In U.S. History](https://fivethirtyeight.com/features/the-dallas-shooting-was-among-the-deadliest-for-police-in-u-s-history/)
# 2. [Officer Down Memorial Page](https://www.odmp.org/)
# 3. [FiveThirtyEight](https://data.world/fivethirtyeight/police-deaths)
