#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import geopandas as gpd 
import shapely 
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import textwrap
get_ipython().system('pip install country_converter')
import country_converter as coco


# # **Getting the data by Pandas**

# In[ ]:


df1=pd.read_csv('../input/chinese-debt-trap/chinese debt trap all over the world. - projects.csv')
df2=pd.read_csv('../input/chinese-debt-trap/chinese debt trap in Africa (sector wise).csv')


# # **Checking the datas are imported/not**

# In[ ]:


df1.head()


# In[ ]:


df1.rename(columns={'Expand All | Collapse All':'Project'},inplace=True)


# In[ ]:


df2.head()


# # Getting some insight about data types/null values in both the dataframes

# In[ ]:


df1.info()


# **There is a null value in borrower. Lets check that**

# In[ ]:


df1[df1['BORROWER'].isnull()]


# # **As we can see the project name is even not specified we can drop it**

# In[ ]:


df1.dropna(inplace=True)


# In[ ]:


df2.info()


# # Lets check the sectrors

# In[ ]:


df1['SECTOR'].unique()   #there is fTransport in place of transport. let's change it


# In[ ]:


df1['SECTOR']=df1['SECTOR'].apply(lambda x:'Transport' if x=='fTransport' else x)


# # **Let's convert the budgets into numericals**

# In[ ]:


def amount(a):
    y=a
    if ',' in a:
        a=a.replace(',','')
    a=float(a[1:-1])
    if y[-1]=='M':
        return a*10e6
    elif y[-1]=='B':
        return a*10e9
    else:
        return 'Please check'


# In[ ]:


df1['AMOUNT']=df1['AMOUNT'].apply(amount)
df2['$ Allocation']=df2['$ Allocation'].apply(amount)


# #  **Let's change the year data into timestamp datatype**

# In[ ]:


df1['YEAR']=pd.to_datetime(df1['YEAR'].map(str)).dt.year
df2['Year']=pd.to_datetime(df2['Year'].map(str)).dt.year


# # Lets create a contry code column which may help us in creating plot in world map.

# In[ ]:


def coco_conv(x):
    try:
        x=coco.convert(x)
    except:
        x=None
    return x


# In[ ]:


df1['Country code']=df1['Country'].apply(coco_conv)


# In[ ]:


df2['Country code']=df2['Country'].apply(coco_conv)


# In[ ]:


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# In[ ]:


world.head()


# In[ ]:


world.columns=['pop_est', 'continent', 'name', 'Country code', 'gdp_md_est', 'geometry']


# In[ ]:


df1=pd.merge(df1,world,on='Country code',how='inner')
df2=pd.merge(df2,world,on='Country code',how='inner')


# In[ ]:


df1.head()    


# In[ ]:


df2.head()


# # Converting pandas to geopandas for easier plotting

# In[ ]:


df1_geo = gpd.GeoDataFrame(df1)
df2_geo=gpd.GeoDataFrame(df2)


# # Lets plot The countries on the world map

# **The contries which are still safe from thr debt trap are green and the others are in red with a amount of debt propotional to depth of red color**

# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2,gridspec_kw={'width_ratios': [2, 1]},figsize=(20,10))
plt.suptitle("China's investment World-wide",size=20,weight='bold')
plt.subplot(1,2,1)
plt.title('The debt trap of china on world map')
world.plot(color='green',ax=ax1)
df1_geo.plot(ax=ax1,legend=True,cmap='Reds_r',cax='AMOUNT') 
world[world['Country code'] == 'CHN'].plot(color='yellow',ax=ax1)
data=df1.groupby('Country').sum().reset_index().sort_values('AMOUNT',ascending=False)[['Country','AMOUNT']]
plt.subplot(1,2,2)
plt.title('Top 30 contries with maximum debt')
plt.xticks(rotation=90)
sns.barplot(data=data.head(30),x='Country',y='AMOUNT',ax=ax2,palette='Reds_r')
plt.tight_layout()


# <h1>I have attached this map from internet to show the strategic paths in world trade to compare with the above ploted map to understand the china's debt trap plan <h1>
# <img src="https://i0.wp.com/transportgeography.org/wp-content/uploads/Map-Trade-Routes-1400-1800.png?resize=900%2C450&amp;ssl=1" jsaction="load:XAeZkd;" jsname="HiaYvf" class="n3VNCb KAlRDb" alt="Major Global Trade Routes, 1400-1800 | The Geography of Transport Systems" data-noaft="1" style="width: 450px; height: 325px; margin: 0px;">

# **The contries in Asia which are still safe from thr debt trap are green and the others are in red with a amount of debt propotional to depth of red color**

# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2,gridspec_kw={'width_ratios': [3,2]},figsize=(20,7))
plt.suptitle('Investment of china in Asia & sector wise distribution',size=20,weight='bold')
plt.subplot(1,2,1)
plt.title('Investment of china in Asia')
world[world.continent == 'Asia'].plot(figsize=(15,15),color='green',ax=ax1)
df1_geo[df1_geo['continent']=='Asia'].plot(ax=ax1,legend=True,cmap='Reds_r',cax='AMOUNT') 
world[world['Country code'] == 'CHN'].plot(ax=ax1,figsize=(15,15),color='yellow')
world[world.continent == 'Asia'].apply(lambda x: ax1.annotate(text=x['name'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)

plt.subplot(1,2,2)
plt.title('Sector-wise investment of china')
df1_geo[df1_geo.continent=='Asia'].groupby('SECTOR',as_index=False).sum().sort_values('AMOUNT',ascending=False).plot.bar(x='SECTOR',y='AMOUNT',ax=ax2,color='red')


# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2,gridspec_kw={'width_ratios': [3,2]},figsize=(20,7))
plt.suptitle('Investment of china in Africa & sector wise distribution',size=20,weight='bold')
plt.subplot(1,2,1)
plt.title('Investment of china in Africa')
world[world.continent == 'Africa'].plot(figsize=(15,15),color='green',ax=ax1)
df2_geo.plot(ax=ax1,legend=True,cmap='Reds_r',cax='$ Allocation') 
world[world.continent == 'Africa'].apply(lambda x: ax1.annotate(text=x['name'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)

plt.subplot(1,2,2)
plt.title('Sector-wise investment of china')
df2_geo.groupby('Invested On',as_index=False).sum().sort_values('$ Allocation',ascending=False).plot.bar(x='Invested On',y='$ Allocation',ax=ax2,color='red')


# **The contries in africa which are still safe from thr debt trap are green and the others are in red with a amount of debt propotional to depth of red color**

# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2,gridspec_kw={'width_ratios': [2,1]},figsize=(20,10))
plt.suptitle("China's investmenton africa & country-wise distribution",size=20,weight='bold')
plt.subplot(1,2,1)
plt.title('The debt trap of china on African countries')
world[world['continent']=='Africa'].plot(color='green',ax=ax1)
df2_geo.plot(ax=ax1,legend=True,cmap='Reds_r',cax='AMOUNT') 
world[world.continent == 'Africa'].apply(lambda x: ax1.annotate(text=x['name'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)

data=df2_geo.groupby('Country').sum().reset_index().sort_values('$ Allocation',ascending=False)[['Country','$ Allocation']]
plt.subplot(1,2,2)
plt.title('Top 30 contries in Africa with maximum debt')
plt.xticks(rotation=90)
sns.barplot(data=data.head(30),x='Country',y='$ Allocation',ax=ax2,palette='Reds_r')
plt.tight_layout()
plt.show()


# In[ ]:


data=df2_geo[df2_geo.Country=='Angola'].groupby('Invested On',as_index=False).sum().sort_values('$ Allocation',ascending=False)
plt.figure(figsize=(10,5))
plt.xticks(rotation=90)
sns.barplot(data=data,y='$ Allocation',x='Invested On')


# # Year wise spending of china internationally

# In[ ]:


plt.figure(figsize=(10,7))
plt.xticks(rotation=90)
data=df1.groupby('YEAR').sum().reset_index().sort_values('YEAR').sort_values('YEAR')
display(data)
sns.barplot(data=data,x='YEAR',y='AMOUNT',palette='spring')


# # Let's check how china invested money in different continent & different sector

# In[ ]:


data=df1_geo.groupby(['continent','SECTOR'],as_index=False).sum()
plt.figure(figsize=(20,7))
ax=sns.barplot(data=data,x='continent',y='AMOUNT',hue='SECTOR')
ax.legend(loc='upper center')


# **We can see china is focousing on power in ASIA region for it's power requirement like electricity,transport in AFRICA for mineral rich region and trade paths and pipelines in europe including russia for gas supply.**

# # Lets check top 25 countries on which china invested the most(world wide)

# In[ ]:


plt.figure(figsize=(15,10))
plt.suptitle('top 25 countries on which china invested the most(world wide)',size=20,weight='bold')
data=df1.groupby('Country').sum().reset_index().sort_values('AMOUNT',ascending=False)[['Country','AMOUNT']]
plt.subplot(1,2,1)
plt.xticks(rotation=90)
sns.barplot(data=data.head(25),x='Country',y='AMOUNT')
plt.subplot(1,2,2)
plt.axis('off')
plt.tight_layout()
plt.table(cellText=data.head(25).values, colLabels=data.columns, loc='center')


# In[ ]:


plt.figure(figsize=(15,10))
plt.suptitle("top 25 countrie's Government on which china invested the most(world wide)",size=20,weight='bold')
data=df1[df1['BORROWER']=='Government'].groupby('Country').sum().reset_index().sort_values('AMOUNT',ascending=False)[['Country','AMOUNT']]
plt.subplot(1,2,1)
plt.xticks(rotation=90)
sns.barplot(data=data.head(25),x='Country',y='AMOUNT')
plt.subplot(1,2,2)
plt.axis('off')
plt.tight_layout()
plt.table(cellText=data.head(25).values, colLabels=data.columns, loc='center')


# # We can analyse indivisual contries as follows
# **You can put any country's name from the data**

# In[ ]:


country='Sri Lanka'
data=df1[df1['Country']==country]
ax=plt.figure(figsize=(15,10))
plt.suptitle(f"China's investments on {country}",weight='bold',size=20)
plt.subplot(2,2,1)
plt.title('Year-wise Investment')
sns.lineplot(data=data.groupby('YEAR').sum().reset_index(),x='YEAR',y='AMOUNT',color='r')
plt.subplot(2,2,2)
plt.title('Sector-wise Investment')
data['SECTOR'].value_counts().plot.pie(autopct='%.2f%%')
plt.subplot(2,2,3)
plt.title('Borrower-wise Amount invested')
plt.xticks(rotation=90)
ax=sns.barplot(data=data.groupby('BORROWER').sum().reset_index(),x='BORROWER',y='AMOUNT')
labels = [textwrap.fill(label.get_text(), 12) for label in ax.get_xticklabels()]
ax.set_xticklabels(labels)
plt.subplot(2,2,4)
plt.title('YEAR-wise Amount invested')
ax=sns.barplot(data=data,x='YEAR',y='AMOUNT',hue='SECTOR',ci=False)
labels = [textwrap.fill(label.get_text(), 12) for label in ax.get_xticklabels()]
display(data.sort_values('AMOUNT',ascending=False)[['Project','YEAR','AMOUNT']].reset_index(drop=True))


# **We can clearly see from the above graphs Sri Lanka government took heavy loans during 2018 to develope Transport & power which led to it's government collapse**

# In[ ]:




