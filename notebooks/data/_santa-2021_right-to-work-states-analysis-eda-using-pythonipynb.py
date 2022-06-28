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


# <img src="https://media.giphy.com/media/10LKovKon8DENq/giphy.gif">

# let’s get our environment ready with the libraries we’ll need and then import the data!

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Check out the data

# In[ ]:


df=pd.read_csv(r'../input/right-to-work-dataset/RightToWork.csv')


# In[ ]:


df.head()


# Let's get the information from our data

# In[ ]:


df.info()


# Let’s get the statistic information from our data.

# In[ ]:


df.describe()


# It’s time to figure out our missing value in our dataset.

# In[ ]:


import missingno as msno
msno.matrix(df)


# There is no any missing value in our dataset.
# 
# Now, It’s time to get the data summary. For this purpose we are going to visualise the result using Seaborn library.

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(df.describe()[1:].transpose(),
            annot=True,linecolor="w",
            linewidth=2,cmap=sns.color_palette("tab20"))
plt.title("Data summary")
plt.show()


# Let’s get the correlation for each feature in our dataset.

# In[ ]:


cor_mat= df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)


# Now, It is time to create pairplot in order to see the correlation for each feature.

# In[ ]:


plt.figure(figsize=(25,15))
sns.pairplot(df)


# It’s time to create bar plot in order to see the count of state for each region.

# In[ ]:


sns.catplot(x="Region", kind="count", palette="ch:.26", data=df, size = 9)


# Let’s visualise the count of state in each region based on the right to work criteria.

# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(x='Region',data=df,hue='RightToWork',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Now, It is the time to visualise the states which they don’t have right to work.

# In[ ]:


norighttoworkstate = df[df['RightToWork']=='No']
sns.catplot(x="StateAbbrev", palette="ch:.26", data=norighttoworkstate, size = 11, kind = 'count')


# Let’s create the pie chart to see the percentage of right to work which is allowed and not in our dataset.

# In[ ]:


explode = (0.1,0)  
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(df['RightToWork'].value_counts(), explode=explode,labels=['Yes','No'], autopct='%1.1f%%',
        shadow=True)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()


# It is the time to visualise the Distribution of Poverty Rate by Region using Seaborn library.

# In[ ]:


plt.figure(figsize=(15,8))
plt.title("Distribution of Poverty Rate by Region")
for i in df['Region'].unique():
    sns.distplot(df[(df['Region']==i)]['PovertyRate2022'], hist=False, kde=True, label=i)


# Now, Let’s create heatmap based on Poverty Rate and Median Household Income.

# In[ ]:


plt.figure(figsize=(12,8))
sns.set()
sns.kdeplot( df['PovertyRate2022'], df['MedianHouseholdIncome2022'],
                 cmap="plasma", shade=True, shade_lowest=False)


# Let’s create Scatter plot of “Poverty Rate vs Median Household Income” based on right to work criteria.

# In[ ]:


plt.figure(figsize=(15,8))
sns.scatterplot(x='PovertyRate2022',y='MedianHouseholdIncome2022',data=df,palette='viridis',hue='RightToWork')
plt.title('Scatter plot of Poverty Rate vs Median Household Income')


# Let’s create Scatter plot of “Union Member Density vs Median Household Income” based on right to work criteria.

# In[ ]:


plt.figure(figsize=(15,8))
sns.scatterplot(x='UnionMemberDensity2021',y='MedianHouseholdIncome2022',data=df,palette='coolwarm',hue='RightToWork')
plt.title('Scatter plot of Union Member Density vs Median Household Income')


# Let’s see the Top 5 States By Poverty Rate.

# In[ ]:


top5states=df.sort_values(by='MedianHouseholdIncome2022',ascending=False).head(5)
fig,ax=plt.subplots(figsize=(16,6))
ax=sns.barplot(x='StateName',y='MedianHouseholdIncome2022',data=top5states,palette="rocket")
ax.set_title('Top 5 States By Avg Income')


# For the last analysis, Let’s see the Top 10 States By Poverty Rate based on right to work.

# In[ ]:


top10=df.sort_values(by='PovertyRate2022',ascending=False).head(10)
fig,ax=plt.subplots(figsize=(16,6))
ax=sns.barplot(x='StateName',y='PovertyRate2022',data=top10,palette="rocket_r", hue = 'RightToWork')
ax.set_title('Top 10 States By Poverty Rate')


# I hope you liked this analysis. Thanks for your time to read this article.

# **Please Upvote this notebook if you liked it**

# ![ezgif-3-e175d6dd34.gif](attachment:03d532ce-8cb4-454f-9832-f907bd50499e.gif)
