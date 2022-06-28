#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# In[ ]:


df = pd.read_csv('../input/google-playstore-apps/Google-Playstore.csv')


# # Basic Exploration and Data Cleaning

# In[ ]:


df.head()


# In[ ]:


df.info()


# **Checking missing values in all categories**

# In[ ]:


missing_values = df.isnull().sum()
missing_values


# **Cheking total count of missing values**

# In[ ]:


total_missing_counts = missing_values.sum()
total_missing_counts


# **Checking how many sells there are**

# In[ ]:


total_cells = np.prod(df.shape)
total_cells


# **What is the percentage of missing values from the whole dataset**

# In[ ]:


percent_of_missing_values = (total_missing_counts/total_cells) * 100
percent_of_missing_values
#0.36%


# **we have only 0.36% missing values so it does not look bad**

# # Issues list for dataset
# * Drop unnesseary categories
# * Missing values in serveral categories

# # Data Cleaning

# In[ ]:


# Drop unnesseary categories
drop_list = ['App Id', 'Minimum Android',
             'Developer Id', 'Developer Website',
             'Developer Email', 'Privacy Policy',
            'Ad Supported', 'In App Purchases',
            'Editors Choice', 'Scraped Time']


df.drop(drop_list, axis='columns', inplace=True)


# In[ ]:


# Drop null values
df.dropna(inplace=True)


# In[ ]:


check_null_values = df.isnull().sum()
check_null_values


# # What is my goal?
# * Figure out the most popular categories, app
# * Find the app, what has heighst rating and install number
# * Drive conclusion

# # EDA

# In[ ]:


df['Category'].value_counts()


# Top 7 categories:
# 1. Education
# 2. Music & Audio
# 3. Tools
# 4. Business
# 5. Entertainment
# 6. Books & Reference
# 7. Lifestyle

# In[ ]:


top_7_list = ['Education', 'Music & Audio', "Tools", "Business", "Entertainment", "Books & Reference", "Lifestyle"]
top = df[df['Category'].isin(top_7_list)].reset_index(drop=True)


# In[ ]:


top.head()


# In[ ]:


df.head()


# In[ ]:


fig, ax = plt.subplots()

ax.hist(top['Rating'], bins=20)

ax.set(title="Density of Rating",
      xlabel="Rating",
      ylabel="count")

plt.show()


# It seems like there are much more apps with 0 rating

# In[ ]:


bigger_than_0 = top[top['Rating'] > 0]['Rating']


# In[ ]:


fig, ax = plt.subplots()

ax.hist(bigger_than_0, bins=15)

ax.set(title="A Histogram of app ratings",
      xlabel="Ratings",
      ylabel="Count")

plt.show()


# Histogram shows that majority of the apps are rated between 3.8 ~ 4.8.
# <br>
# You see also so many 5 stars rating.

# In[ ]:


with pd.option_context('float_format', '{:f}'.format):
    print(top["Rating Count"].describe())


# In[ ]:


fig, ax = plt.subplots()

d = top[top["Rating Count"] < 35]

ax.hist(d["Rating Count"], bins=20)

plt.show()


# This shows that only top 25% apps have over 35 rating count. it means that only small portion of apps dominate the app market while more than 75% of apps have only 5 rating count.

# Let's find the apps which have more than 10 milions rating counts

# In[ ]:


ten_mil = top[top["Rating Count"] >= 10000000]
ten_mil.shape


# In[ ]:


ten_mil[["App Name","Rating Count", "Rating"]].sort_values(by="Rating Count", ascending=False)


# So you see here the most popular apps 

# In[ ]:


ax = ten_mil['Category'].value_counts().plot.barh()

ax.set(title='Comparison of Rating Count of Most Popular Apps by Category',
       ylabel='Rating Count')

plt.show();


# The apps that have tools as a caregory, are most popular

# In[ ]:


fig, ax = plt.subplots()


top["Installs"].value_counts().plot.barh()

ax.set(title='Proportion of install categories',
       xlabel='Proportion', ylabel='')

plt.show();


# This shows that only few apps have more than 1 milion install counts

# In[ ]:


not_free = top[top["Price"] != 0]
with pd.option_context('float_format', '{:f}'.format):
    print(not_free["Price"].describe())


# There is a huge gap between max and min.
# <br>
# Over 50% of apps are just about 3 dollar but the max app is 399

# In[ ]:


fig, ax = plt.subplots()

ax.hist(not_free["Price"], bins=10)

ax.set(title="A Histogram of App Price",
      xlabel="Price",
      ylabel="Count")

plt.show()


# As you can see the most apps cost around 0 ~ 40 dollar

# # Conclusion

# The app market is very competitive. As always in the buseness wolrd, only small porpotion of apps dominate the entire market.
# <br>
# Only 1/4 of all apps have over 35 rating counts and most popular apps have over 4 star ratings, milions rating counts
# <br>
# Most popular category is tools, apps like google or spotify
# 
