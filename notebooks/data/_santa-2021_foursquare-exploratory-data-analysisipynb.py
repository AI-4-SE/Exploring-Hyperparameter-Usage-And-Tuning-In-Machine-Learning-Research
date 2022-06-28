#!/usr/bin/env python
# coding: utf-8

# # FourSquare - Exploratory Data Analysis
# The aim of this notebook is to understand data by looking at the high level stats of train dataset and pairs dataset. I will keep adding to this notebook as and when I think of something to explore. Hope this is useful for you. 
# 
# Contents:<br>
# 1. [Train Data](#1)
#     1. [Sample Data](#1.1)
#     1. [Completeness](#1.2)
#     1. [Uniqueness](#1.3)
#         1. [By percentage](#1.3.1)
#     1. [Records Distribution](#1.4)
#         1. [By Country](#1.4.1)
#         1. [By Category](#1.4.2)
#         1. [By Languages used in the 'name' column](#1.4.3)
# 1. [Pairs Data](#2)
#     1. [Sample Data](#2.1)
#     1. [Latitude and Longitude differences for the pairs split by match flag](#2.2)
#         1. [Without Outliers](#2.2.1)
#     1. [Column Matches](#2.3)
#     

# In[ ]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# <a id="1"></a> <br>
# ## 1. Train Data
# train.csv - The training set, comprising eleven attribute fields for over one million place entries, together with:
# * id - A unique identifier for each entry.
# * point_of_interest - An identifier for the POI the entry represents. There may be one or many entries describing the same POI. Two entries "match" when they describe a common POI.
# match - Whether (True or False) the pair of entries describes a common POI.

# In[ ]:


train = pd.read_csv("/kaggle/input/foursquare-location-matching/train.csv")


# In[ ]:


print("Train data set contains {i} records and {j} dimensions".format(i=train.shape[0], j=train.shape[1]))


# <a id="1.1"></a> <br>
# ### 1.1 Sample Data

# In[ ]:


train.head()


# <a id="1.2"></a> <br>
# ### 1.2 Completeness

# In[ ]:


na_df = (train.isnull().sum() / len(train)) * 100      
na_df = na_df.sort_values()
sns.set(rc={'figure.figsize':(7,7)})
ax = sns.barplot(x=na_df.values, y=na_df.index)
ax.set(xlabel='Missing Ratio %', ylabel='Column Name')
plt.show()


# In[ ]:


print("Fully populated columns:", list(na_df[na_df==0].index),"\n")
print(list(na_df[(na_df>0)&(na_df<1)].index), "are not populated for a few records(0.001% null)")


# <a id="1.3"></a> <br>
# ### 1.3 Uniqueness

# In[ ]:


train.nunique().reset_index().rename(columns={'index':'Column Name', 0:'Number of unique values'})


# <a id="1.3.1"></a> <br>
# #### By percentage

# In[ ]:


unique_df = (train.nunique() / len(train)) * 100      
unique_df = unique_df.sort_values(ascending=False)
sns.set(rc={'figure.figsize':(7,7)})
ax = sns.barplot(x=unique_df.values, y=unique_df.index)
ax.set(xlabel='Unique Values %', ylabel='Column Name')
plt.show()


# <a id="1.4"></a> <br>
# ### 1.4 Records Distribution

# <a id="1.4.1"></a> <br>
# #### By Country

# In[ ]:


country_df = (train['country'].value_counts() / len(train)) * 100      
country_df_30 = country_df.head(30)
country_df_30['OTHER'] = country_df.reset_index()[30:].sum().values[1]
sns.set(rc={'figure.figsize':(20,5)})
ax = sns.barplot(x=country_df_30.index, y=country_df_30.values)
ax.set(xlabel='Country Code', ylabel='Records %')
plt.show()


# In[ ]:


print("Number of unique countries in the dataset: ", train['country'].nunique())


# <a id="1.4.2"></a> <br>
# #### By category

# In[ ]:


categories_df = (train['categories'].value_counts() / len(train)) * 100      
categories_df_50 = categories_df.head(50)
categories_df_50['OTHER'] = categories_df_50.reset_index()[50:].sum().values[1]
sns.set(rc={'figure.figsize':(20,5)})
ax = sns.barplot(x=categories_df_50.index, y=categories_df_50.values)
ax.set(xlabel='Categories', ylabel='Records %')
plt.xticks(rotation = 90)
plt.show()


# * Remember that catergories are populated for less than 10% of the records

# In[ ]:


print("Number of unique categories in the dataset: ", train['categories'].nunique())


# <a id="1.4.3"></a> <br>
# #### By Languages used in the 'name' column
# Disclaimer: Can't comment on the accuracy accuracy of langid which I have used here to detect which languages are used in the names.

# In[ ]:


# !pip install langid
# import langid

#This takes a while to run
train['name_lang'] = train['name'].apply(lambda x: x if pd.isnull(x) else langid.classify(x)[0])


# In[ ]:


name_lang_df = (train['name_lang'].value_counts() / len(train)) * 100      
name_lang_df_30 = name_lang_df.head(30)
name_lang_df_30['OTHER'] = name_lang_df_30.reset_index()[30:].sum().values[1]
sns.set(rc={'figure.figsize':(20,5)})
ax = sns.barplot(x=name_lang_df_30.index, y=name_lang_df_30.values)
ax.set(xlabel='Language', ylabel='Records %')
plt.show()


# <a id="2"></a> <br>
# ## 2. Pairs Dataset
# * pairs.csv - A pregenerated set of pairs of place entries from train.csv designed to improve detection of matches. You may wish to generate additional pairs to improve your model's ability to discriminate POIs.
# 
# * Additional Explanation by competition organizers: All matching and non-matching pairs in pairs.csv come from the information in train.csv; there is no additional information in pairs.csv that does not exist in the train.csv file. Matches are pairs of places with the same point_of_interest ids, non-matches are samples of places with different point_of_interest ids. pairs.csv contains samples of matching and non-matching pairs with the purpose to help model training, and it is far from inclusive of all combinations that can be generated from train.csv. You may choose to use pairs.csv as is, modify it (remove matching and non-matching pairs or add new ones generated from train.csv), or disregard it completely, depending on your training strategy.

# In[ ]:


pairs = pd.read_csv("/kaggle/input/foursquare-location-matching/pairs.csv")


# In[ ]:


print("Pairs data set contains {i} records and {j} dimensions".format(i=pairs.shape[0], j=pairs.shape[1]))


# <a id="2.1"></a> <br>
# ### 2.1 Sample data

# In[ ]:


pairs.head() 


# In[ ]:


pairs['latitude_diff'] = abs(pairs['latitude_1'] - pairs['latitude_2'])
pairs['longitude_diff'] = abs(pairs['longitude_1'] - pairs['longitude_2'])


# <a id="2.2"></a> <br>
# ### 2.2 Latitude and Longitude differences for the pairs split by match flag

# In[ ]:


f, ax = plt.subplots(1, 2)
sns.set(rc={'figure.figsize':(7,5)})
sns.boxplot(data=pairs, y='latitude_diff', x='match', orient='v', ax=ax[0])
sns.boxplot(data=pairs, y='longitude_diff', x='match', orient='v', ax=ax[1])
plt.tight_layout()
plt.show()


# <a id="2.2.1"></a> <br>
# #### Without Outliers

# In[ ]:


f, ax = plt.subplots(1, 2)
sns.set(rc={'figure.figsize':(7,5)})
sns.boxplot(data=pairs, y='latitude_diff', x='match', orient='v', showfliers=False, ax=ax[0])
sns.boxplot(data=pairs, y='longitude_diff', x='match', orient='v', showfliers=False, ax=ax[1])
plt.tight_layout()
plt.show()


# <a id="2.3"></a> <br>
# ### 2.3 Column Matches 

# In[ ]:


cols_to_compare = {}
for col in [ 'name', 'latitude', 'longitude', 'address', 'city', 'state', 'zip', 'country', 'url', 'phone', 'categories']:
    cols_to_compare[col+'_1_'+col+'_2'+'_match'] = (col+'_1', col+'_2') 


# In[ ]:


for key in cols_to_compare.keys():
    pairs[key] = pairs[cols_to_compare[key][0]]==pairs[cols_to_compare[key][1]]


# In[ ]:


pairs.head(5)


# In[ ]:


pairs_col_compare = pairs.groupby('match').sum()[cols_to_compare.keys()]
pairs_col_compare = (pairs_col_compare/pairs.shape[0])*100


# In[ ]:


match_false = pairs_col_compare.loc[False].reset_index()
match_true = pairs_col_compare.loc[True].reset_index()


# In[ ]:


f, ax = plt.subplots(1, 2)
sns.set(rc={'figure.figsize':(15,7)})
sns.barplot(data=match_true, y=True, x='index', ax=ax[0])
sns.barplot(data=match_false, y=False, x='index', ax=ax[1])
ax[0].set(xlabel='Matched Column', ylabel='Matched pairs % when match=True')
plt.setp(ax[0].get_xticklabels(), rotation=90)
ax[1].set(xlabel='Matched Column', ylabel='Matched pairs % when match=False')
plt.setp(ax[1].get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()


# Eaxmple interpretation for the above graph: 
# * (left) For ~70% of the pairs records the country_1 and country_2 columns are the same when match==True
# * (right) For ~30% of the pairs records the country_1 and country_2 columns are the same when match==False
