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


import zipfile
file_list = [
    '/kaggle/input/instacart-market-basket-analysis/aisles.csv.zip',
    '/kaggle/input/instacart-market-basket-analysis/orders.csv.zip',
    '/kaggle/input/instacart-market-basket-analysis/sample_submission.csv.zip',
    '/kaggle/input/instacart-market-basket-analysis/order_products__train.csv.zip',
    '/kaggle/input/instacart-market-basket-analysis/products.csv.zip',  
    '/kaggle/input/instacart-market-basket-analysis/order_products__prior.csv.zip',    
    '/kaggle/input/instacart-market-basket-analysis/departments.csv.zip']

for file_name in file_list:
    with zipfile.ZipFile(file=file_name) as target_zip:
        target_zip.extractall()


# In[ ]:


aisles_df = pd.read_csv('./aisles.csv')
orders_df = pd.read_csv('./orders.csv')
order_products__train_df = pd.read_csv('./order_products__train.csv')
sample_submission_df = pd.read_csv('./sample_submission.csv')
departments_df = pd.read_csv('./departments.csv')
products_df = pd.read_csv('./products.csv')
order_products__prior_df = pd.read_csv('./order_products__prior.csv')


# In[ ]:


products_merged_df=pd.merge(products_df, aisles_df, on="aisle_id")


# In[ ]:


products_merged_df=pd.merge(products_merged_df, departments_df, on="department_id")


# In[ ]:


order_products__train_df=pd.merge(order_products__train_df, products_merged_df, on="product_id")
order_products__prior_df=pd.merge(order_products__prior_df, products_merged_df, on="product_id")


# In[ ]:


merged_data=pd.merge(orders_df, order_products__prior_df, on="order_id")


# In[ ]:


df=pd.DataFrame(columns=merged_data['aisle'].unique()) 
newobj = {}
temp_order_id=0
for row in merged_data.head(100000).itertuples(index=True, name='Pandas'):
    if(temp_order_id!=row.order_id):
        if(temp_order_id!=0):
            b = pd.DataFrame.from_dict([newobj]) 
            df=df.append(b, ignore_index = True)
        newobj={}
        temp_order_id=row.order_id
        newobj["Index"]=row.order_id
    newobj[row.aisle]=1


# In[ ]:


#nan값 0으로 치환
df.fillna(0)


# In[ ]:


#index값 수정
df.set_index('Index')


# In[ ]:


#상관관계 추정
df.corr(method='pearson')


# In[ ]:


#상관관계 히트맵 
import plotly.express as px

fig = px.imshow(df)
fig.show()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3)


# In[ ]:


#knn
from sklearn.model_selection import train_test_split
training_data, validation_data , training_labels, validation_labels = train_test_split(df_data, df_labels, test_size = 0.2, random_state = 100)


# In[ ]:


df.append(, ignore_index = True) 


# In[ ]:


grouped_df = merged_data.groupby("user_id")["reordered"].aggregate("sum").reset_index()
grouped_df.loc[grouped_df["reordered"] >= 1] = 1
grouped_df.reordered.value_counts() / grouped_df.shape[0]


# In[ ]:





# In[ ]:


product_ordered_count=merged_data.groupby(['product_name'])['order_id'].aggregate("count").reset_index().rename(columns={'order_id': 'ordered_count'})


# In[ ]:


product_reordered_count=merged_data.query('(reordered==1)').groupby(['product_name'])['order_id'].aggregate("count").reset_index().rename(columns={'order_id': 'reordered_count'})


# In[ ]:


product_count=pd.merge(product_ordered_count, product_reordered_count, on="product_name").query('(ordered_count>1000)')


# In[ ]:


product_count['ratio']=product_count['reordered_count']/product_count['ordered_count']


# In[ ]:


product_count.sort_values(by="ratio", ascending=False)


# In[ ]:


grouped_df = orders_df.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()


# In[ ]:


import plotly.express as px

fig = px.density_heatmap(grouped_df, x='order_hour_of_day', y="order_dow", z='order_number', nbinsx=24, nbinsy=7)
fig.show()


# In[ ]:




