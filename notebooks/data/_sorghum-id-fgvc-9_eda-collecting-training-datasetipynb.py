#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


# In[ ]:


src = '/kaggle/input/AI4Code/'


# # Overview

# In[ ]:


get_ipython().system('ls /kaggle/input/AI4Code/')


# * **train/** - A folder comprising about 140,000 JSON files with the filenames corresponding to the id field in the csv files. Each file contains the code and markdown cells of a Kaggle notebook. The code cells are in their original (correct) order. The markdown cells have been shuffled and placed after the code cells.

# In[ ]:


train_jsons = os.listdir(src + '/train')[:10]
train_jsons


# * **train_orders.csv** - Gives the correct order of the cells for each notebook in the train/ folder.
# * **id** - The notebook in file {id}.json.
# * **cell_order** - A space delimited list of the correct cell ordering given in terms of the order in {id}.json.

# In[ ]:


train_orders_df = pd.read_csv(src + 'train_orders.csv')
print('train_orders:')
display(train_orders_df.head())


# ## Most of notebooks have 21 cells (~3100 of 140000)

# In[ ]:


train_orders_df['length'] = train_orders_df['cell_order'].apply(lambda x: len(x.split(' ')))

import plotly.express as px
fig = px.histogram(train_orders_df, x="length",title='Number of cells in notebook')
fig.show()


# * **train_ancestors.csv** - On Kaggle, a user may "fork" (that is, copy) the notebook of another user to create their own version. This file contains the forking history of notebooks in the training set. Note: There is no corresponding file for the test set.
# * **ancestor_id** - Identifies sets of notebooks that have a common origin or "ancestor". As no notebook in the test set has an ancestor in the training set, you may find this field to be of use as a grouping factor when constructing validation splits.
# * **parent_id** - Indicates that some version of the notebook id was forked from some version of the notebook parent_id. The notebook parent_id may or may not be present in the training data. (The parent may be missing because someone had forked a private notebook of their own, for instance.)

# In[ ]:


train_ancestors_df = pd.read_csv(src + 'train_ancestors.csv')
print('train_ancestors:')
display(train_ancestors_df.head())


# * **sample_submission** looks like this

# In[ ]:


sample_submission_df = pd.read_csv(src + 'sample_submission.csv')
print('sample_submission:')
display(sample_submission_df.head())


# -------------------------------------------------
# -------------------------------------------------

# # Functions to work with notebooks and some findings

# In[ ]:


def read_notebook_from_train_orders(i):
    """
    This function reads a notebook from i-th line of train_orders.csv with a correct cell order
    """
    id_, cell_order, length = train_orders_df.iloc[i]
    path = src + 'train/' + id_ +'.json'
    cell_order = cell_order.split( )
    #print(cell_order)
    notebook_df = pd.read_json(
                            path,
                            dtype={'cell_type': 'category', 'source': 'str'}
                            ).rename_axis('cell_id')
    return notebook_df.loc[cell_order] # put cells in a correct cell order 


# ## Example:

# In[ ]:


read_notebook_from_train_orders(1)


# ## Most notebooks have roughly the same number of code and markdown cells, but there are a lot of notebooks that have a lot of code with small number of markdowns

# In[ ]:


ratios = []
N = len(train_orders_df)

for i in tqdm(range(N // 40)):
    table = read_notebook_from_train_orders(i)
    id_, cell_order, length = train_orders_df.iloc[i]
    path = src + 'train/' + id_ +'.json'
    cell_order = cell_order.split()
    #print(cell_order)
    table = pd.read_json(
                        path,
                        dtype={'cell_type': 'category'}
                        ).rename_axis('cell_id')
    n_code, n_markdown = table.cell_type.value_counts()
    ratios.append(n_code / n_markdown)

px.histogram(ratios,title='histogram of ratio of number of code cells to markdown cells')


# In[ ]:


df_with_parent_id = train_ancestors_df[train_ancestors_df.parent_id.notna()]
display(df_with_parent_id.head())
parent_ids = df_with_parent_id.parent_id.tolist()
print(f'There are {len(parent_ids)} notebooks with parent_id\nbut none of them are represented in train_orders.csv')
train_orders_df[train_orders_df.id.isin(parent_ids)]


# * "... you may find **ancestor** field to be of use as a **grouping factor** when constructing validation splits"

# In[ ]:


train_ancestors_df.ancestor_id.value_counts()


# * There are 65 notebooks with **4569bfc1** as an ancestor

# In[ ]:


train_ancestors_df[train_ancestors_df.ancestor_id == '4569bfc1']


# * Here are several examples of these notebooks that starts from the same markdown:

# In[ ]:


read_notebook_from_train_orders(10760)


# In[ ]:


read_notebook_from_train_orders(2613)


# In[ ]:


read_notebook_from_train_orders(4918)


# In[ ]:


print('2613:', read_notebook_from_train_orders(2613).source[0])
print('4918:', read_notebook_from_train_orders(4918).source[0])
print('10760:', read_notebook_from_train_orders(10760).source[0])


# ### By the way, their last markdowns are also the same:

# In[ ]:


print('2613:', read_notebook_from_train_orders(2613).source[-1])
print('4918:', read_notebook_from_train_orders(4918).source[-1])
print('10760:', read_notebook_from_train_orders(10760).source[-1])


# --------------------------------------------------

# # Collecting training dataset for your NLP / Ai model

# In[ ]:


N = 1000 # number of notebooks to add
data = []

for i in range(N):
    data.append(read_notebook_from_train_orders(N)[['cell_type','source']])


# In[ ]:


data[666]


# In[ ]:




