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


import seaborn as sns
import matplotlib.pyplot as plt
#import matplotlib as plt


# In[ ]:


with open('/kaggle/input/hashcode-drone-delivery/busy_day.in') as file:
    myfile=file.read().splitlines()


# In[ ]:


myfile


# In[ ]:


index=0


# In[ ]:


simulation_parameters=myfile[index].split(" ")


# In[ ]:


simulation_parameters


# In[ ]:


print("row of grid, column of grid , drones, turns, max load in units(u)", myfile[0], 
      "\n differnet type of products",myfile[1],
      "\n product types weight",myfile[2],
      "\n warhouses", myfile[3],
      "\n First warehouse location at first warehouse (row, column):",myfile[4],
      '\n Inventory of products:',myfile[5],
      "\n second warehouse location at first warehouse (row, column):",myfile[6],
      '\n Inventory of products:',myfile[7],
      '\n last warehouse location (row, column)  :',myfile[22],
      '\n Inventory of products at last ware house:',myfile[23],
      '\n Number of orders:',myfile[24],
      '\n First order to be delivery at:',myfile[25],
      '\n Number of items in order:',myfile[26],
      '\n Items of product types:',myfile[27]   )  


# In[ ]:


s=len(myfile[23])
s


# # warhouses locations 

# In[ ]:


warhouse_locs=myfile[4:24:2]


# In[ ]:


warhouse_row=[warhouse_row.split()[0] for warhouse_row in warhouse_locs]
warhouse_col=[warhouse_col.split()[1] for warhouse_col in warhouse_locs]
warehouse_df=pd.DataFrame({"warhouse_row":warhouse_row,"warhouse_col":warhouse_col}).astype(np.uint16)
warehouse_df



# In[ ]:


sns.relplot( y="warhouse_row", x="warhouse_col" ,data=warehouse_df)


# In[ ]:


product_cols=[f'warehouse_{i}' for i in range(10)]
products=[product.split() for product in myfile[5:24:2]]
product_df=pd.DataFrame(products).T
product_df.columns=product_cols
product_df


# In[ ]:





# In[ ]:


#for i in order_df:
 #   if i > 0:
  #      print(i)


# In[ ]:


sub_orders=[f'prod_{i}' for  i in range(19)]
order_df=pd.DataFrame([orders.split() for orders in myfile[27:3775:3] ]).fillna(0).astype(np.uint16)
                                                                                         
                                                                                         #("int")
order_df.columns=sub_orders
order_df["order_x"]=[orders.split()[0] for orders in myfile[25:3775:3]]
order_df["order_y"]=[orders.split()[1] for orders in myfile[25:3775:3]]
order_df["order_items"]= myfile[26:3775:3]

order_df=order_df.astype("int")
order_df


# 

# In[ ]:


order_df.dtypes


# In[ ]:


import seaborn as sns

fig=plt.figure(figsize=(10,15))
sns.scatterplot(x="order_x", y="order_y", data=order_df,s=30, alpha=0.5)

#plt.scatter(, x=, y=)


# In[ ]:


x= range(400)
y = range(400,600)
fig = plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(222)

ax1.scatter(warehouse_df['warhouse_row'], warehouse_df['warhouse_col'], s=60, c='b', marker="o", label='ware_house')
ax1.scatter(order_df["order_x"], order_df['order_y'], s=20, c='r', marker="o", label='orders')
plt.legend(loc='upper left');
plt.show()


# In[ ]:


order_df1=order_df.copy()


# In[ ]:


order_id=np.arange(1250)
our_orders=order_df1.iloc[:,0:19]

our_orders["order_id"]=order_id
our_orders


# 

# **

# In[ ]:


our_orders =our_orders.set_index("order_id").stack()


# In[ ]:


our_orders


# In[ ]:


our_orders=pd.DataFrame(our_orders)
our_orders.columns=["product_id"]


# In[ ]:


our_orders.drop(our_orders.loc[our_orders['product_id']==0].index, inplace=True)



# In[ ]:


our_orders


# In[ ]:


product_id=np.arange(400)
product_df["product_id"]=product_id


# In[ ]:


product_df1 =product_df.set_index("product_id").stack()
product_df1=pd.DataFrame(product_df1)
product_df1.columns=["ware_house"]
product_df1=product_df1.astype("int")

product_df1.drop(product_df1.loc[product_df1["ware_house"]==0].index, inplace=True)





# In[ ]:


product_df1


# In[ ]:


our_orders

