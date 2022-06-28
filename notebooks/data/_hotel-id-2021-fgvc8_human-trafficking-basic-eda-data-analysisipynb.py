#!/usr/bin/env python
# coding: utf-8

# <h2><center>Hotel-ID to Combat Human Trafficking 2021 - FGVC8</center></h2>
# <h3><center>Recognizing hotels to aid Human trafficking investigations</center></h3>
# <center><img src = "https://polarisproject.org/wp-content/uploads/2019/01/800x640-marriott-blog.jpg" width = "800" height = "640"/></center>  

# ## Contents
# <ul>
#     <li><h3>Motive</h3></li>
#     <li><h3>Import Libraries</h3></li>
#     <li><h3>Chains Vs Num Hotels</h3></li>
#     <li><h3>Chains Vs Num Images</h3></li>
#     <li><h3>Hotels Vs Num Images</h3></li>
#     <li><h3>Conclusion</h3></li>
#     <li><h3>Random Images</h3></li>
# </ul>

# <h2><center>Motive</center></h2>

# Human trafficking, a form of modern dat slavery, is a global problem affecting people of all ages. It is estimated that approximately 1,000,000 people are trafficked each year globally and that between 20,000 and 50,000 are trafficked into the United States, which is one of the largest destinations for victims of the sex-trafficking trade.
# 
# Victims of human trafficking are kept in hotel rooms and sometimes photographed which can be used in later part of this crime. Identifying the hotels  from these images will play a vital role in lessening the astrounding noted case of human trafficking. Besides, it will also help the law enforcers to catch the criminals.
# 
# In this competition, we are tasked to identify hotels from 88 different chains by their images which can be later used for the abovementioned cause.

# # Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
import sys
from tqdm.autonotebook import tqdm
import seaborn as sns
import glob


# In[ ]:


df = pd.read_csv("../input/hotel-id-2021-fgvc8/train.csv")
df.head()


# In[ ]:


df.info()


# <center><h2> Chains VS Num Hotels </h2></center>

# There are 88 different chains. Each chains have some hotels under them. In this section, we will look at the number of hotels under different chains.

# <h3>Let's look at the chains having minimum and maximum different hotels.</h3>

# In[ ]:


chain_ids = []
chain_values = []
for chain_id in df.chain.unique():
    chain_ids.append(str(chain_id))
    chain_values.append(len(df[df.chain == chain_id].hotel_id.unique()))
    
chain_ids = [x for _, x in sorted(zip(chain_values, chain_ids))]
chain_values = sorted(chain_values)


figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,8), squeeze=False)


names = chain_ids[:20]
values = chain_values[:20]

sns.barplot(x=names, y=values, ax = axes[0][0])
plt.xticks(rotation=45)


names = chain_ids[-20:]
values = chain_values[-20:]

sns.barplot(x=names, y=values, ax = axes[0][1])
plt.xticks(rotation=45)

axes[0, 0].set_title("Min 20")
axes[0, 1].set_title("Max 20")
axes[0,0].tick_params(labelrotation=45)
axes[0,1].tick_params(labelrotation=45)
plt.setp(axes[-1, :], xlabel='Chain Id')
plt.setp(axes[:, 0], ylabel='Hotel Count')
plt.tight_layout()    
plt.show()


# <h3>Let's look at the distribution.</h3>

# In[ ]:


plt.figure(figsize=(15, 8))
_ = plt.hist(chain_values, bins=30)


# <center><h2> Chains VS Num Images </h2></center>

# <h3>Let's look at the chains having minimum and maximum images.</h3>

# In[ ]:


chain_ids = []
chain_values = []
for chain_id in df.chain.unique():
    chain_ids.append(str(chain_id))
    chain_values.append(len(df[df.chain == chain_id]))
    
chain_ids = [x for _, x in sorted(zip(chain_values, chain_ids))]
chain_values = sorted(chain_values)



figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8), squeeze=False)


names = chain_ids[:20]
values = chain_values[:20]

#plt.figure(figsize=(20, 10))
sns.barplot(x=names, y=values, ax = axes[0][0])
plt.xticks(rotation=45)




names = chain_ids[-20:]
values = chain_values[-20:]

#plt.figure(figsize=(20, 10))
sns.barplot(x=names, y=values, ax = axes[0][1])
plt.xticks(rotation=45)


axes[0, 0].set_title("Min 20")
axes[0, 1].set_title("Max 20")
axes[0,0].tick_params(labelrotation=45)
axes[0,1].tick_params(labelrotation=45)
plt.setp(axes[-1, :], xlabel='Chain Id')
plt.setp(axes[:, 0], ylabel='Image Count')
plt.tight_layout()    
plt.show()


# <h2>The distribution is</h2>

# In[ ]:


plt.figure(figsize=(15, 8))
_ = plt.hist(chain_values, bins=30)


# <center><h2> Hotels VS Num Images </h2></center>

# <h2> Minimum and Maximum Images</h2>

# In[ ]:


hotel_ids = []
image_values = []
for hotel_id in df.hotel_id.unique():
    hotel_ids.append(str(hotel_id))
    image_values.append(len(df[df.hotel_id == hotel_id]))
    
hotel_ids = [x for _, x in sorted(zip(image_values, hotel_ids))]
image_values = sorted(image_values)



figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8), squeeze=False)

names = hotel_ids[:20]
values = image_values[:20]

sns.barplot(x=names, y=values, ax = axes[0][0])


names = hotel_ids[-20:]
values = image_values[-20:]

sns.barplot(x=names, y=values, ax = axes[0][1])


axes[0, 0].set_title("Min 20")
axes[0, 1].set_title("Max 20")
axes[0,0].tick_params(labelrotation=45)
axes[0,1].tick_params(labelrotation=45)
plt.setp(axes[-1, :], xlabel='Hotel Id')
plt.setp(axes[:, 0], ylabel='Image Count')
plt.tight_layout()    
plt.show()


# ## Distribution

# In[ ]:


plt.figure(figsize=(15, 8))
_ = plt.hist(image_values, bins=30)


# <center><h2>Conclusion</h2></center>

# From the above EDA we can come to these conclusions.
# 
# <ul>
#     <li>There are 88 different chains. Each chain manages various number of hotels. The number of managed hotels starts from 1 and reaches a maximum of 1750. But the majority of the chains have less than 100 hotels.</li> 
#     <li>Number of samples taken from a chain can range from 10 to 20,000 at max. 60% of the chains have less than 500 samples.</li>
#     <li>There are 7770 different hotels listed in the dataset. In worst case scenario, there is one hotel having only one sample. Whereas, the maximun number of images per hotel can be around 90. Majority of the hotels have around 20 samples</li>
#     
# </ul>

# ## Some Random Images

# In[ ]:


files = glob.glob("../input/hotel-id-2021-fgvc8/train_images/*/*")


# In[ ]:


figure, axes = plt.subplots(nrows=5, ncols=3, figsize=(20,15))
for i in range(15):
    path = np.random.choice(files)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[i//3, i%3].imshow(image)

