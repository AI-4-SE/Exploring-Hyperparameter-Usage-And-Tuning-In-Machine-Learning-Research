#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os

BASE_PATH = "../input/landmark-recognition-2021"

train_df = pd.read_csv(BASE_PATH + "/train.csv")
submission_df = pd.read_csv(BASE_PATH + "/sample_submission.csv")


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
import seaborn as sns


# In[ ]:


print(train_df.shape)
print(submission_df.shape)


# In[ ]:


print(train_df.head())


# In[ ]:


print(submission_df.head())


# ## Plot some samples and see the resolutions

# In[ ]:


def image_grid3x3(image_array, landmarks):
    fig = plt.figure(figsize=(12., 12.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(3, 3),
                     axes_pad=1)
    
    for idx, (ax, im) in enumerate(zip(grid, image_array)):
        ax.imshow(im)
        ax.set_title(landmarks[idx])
        ax.set_xlabel(f'{im.shape}')
        
    plt.show()


# In[ ]:


def make_img_path(img_id):
    return "/".join([char for char in img_id[:3]]) + "/" + img_id + ".jpg"


# In[ ]:


def get_img_numpy(img_id, base="/train"):
    img_path = make_img_path(img_id)
    img = Image.open(base + "/" + img_path)
    return np.asarray(img)


# In[ ]:


img_array = [get_img_numpy(img, BASE_PATH + "/train") for img in train_df['id'][1000:1009]]

image_grid3x3(img_array, [landmark for landmark in train_df['landmark_id'][1000:1009]])


# In[ ]:


img_array = [get_img_numpy(img, BASE_PATH + "/train") for img in train_df['id'][10000:10009]]

image_grid3x3(img_array, [landmark for landmark in train_df['landmark_id'][10000:10009]])


# ## Get some graphs going about class distributions

# In[ ]:


sns.histplot(data=train_df, x="landmark_id", bins=1000)


# ### Conclusions:
# 
# - We need to resize images to some reasonable threshold.
# - We also need augmentation techniques to solve class imbalance.
#   - HIGH_CLASS_SAMPLES=6272
#   - MIN_CLASS_SAMPLES=2
