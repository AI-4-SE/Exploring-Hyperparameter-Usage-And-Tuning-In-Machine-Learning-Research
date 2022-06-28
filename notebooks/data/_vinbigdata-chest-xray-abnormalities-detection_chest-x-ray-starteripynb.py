#!/usr/bin/env python
# coding: utf-8

# # Intro
# Welcome to the [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data) compedition.
# 
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/24800/logos/header.png)
# 
# We consider 14 critical radiographic findings as listed below (click for further informations):
# 
# 0 - [Aortic enlargement](https://en.wikipedia.org/wiki/Aortic_aneurysm) <br>
# 1 - [Atelectasis](https://en.wikipedia.org/wiki/Atelectasis) <br>
# 2 - [Calcification](https://en.wikipedia.org/wiki/Calcification) <br>
# 3 - [Cardiomegaly](https://en.wikipedia.org/wiki/Cardiomegaly) <br>
# 4 - [Consolidation](https://en.wikipedia.org/wiki/Pulmonary_consolidation) <br>
# 5 - [ILD](https://en.wikipedia.org/wiki/Interstitial_lung_disease) <br>
# 6 - [Infiltration](https://en.wikipedia.org/wiki/Infiltration_(medical)) <br>
# 7 - [Lung Opacity](https://en.wikipedia.org/wiki/Ground-glass_opacity) <br>
# 8 - [Nodule/Mass](https://en.wikipedia.org/wiki/Lung_nodule) <br>
# 9 - Other lesion <br>
# 10 - [Pleural effusion](https://en.wikipedia.org/wiki/Pleural_effusion) <br>
# 11 - [Pleural thickening](https://en.wikipedia.org/wiki/Pleural_thickening) <br>
# 12 - [Pneumothorax](https://en.wikipedia.org/wiki/Pneumothorax) <br>
# 13 - [Pulmonary fibrosis](https://en.wikipedia.org/wiki/Pulmonary_fibrosis#:~:text=Pulmonary%20fibrosis%20is%20a%20condition,%2C%20pneumothorax%2C%20and%20lung%20cancer.)
# 
# <span style="color: royalblue;">Please vote the notebook up if it helps you. Feel free to leave a comment above the notebook. Thank you. </span>

# # Libraries

# In[ ]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pydicom as dicom
import cv2

import warnings
warnings.filterwarnings("ignore")


# # Path

# In[ ]:


path = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/'
os.listdir(path)


# # Load Data

# In[ ]:


train_data = pd.read_csv(path+'train.csv')
samp_subm = pd.read_csv(path+'sample_submission.csv')


# # Overview

# In[ ]:


print('Number train samples:', len(train_data.index))
print('Number test samples:', len(samp_subm.index))


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(12, 4))
x = train_data['class_name'].value_counts().keys()
y = train_data['class_name'].value_counts().values
ax.bar(x, y)
ax.set_xticklabels(x, rotation=90)
ax.set_title('Distribution of the labels')
plt.grid()
plt.show()


# As we can see the dataset is inbalanced.

# # Read dicom Files

# In[ ]:


idnum = 2
image_id = train_data.loc[idnum, 'image_id']
data_file = dicom.dcmread(path+'train/'+image_id+'.dicom')
img = data_file.pixel_array


# Print meta data of the image:

# In[ ]:


print(data_file)


# In[ ]:


print('Image shape:', img.shape)


# In[ ]:


bbox = [train_data.loc[idnum, 'x_min'],
        train_data.loc[idnum, 'y_min'],
        train_data.loc[idnum, 'x_max'],
        train_data.loc[idnum, 'y_max']]
fig, ax = plt.subplots(1, 1, figsize=(20, 4))
ax.imshow(img, cmap='gray')
p = matplotlib.patches.Rectangle((bbox[0], bbox[1]),
                                 bbox[2]-bbox[0],
                                 bbox[3]-bbox[1],
                                 ec='r', fc='none', lw=2.)
ax.add_patch(p)
plt.show()


# # Show Examples
# Plot 3 images of every class with the bounding boxes:

# In[ ]:


def plot_example(idx_list):
    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    fig.subplots_adjust(hspace = .1, wspace=.1)
    axs = axs.ravel()
    for i in range(3):
        image_id = train_data.loc[idx_list[i], 'image_id']
        data_file = dicom.dcmread(path+'train/'+image_id+'.dicom')
        img = data_file.pixel_array
        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(train_data.loc[idx_list[i], 'class_name'])
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
        if train_data.loc[idx_list[i], 'class_name'] != 'No finding':
            bbox = [train_data.loc[idx_list[i], 'x_min'],
                    train_data.loc[idx_list[i], 'y_min'],
                    train_data.loc[idx_list[i], 'x_max'],
                    train_data.loc[idx_list[i], 'y_max']]
            p = matplotlib.patches.Rectangle((bbox[0], bbox[1]),
                                             bbox[2]-bbox[0],
                                             bbox[3]-bbox[1],
                                             ec='r', fc='none', lw=2.)
            axs[i].add_patch(p)
            
for num in range(15):
    idx_list = train_data[train_data['class_id']==num][0:3].index.values
    plot_example(idx_list)


# # Write Output
# 

# In[ ]:


samp_subm.to_csv('submission.csv', index=False)

