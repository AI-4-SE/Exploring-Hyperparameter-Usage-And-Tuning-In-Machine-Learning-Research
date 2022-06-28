#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import collections
import json
import os
import uuid

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import tifffile as tiff 
import seaborn as sns


# In[ ]:


get_ipython().system('ls ../input/hubmap-kidney-segmentation/')


# In[ ]:


train = pd.read_csv("../input/hubmap-kidney-segmentation/train.csv")
train.info()


# In[ ]:


metadata = pd.read_csv("../input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv")
metadata.info()


# In[ ]:


samplesubmission = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
samplesubmission.info()


# In[ ]:


img_id_1 = "aaa6a05cc"
image_1 = tiff.imread('../input/hubmap-kidney-segmentation/train/' + img_id_1 + ".tiff")
print("This image's id:", img_id_1)
image_1.shape


# In[ ]:


plt.figure(figsize=(15, 15))
plt.imshow(image_1)


# In[ ]:


plt.figure(figsize=(8,8))
plt.imshow(image_1[5200:5600, 5600:6000, :])


# In[ ]:


img_id_4 = "e79de561c"
image_4 = tiff.imread('../input/hubmap-kidney-segmentation/train/' + img_id_4 + ".tiff")
print("This image's id:", img_id_4)
image_4.shape


# In[ ]:


image_4 = image_4[0][0].transpose(1, 2, 0)
image_4.shape


# In[ ]:


plt.figure(figsize=(10, 10))
plt.imshow(image_4)


# In[ ]:


mask_rle = train[train["id"]==img_id_1]["encoding"].iloc[-1]


# In[ ]:


s = mask_rle.split()


# In[ ]:


shape=((image_1.shape[1], image_1.shape[0]))


# In[ ]:


starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]


# In[ ]:


starts -= 1
ends = starts + lengths
img = np.zeros(shape[0]*shape[1], dtype=np.uint8)


# In[ ]:


for lo, hi in zip(starts, ends):
    img[lo:hi] = 1
    image_reshaped = img.reshape(shape).T


# In[ ]:


image_reshaped.shape


# In[ ]:


plt.figure(figsize=(10,10))
plt.imshow(image_reshaped, cmap='coolwarm', alpha=0.5)


# In[ ]:


plt.figure(figsize=(10,10))
plt.imshow(image_1)
plt.imshow(image_reshaped, cmap='coolwarm', alpha=0.5)


# In[ ]:


with open("../input/hubmap-kidney-segmentation/train/e79de561c.json") as f:
    e79de561c_json = json.load(f)
    
print("lenght of json:", len(e79de561c_json))
print(e79de561c_json[0])


# In[ ]:


def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def draw_structure(structures, im):
    """
    anatomical_structure: list of points of anatomical_structure poligon.
    im: numpy array of image read from tiff file.
    """
    
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    for structure in structures:
        structure_flatten = list(flatten(structure["geometry"]["coordinates"][0]))
        structure = []
        for i in range(0, len(structure_flatten), 2):
            structure.append(tuple(structure_flatten[i:i+2]))
        
        draw.line(structure, width=100, fill='Red')
    return im


# In[ ]:


plt.figure(figsize=(8,8))
image_4_with_line = draw_structure(e79de561c_json, image_4)
plt.imshow(image_4_with_line)


# In[ ]:


with open(f"../input/hubmap-kidney-segmentation/train/{img_id_1}-anatomical-structure.json") as f:
    anatomical_structure_json = json.load(f)
    
anatomical_structure_json


# In[ ]:


plt.figure(figsize=(8,8))
image_1_with_line = draw_structure(anatomical_structure_json, image_1)
plt.imshow(image_1_with_line)


# In[ ]:


metadata.head()


# In[ ]:


metadata.shape


# In[ ]:


ds_info = metadata


# In[ ]:


def train_or_test(image_file):
    id, _ = image_file.split(".")
    if id in list(train["id"]):
        return "train"
    else:
        return "test"
    
ds_info["category"] = ds_info["image_file"].map(train_or_test)


# In[ ]:


plt.style.use("Solarize_Light2")


# In[ ]:


plt.figure(figsize=(15, 5))
g = sns.countplot(data=ds_info, x="patient_number", hue="category", palette=sns.color_palette("Set2", 8))
g.set_title("Number of images per patient")


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(10,5), gridspec_kw=dict(wspace=0.1, hspace=0.6))
fig.suptitle("race and ethnicity", fontsize=15)
g = sns.countplot(data=ds_info, x="race", hue="category", palette=sns.color_palette("Set2", 8),ax=axes[0])
g.set_title("distribution of race", fontsize=12)
g = sns.countplot(data=ds_info, x="ethnicity", palette=sns.color_palette("Set2", 8), hue="category",ax=axes[1])
g.set_title("distribution of ethnicity", fontsize=12)


# In[ ]:


#Create figure and Axes. And set title.
fig, axes = plt.subplots(2, 2, figsize=(10,6), gridspec_kw=dict(wspace=0.1, hspace=0.6))
fig.suptitle("Sex and age", fontsize=15)

#Too check layout, I'll show text on each Axes.
gs = axes[0, 1].get_gridspec()
axes[0, 0].remove()
axes[1, 0].remove()
#Add gridspec we got
axbig = fig.add_subplot(gs[:, 0])

g = sns.countplot(data=ds_info, x="sex", hue="category", palette=sns.color_palette("Set2", 8),ax=axbig)
g.set_title("distribution of sex", fontsize=12)

#Add three plots.
g = sns.distplot(ds_info[ds_info["category"]=="train"]["age"], color="tomato", kde=False, rug=False,ax=axes[0,1])
g.set(xlim=(30,80))
g.set(ylim=(0,3))
g.set_title("distribution of age for train", fontsize=12)

g = sns.distplot(ds_info[ds_info["category"]=="test"]["age"], color="teal", kde=False, rug=False, ax=axes[1,1])
g.set(xlim=(30,80))
g.set(ylim=(0,3))
g.set_title("distribution of age for test", fontsize=12)


# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(10,10), gridspec_kw=dict(wspace=0.1, hspace=0.4))
fig.suptitle("Physical information for train", fontsize=15)


g = sns.distplot(ds_info[ds_info["category"]=="train"]["weight_kilograms"], color="tomato", kde=False, rug=False, ax=axes[0,0])
g.set(xlim=(55,135))
g.set(ylim=(0,5))
g.set_title("weight_kilograms", fontsize=12)

g = sns.distplot(ds_info[ds_info["category"]=="train"]["height_centimeters"], color="tomato", kde=False, rug=False, ax=axes[0,1])
g.set(xlim=(155,195))
g.set(ylim=(0,5))
g.set_title("height_centimeters", fontsize=12)

g = sns.distplot(ds_info[ds_info["category"]=="train"]["bmi_kg/m^2"], color="tomato", kde=False, rug=False, ax=axes[1,0])
g.set(xlim=(22,37.5))
g.set(ylim=(0,5))
g.set_title("bmi_kg/m^2", fontsize=12)

g = sns.countplot(ds_info[ds_info["category"]=="train"]["laterality"], ax=axes[1,1])
g.set_title("laterality", fontsize=12)


fig, axes = plt.subplots(2, 2, figsize=(10,10), gridspec_kw=dict(wspace=0.1, hspace=0.4))
fig.suptitle("Physical information for test", fontsize=15)


g = sns.distplot(ds_info[ds_info["category"]=="test"]["weight_kilograms"], color="teal", kde=False, rug=False, ax=axes[0,0])
g.set(xlim=(55,135))
g.set(ylim=(0,5))
g.set_title("weight_kilograms", fontsize=12)

g = sns.distplot(ds_info[ds_info["category"]=="test"]["height_centimeters"], color="teal", kde=False, rug=False, ax=axes[0,1])
g.set(xlim=(155,195))
g.set(ylim=(0,5))
g.set_title("height_centimeters", fontsize=12)

g = sns.distplot(ds_info[ds_info["category"]=="test"]["bmi_kg/m^2"], color="teal", kde=False, rug=False, ax=axes[1,0])
g.set(xlim=(22,37.5))
g.set(ylim=(0,5))
g.set_title("bmi_kg/m^2", fontsize=12)

g = sns.countplot(ds_info[ds_info["category"]=="test"]["laterality"], ax=axes[1,1])
g.set_title("laterality", fontsize=12)


# In[ ]:


ds_info["Ratio_of_medulla_to_cortex"] = ds_info["percent_medulla"] / ds_info["percent_cortex"] 


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(10,5), gridspec_kw=dict(wspace=0.1, hspace=0.6))
fig.suptitle("distribution of ratio of medulla to cortex", fontsize=15)
g = sns.distplot(ds_info[ds_info["category"]=="train"]["Ratio_of_medulla_to_cortex"], color="tomato",kde=False, rug=False, ax=axes[0])
g.set(ylim=(0,5))
g.set_title("train", fontsize=12)
g = sns.distplot(ds_info[ds_info["category"]=="test"]["Ratio_of_medulla_to_cortex"], color="teal", kde=False, rug=False, ax=axes[1])
g.set(ylim=(0,5))
g.set_title("test", fontsize=12)


# In[ ]:


submission = ds_info.to_csv("ds_info.csv")
ds_info.to_csv("submission.csv")

