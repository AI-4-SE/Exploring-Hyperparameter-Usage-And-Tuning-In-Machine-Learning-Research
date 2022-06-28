#!/usr/bin/env python
# coding: utf-8

# ## A look at the data

# Let's start out by setting up our environment by importing the required modules and setting a random seed:

# In[ ]:


import numpy as np
import os
import pandas as pd
from fastai.vision.all import *


# In[ ]:


set_seed(999,reproducible=True)


# In[ ]:


dataset_path = Path('../input/cassava-leaf-disease-classification')
os.listdir(dataset_path)


# In[ ]:


train_df = pd.read_csv(dataset_path/'train.csv')


# In[ ]:


train_df.head()


# In[ ]:


train_df['path'] = train_df['image_id'].map(lambda x:dataset_path/'train_images'/x)
train_df = train_df.drop(columns=['image_id'])
train_df = train_df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
train_df.head(10)


# In[ ]:


len_df = len(train_df)
print(f"There are {len_df} images")


# In[ ]:


#delete part of label 3 data. Makes result worse!!! 
#0.3 frac gives about 76% accuracy, 
#0.7 frac gives about 82% accuracy

#filtered_train_df = train_df[train_df['label'] != 3]
#label3_df = train_df[train_df['label'] == 3]
#label3_df = label3_df.sample(frac = 0.7)

#result = pd.concat([filtered_train_df,label3_df])
#train_df = result.sample(frac=1).reset_index(drop=True) #shuffle dataframe


# All dataset >21,000 images
# 
# The distribution of the different classes:

# In[ ]:


train_df['label'].hist(figsize = (10, 5))


# Categories
# 0. Cassava Bacterial Blight (CBB)
# 1. Cassava Brown Streak Disease (CBSD)
# 2. Cassava Green Mottle (CGM)
# 3. Cassava Mosaic Disease (CMD)
# 4. Healthy
# 

# ## Data loading
# * The item transforms performs a large crop on each of the images.
# * The batch transforms performs random resized crop to 224 and also apply other standard augmentations (in `aug_tranforms`) at the batch level on the GPU.
# * The batch size is set to 256 here.
# 

# In[ ]:


item_tfms = RandomResizedCrop(460, min_scale=0.75, ratio=(1.,1.))
batch_tfms = [*aug_transforms(size=224, max_warp=0), Normalize.from_stats(*imagenet_stats)]
bs=256


# In[ ]:


dls = ImageDataLoaders.from_df(train_df, #pass in train DataFrame
                               valid_pct=0.2, #80-20 train-validation random split
                               seed=999, #seed
                               label_col=0, #label is in the first column of the DataFrame
                               fn_col=1, #filename/path is in the second column of the DataFrame
                               bs=bs, #pass in batch size
                               item_tfms=item_tfms, #pass in item_tfms
                               batch_tfms=batch_tfms) #pass in batch_tfms


# In[ ]:


dls.show_batch()


# ## Model training:

# In[ ]:


# Making pretrained weights work without needing to find the default filename
if not os.path.exists('/root/.cache/torch/hub/checkpoints/'):
        os.makedirs('/root/.cache/torch/hub/checkpoints/')
get_ipython().system("cp '../input/resnet50/resnet50.pth' '/root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth'")


# In[ ]:


learn = cnn_learner(dls, 
                    resnet50, 
                    loss_func = LabelSmoothingCrossEntropy(), 
                    metrics = [accuracy], 
                    #cbs=MixUp()
                   ).to_native_fp16()


# In[ ]:


learn.lr_find()


# `learn.fine_tune` trains frozen pretrained model for a single epoch (using one-cycle training), then train the whole pretrained model for several epochs using one-cycle training.

# In[ ]:


learn.fine_tune(5,base_lr=1e-2)


# In[ ]:


learn.recorder.plot_loss()


# In[ ]:


#Remove CBS to prevent mess up with learn.get_preds, predict, etc
#learn.remove_cbs([MixUp])


# Put the model back to fp32, and export the model 

# In[ ]:


learn = learn.to_native_fp32()


# In[ ]:


learn.export()


# **Checking the confusion matrix:**

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()

