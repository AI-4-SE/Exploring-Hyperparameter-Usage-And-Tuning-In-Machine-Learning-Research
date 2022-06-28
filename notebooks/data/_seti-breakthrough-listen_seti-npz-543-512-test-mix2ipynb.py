#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from tqdm.auto import tqdm
import os
import cv2
save_dir = f'/kaggle/tmp/train/'
os.makedirs(save_dir, exist_ok=True)

for dirname, _, filenames in os.walk(f'../input/seti-breakthrough-listen/test'):
    for filename in tqdm(filenames):
        img = np.load(os.path.join(dirname, filename))
        img = img + 25 
        img0 = img[0]
        img2 = img[2]
        img4 = img[4]
        img024 = (img[0] + img[2] + img[4]) / 3
        img02 = np.vstack((img0,img2))
        img4024 = np.vstack((img4,img024))
        img = np.hstack((img02,img4024))
        np.save(save_dir + filename, img)

get_ipython().system('tar -zcf image.tar.gz -C "/kaggle/tmp/train/" .')

