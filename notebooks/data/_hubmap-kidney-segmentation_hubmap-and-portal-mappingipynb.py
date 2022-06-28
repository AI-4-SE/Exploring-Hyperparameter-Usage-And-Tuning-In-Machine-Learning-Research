#!/usr/bin/env python
# coding: utf-8

# 
# # HuBMAP and Portal Mapping

# <h4> External data from portal.hubmapconsortium added to kaggle datasets is mapped to competition data here </h4>
# <h4> Initial assumption looks at matching image height and width to HuBMAP-20-dataset_information.csv </h4>
# <h4> Imagehash is used to consider if these pairs are likely matches </h4>
# <h4> File for output created that includes HuBMAP dataset information and flags file in test/train, and if FFPE </h4>

# In[ ]:


# https://www.kaggle.com/c/hubmap-kidney-segmentation/discussion/233336
#  External data source ---HuBMAP website : https://portal.hubmapconsortium.org/search?entity_type[0]=Dataset

# # in notebook add data by url for datasets 
# https://www.kaggle.com/narainp/hub-ext-2
# https://www.kaggle.com/narainp/hubmap-ext


# In[ ]:


# imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import tifffile, cv2, gc
from pathlib import Path

from PIL import Image
import imagehash

gc.enable()


# In[ ]:


DATA_PATH1 = '../input/hubmap-ext'
DATA_PATH2 = '../input/hub-ext-2'

print(f'No. of ext images : {len(os.listdir(DATA_PATH1))}')
print(f'No. of ext2 images: {len(os.listdir(DATA_PATH2))}')


# In[ ]:


fnames1 = np.array(os.listdir(DATA_PATH1))
fnames2 = np.array(os.listdir(DATA_PATH2))
print('ext images :',fnames1)
print('ext2 images:',fnames2)


# In[ ]:


# dataset_information used for possible match by same height, width
info = pd.read_csv('../input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv')
info.head()


# In[ ]:


# to check if image will belong to train or test
train = pd.read_csv('../input/hubmap-kidney-segmentation/train.csv')
test =  pd.read_csv('../input/hubmap-kidney-segmentation/sample_submission.csv')
test.head()


# In[ ]:


path1 = Path(DATA_PATH1)
path2 = Path(DATA_PATH2)
hubpath = Path('../input/hubmap-kidney-segmentation/train')
hubpathtst = Path('../input/hubmap-kidney-segmentation/test')
path1, path2


# In[ ]:


def get_hash(image):
    if len(image.shape) == 5:
        image = image.squeeze().transpose(1, 2, 0)
    elif image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    #if image.shape[0] == 3:
    #    image = image.transpose(1, 2, 0)
    image = Image.fromarray(image)    
    hash =  imagehash.average_hash(image)
    del image
    gc.collect()
    return hash


# In[ ]:


def get_hash_hubmap(hubimg):    
    if hubimg.split('.')[0] in test.id.values:
        himage = tifffile.imread(hubpathtst/f"{hubimg}")
    else:
        himage = tifffile.imread(hubpath/f"{hubimg}")
    if len(himage.shape) == 5:
        himage = himage.squeeze().transpose(1, 2, 0)
    elif himage.shape[0] == 3:
        himage = himage.transpose(1, 2, 0)
        
    #if himage.shape[0] == 3:
    #    himage = himage.transpose(1, 2, 0)
    himage = Image.fromarray(himage)  
    hash = imagehash.average_hash(himage)
    del himage
    gc.collect()
    
    return hash


# In[ ]:


# if images are too big notebook will exceed memory, for now these are skipped
print(fnames1[9], fnames2[2])  # testing showed these are too big for imagehash,   h x w 1731207120,  2025172800
hwfails = 1731207120


# # Match first external dataset

# In[ ]:


h1 = []
w1 = []
im1 = []
trn1 = []
hash1 = []
for i in range(len(fnames1)):
    image = tifffile.imread(path1/fnames1[i])  
    # print(fnames1[i])  # testing
    h1.append(image.shape[1])
    w1.append(image.shape[2])
    if (len(info.image_file[(info.height_pixels==image.shape[1]) &  (info.width_pixels==image.shape[2])])) > 0:
        imfile = info.image_file[(info.height_pixels==image.shape[1]) &  (info.width_pixels==image.shape[2])].values[0]  
        trn1.append(imfile.split('.')[0] in test.id.values)            
        if (image.shape[1]*image.shape[2])>=hwfails :   #hwfails  h x w 1731207120  if i==9 known to fail
            hash1.append('too big will fail')
        else:    
            ihash = get_hash(image)
            del image
            gc.collect()
            hhash = get_hash_hubmap(imfile)
            match = ihash==hhash 
            hash1.append(match)
    else:
        imfile = 'unknown'   
        trn1.append('unknown')
        hash1.append('unknown')
    im1.append(imfile)          
            
   # del image
    gc.collect()
    


# In[ ]:


#h1, w1, im1,trn1, hash1   # to check


# In[ ]:


hubportal = pd.DataFrame(
                {          
                'portal_file' : fnames1,
                'p_height_pixels' : h1,
                'p_width_pixels'  : w1,  
                'p_in_test' : trn1,  
                'p_hash_match' : hash1,    
                'image_file' : im1   
                })    
#hubportal.head()


# In[ ]:


# if portal file contains FFPE set flag is FFPE  
def check_4_ffpe(x):
    return ('FFPE' in x.split('.')[0].split('_'))


# In[ ]:


hubportal['is_FFPE'] =   hubportal['portal_file'].apply(lambda x:  check_4_ffpe(x ))
hubportal['p_data_path'] = DATA_PATH1
hubportal.head(10)


# In[ ]:


# this is the one that crashes, may be alternatives that will work TBA 
#ppath = Path(hubportal.p_data_path[9])
#pfname = hubportal.portal_file[9]
#image = tifffile.imread(ppath/pfname)
#if image.shape[0] == 3:
#    image = image.transpose(1, 2, 0)
#image = Image.fromarray(image)
#hash = imagehash.average_hash(image)
#print(hash)


# # Match second external dataset

# In[ ]:


h2 = []
w2 = []
im2 = []
trn2 = []
hash2 = []
for i in range(len(fnames2)):
   # print(fnames2[i])  # testing
    image = tifffile.imread(path2/fnames2[i])    
    h2.append(image.shape[1])
    w2.append(image.shape[2])
    if (len(info.image_file[(info.height_pixels==image.shape[1]) &  (info.width_pixels==image.shape[2])])) > 0:
        imfile = info.image_file[(info.height_pixels==image.shape[1]) &  (info.width_pixels==image.shape[2])].values[0]
        trn2.append(imfile.split('.')[0] in test.id.values) 
        if (image.shape[1]*image.shape[2])>=hwfails :   #hwfails  h x w 1731207120 #if i == 2: # h x w 2025172800
            hash2.append('too big will fail')
        else:  
            ihash = get_hash(image)
            del image
            gc.collect()
            hhash = get_hash_hubmap(imfile)
            
            match = ihash==hhash 
            hash2.append(match)
    else:
        imfile = 'unknown'   
        trn2.append('unknown')
        hash2.append('unknown')  
    im2.append(imfile)      
        
    #del image
    gc.collect()
    
        


# In[ ]:


#h2,w2,im2,trn2,hash2 # to check


# In[ ]:


hubportal2 = pd.DataFrame(
                {          
                'portal_file' : fnames2,
                'p_height_pixels' : h2,
                'p_width_pixels'  : w2, 
                'p_in_test' : trn2,    
                'p_hash_match' : hash2,     
                'image_file' : im2   
                })    
hubportal2.head()


# In[ ]:


hubportal2['is_FFPE'] =   hubportal2['portal_file'].apply(lambda x:  check_4_ffpe(x ))
hubportal2['p_data_path'] = DATA_PATH2
hubportal2.head(11)


# # Merge matches and dataset information to consolidate mapping 

# In[ ]:


hubportal_info = hubportal.append(hubportal2).reset_index(drop=True)


# In[ ]:


hubportal_info= pd.merge(hubportal_info, info, how='left', on='image_file')


# In[ ]:


hubportal_info.head()


# In[ ]:


len(hubportal_info[hubportal_info.width_pixels.isnull()]), hubportal_info.portal_file[hubportal_info.width_pixels.isnull()]
# 2 in portal not matched 
# VAN0011-RK-3-10-PAS_registered.ome.tif   other matched VAN0011 is for patient 67177
# VAN0003-LK-32-21-PAS_registered.ome.tif  other matched VAN0003 is for patient 65631


# In[ ]:


info_images = info.image_file.values
len(info_images)


# In[ ]:


for i in range(len(info_images)):
    if len(hubportal_info[hubportal_info.image_file== info_images[i]]) ==0:
        print(info_images[i])
# c68fe75ea.tiff only image not matched in hubmap_20_dataset_info for patient 67112    
# other matched image for patient 67112 is VAN0010-LK-160-2-PAS_FFPE.ome.tif 2ec3f1bb9.tiff in test


# In[ ]:


len(hubportal_info[hubportal_info.is_FFPE==1]),len(hubportal_info[hubportal_info.is_FFPE==0])  # is_FFPE ==1 9    is_FFPE ==0  12


# In[ ]:


len(hubportal_info[hubportal_info.p_in_test==True]) # all 5 test images mapped also


# # Save file 

# In[ ]:


hubportal_info.to_csv('hubmap_portal_mapping.csv', index=False)

