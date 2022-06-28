#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[ ]:


len(os.listdir("../input/test"))


# In[ ]:


get_ipython().system('ls ../input/train')


# In[ ]:


CATEGORIES=['Type_1','Type_2','Type_3']
train_dir = '../input/train'
test_dir = '../input/test'
test_dir2 = '../input/test_stg2'


# In[ ]:


import gc
gc.collect()


# In[ ]:


for category in CATEGORIES:
    print('{} {} images'.format(category, len(os.listdir(os.path.join(train_dir, category)))))


# In[ ]:


len(os.listdir(test_dir))


# In[ ]:


train = []
for  category in (CATEGORIES):
    img=list(os.listdir(os.path.join(train_dir, category)))
    for file in img:
        if file.endswith('.jpg'):
            train.append([file,category])
train = pd.DataFrame(train, columns=['img','tag'])
            
        
        #if file.endswith('.jpg'):
            #print (file)
            


# In[ ]:


train.shape


# In[ ]:


train.to_csv('./cervical.csv',index=False)


# In[ ]:


get_ipython().system('mkdir ./train')
get_ipython().system('mkdir ./test')


# In[ ]:


get_ipython().system('ls ../input/train/Type_1')


# In[ ]:


from PIL import Image
im=Image.open('../input/train/Type_3/1284.jpg')


# In[ ]:


im.mode


# In[ ]:


im.resize((512,512))


# In[ ]:


im='../input/train/Type_1/513.jpg'


# ## Lets resize all our images and then save them

# In[ ]:


#First train
from scipy.misc import imread, imsave, imresize
from PIL import Image
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from fastai import transforms

for  category in (CATEGORIES):
    source = os.listdir(os.path.join(train_dir, category))
    source2=f'{train_dir}/{category}/'
    img=list(source)
    destination="./train/"
    for files in img:
        if files.endswith('.jpg'):
            im= cv2.imread(f'{source2}{files}')
            
            print(f'{source2}{files}')
            b,g,r = cv2.split(im)
            im2 = cv2.merge([r,g,b])
            im3 = Image.fromarray(transforms.scale_min(im2,512))
            im3.save(f'{destination}{files}')
        
#try:
        
      
#except (ValueError):
     #print(f'{source2}{files}')
    


# In[ ]:


len(os.listdir('./train'))


# In[ ]:


#Now Test
from scipy.misc import imread, imsave, imresize
from PIL import Image
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from fastai import transforms

source = os.listdir(os.path.join(test_dir))
source2=f'{test_dir}/'
img=list(source)
destination="./test/"
for files in img:
    if files.endswith('.jpg'):
        im= cv2.imread(f'{source2}{files}')
        print(f'{source2}{files}')
        b,g,r = cv2.split(im)
        im2 = cv2.merge([r,g,b])
        im3 = Image.fromarray(transforms.scale_min(im2,512))
        im3.save(f'{destination}{files}')
        


# In[ ]:


#Now Test stage2
from scipy.misc import imread, imsave, imresize
from PIL import Image
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from fastai import transforms

source = os.listdir(os.path.join(test_dir2))
source2=f'{test_dir2}/'
img=list(source)
destination="./test/"
for files in img:
    if files.endswith('.jpg'):
        im= cv2.imread(f'{source2}{files}')
        print(f'{source2}{files}')
        b,g,r = cv2.split(im)
        im2 = cv2.merge([r,g,b])
        im3 = Image.fromarray(transforms.scale_min(im2,512))
        im3.save(f'{destination}{files}')


# In[ ]:


len(os.listdir('./test'))


# In[ ]:


img = cv2.imread('./test/12.jpg')
img2 = cv2.imread('../input/test/12.jpg')


# In[ ]:


plt.imshow(img)


# In[ ]:


train='./train'
test='./test'


# In[ ]:


label_csv = './cervical.csv'
n = len(list(open(label_csv)))-2
val_idxs = get_cv_idxs(n)


# In[ ]:


PATH=''
arch=resnet34
sz=128
data = ImageClassifierData.from_csv(PATH,train,label_csv,bs=10,val_idxs=val_idxs,tfms=tfms_from_model(arch, sz,max_zoom=1.1),test_name=test)


# In[ ]:


learn = ConvLearner.pretrained(arch, data, precompute=False)


# In[ ]:


lrf=learn.lr_find()


# In[ ]:


learn.sched.plot()


# In[ ]:


lr=5e-5
learn.fit(lr,3, cycle_len=1, cycle_mult=2)


# In[ ]:


lr=5e-3
learn.fit(lr,3, cycle_len=1, cycle_mult=2)


# In[ ]:


lr=5e-2
learn.fit(lr,3, cycle_len=1, cycle_mult=2)


# In[ ]:


lrs=np.array([lr/6,lr/3,lr])


# In[ ]:


learn.unfreeze()
learn.fit(lrs,3, cycle_len=1, cycle_mult=2)


# In[ ]:


learn.fit(lrs,3, cycle_len=1, cycle_mult=2)


# In[ ]:


def get_data(sz,bs):
    
    data=ImageClassifierData.from_csv(PATH,train,label_csv,bs,val_idxs=val_idxs,tfms=tfms_from_model(arch, sz,max_zoom=1.1),test_name=test)
    return data


# In[ ]:


learn.set_data(get_data(256,10))


# In[ ]:


learn.freeze()


# In[ ]:


lr=1e-4
learn.fit(lr,3, cycle_len=1, cycle_mult=2)


# In[ ]:


log_preds, y = learn.TTA(is_test=True)
probs = np.mean(np.exp(log_preds),0)
ds=pd.DataFrame(probs)
ds.columns=data.classes
ds.insert(0,'id',[o.rsplit('/', 1)[1] for o in data.test_ds.fnames])


# In[ ]:


#Check for validaton sample
log_preds, y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)


# In[ ]:


accuracy_np(probs,y)


# In[ ]:


# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))

# create a link to download the dataframe
create_download_link(ds)


# In[ ]:




