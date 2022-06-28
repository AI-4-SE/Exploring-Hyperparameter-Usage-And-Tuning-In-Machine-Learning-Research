#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
import sys

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


os.getcwd()


# In[ ]:


get_ipython().system('ls -a')


# In[ ]:


get_ipython().system('ls /kaggle/input/')


# In[ ]:


## fastai import statements for vision not including fastbook
from fastai.vision.all import *
#from fastbook import *


# ## Understanding the path structure and lenghts

# In[ ]:


## ../input/herbarium-2020-fgvc7/nybg2020/train/metadata.json
train_path = "../input/herbarium-2020-fgvc7/nybg2020/train/images/"
test_path = "../input/herbarium-2020-fgvc7/nybg2020/test/images/"
image_path = "../input/herbarium-2020-fgvc7/nybg2020/train/images/000/00/437000.jpg"

export_path = "../input/200kp5maxspec89epochresnet50/export.pkl"
#export_path = "../input/min-5-df-and-export/export-min-5.pkl"
#df_path = "../input/100k-10-epochs/df-100k-10-epochs.csv"


# In[ ]:


## See one image
import os
#print(os.getcwd())
print("Number of folders in train/images/ is", len(next(os.walk(train_path))[1]))
print("Number of folders in train/images/000 is", len(next(os.walk(train_path+"000/"))[1]))
print("Number of files in train/images/000/00/ is", len(next(os.walk(train_path+"000/00/"))[2]))
print("")
print("Number of folders in test/images/ is", len(next(os.walk(test_path))[1]))
print("Number of files in test/images/000/ is", len(next(os.walk(test_path+"001/"))[2]))


# ## Understanding Json file contents

# In[ ]:


train_json_path = train_path+"../metadata.json"
test_json_path = test_path+"../metadata.json"


# In[ ]:


sz_tr = os.path.getsize(train_json_path)
sz_te = os.path.getsize(test_json_path)
print("Size of train metadata is, ",sz_tr/10**6,"MB", "Size of test metadata is", sz_te/10**6, "MB")


# In[ ]:


with open(train_json_path, encoding="utf8", errors='ignore') as f:
     tr_metadata = json.load(f)
        
with open(test_json_path, encoding="utf8", errors='ignore') as f:
     te_metadata = json.load(f)


# In[ ]:


tr_metadata.keys(),te_metadata.keys()


# In[ ]:


## Length of each of the keys!
print([(name,len(tr_metadata[name])) for name in tr_metadata.keys()])
[(name,len(te_metadata[name])) for name in te_metadata.keys()]


# In[ ]:


num = 1030746
print("annotations",tr_metadata["annotations"][num], "type", type(tr_metadata["annotations"]))
print("images",tr_metadata["images"][num])


# In[ ]:


tr_metadata["annotations"][num]["category_id"], tr_metadata["images"][num]["file_name"]


# In[ ]:


## "Licenses" is a list but "info" is not a list. Both have not more than 1 indice at max.
print("categories",tr_metadata["categories"][0:2])# index till 32000 interesting.
print("licenses",tr_metadata["licenses"],"\n")
print("regions",tr_metadata["regions"][0], "\n")
print(tr_metadata["info"], type(tr_metadata["info"]))
print(te_metadata["info"], type(te_metadata["info"]))


# In[ ]:


## Length characteristics of the meta data
n_tr_img = len(tr_metadata["annotations"])
len(tr_metadata["annotations"])==len(tr_metadata["images"]), n_tr_img/10**6


# In[ ]:


im = Image.open("../input/herbarium-2020-fgvc7/nybg2020/train/images/000/00/437000.jpg")
im.to_thumb(250,250)
                


# ## Make Test df (image id, filename, category_od)

# In[ ]:


## never grow a DF, make a list (of lists) and then convert it to pandas (https://stackoverflow.com/a/56746204/5986651)
lst_df_test = [[te_metadata["images"][num]["id"], te_metadata["images"][num]["file_name"]] for num in range(len(te_metadata["images"]))]
lst_df_test[0:5]


# In[ ]:


## Convert list to DF ("category_id" and "image_id" are from "annotations". The "filepath" is from )
dft= pd.DataFrame.from_records(lst_df_test)
dft.columns  = ["image_id", "filepath"]
print(len(dft))
dft[0:10]


# In[ ]:


## size of dft
from pympler import asizeof
asizeof.asizeof(dft)


# In[ ]:


## Split dataframes into several to avoid memory issues
no_splits = 100
dfs = np.array_split(dft,100,axis=0)
len(dfs),len(dfs[1]), len(dft),type(dfs)


# In[ ]:


## check CPU GPU USAGE
def print_cpu_gpu_usage():
    get_ipython().system('gpustat -cp')
    get_ipython().system('free -m')
    #!top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}'
    
print_cpu_gpu_usage()


# ## Prediction

# In[ ]:


## Load learner
def get_x(r): return "../input/herbarium-2020-fgvc7/nybg2020/test/"+r["filepath"]
def get_y(r): return r["category_id"]

learn_inf = load_learner(export_path, cpu=False)
#learn_inf = load_learner("export.pkl", cpu=False)
## check if you are running with CPU or GPU?
learn_inf.dls.device


# In[ ]:


## Size of inference object
asizeof.asizeof(learn_inf)


# In[ ]:


print_cpu_gpu_usage()


# In[ ]:


## predict function
def predict_func(dfv):
    valid_dl = learn_inf.dls.test_dl(dfv, bs=256) ## Make DL
    pred_tens = learn_inf.get_preds(dl=valid_dl) ## Get preds (proltys)
    pred_argmax = pred_tens[0].argmax(dim=1) ## argmax
    pred_categ = learn_inf.dls.vocab[pred_argmax] ## Get Category ID
    dfv.loc[:,"pred_cat"] = pred_categ ## Add to dfv
    print(pred_argmax.shape)
    
    #print("Accuracy is",sum(dfv["pred_cat"] == dfv["category_id"])/len(dfv)) ## Print Accuracy for Valid


# In[ ]:


## Predict for each small dataframe
for i in range(len(dfs)):
    predict_func(dfs[i])
    print_cpu_gpu_usage()
    print(dfs[i][0:10])


# In[ ]:


## Joining the dataframe
dft = pd.concat(dfs)
#dft = dfs[0].append(dfs[1:len(dfs)])


# In[ ]:


## Save output

dft.to_csv("my_results.csv")
dft.drop("filepath",1, inplace=True)
dft.columns = ["Id","Predicted"]
dft.to_csv("my_submission.csv", index=False)


# In[ ]:


print_cpu_gpu_usage

