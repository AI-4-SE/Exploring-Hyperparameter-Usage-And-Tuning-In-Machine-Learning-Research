#!/usr/bin/env python
# coding: utf-8

# ## EXPLORATORY DATA ANALYSIS

# In[ ]:


#Installing and importing the dependency
get_ipython().system('pip install timm')


# In[ ]:


get_ipython().system('pip show timm')


# You may come across the question why i am installing timm in image classification problem i am going use efficientnet so i am installing that.

# In[ ]:


import timm


# In[ ]:


from fastai.vision.all import *


# In[ ]:


train_df = pd.read_csv("../input/paddy-disease-classification/train.csv")
train_df.sample(5)


# In[ ]:


train_df.variety.value_counts()


# In[ ]:


train_df["label"].value_counts()


# In[ ]:


train_df.info()


# In[ ]:


train_df.iloc[0]


# In[ ]:


train_images = get_image_files("../input/paddy-disease-classification/train_images")
train_images


# In[ ]:


train_images[0]


# In[ ]:


samp_sub = pd.read_csv("../input/paddy-disease-classification/sample_submission.csv")
samp_sub.sample()


# In[ ]:


data_dir = Path("../input/paddy-disease-classification")
train_dir = Path("../input/paddy-disease-classification/train_images")
test_dir = Path("../input/paddy-disease-classification/test_images")


# In[ ]:


data_dir.ls()


# In[ ]:


(train_dir/"tungro").ls()


# In[ ]:


Path.BASE_PATH = train_dir


# In[ ]:


train_dir.ls()


# In[ ]:


(train_dir/"hispa").ls()


# In[ ]:


test_dir.ls()


# ## DATA PREPARATION

# I dont use the train_df for labels because paddy images are put in a folder with name, so i am going to use it.

# In[ ]:


paddy = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y= parent_label,
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))
dls = paddy.dataloaders(train_dir)


# Checking our dataloaders are build good aor bad by checking train and valid show batch.

# In[ ]:


dls.valid.show_batch(max_n=10, nrows=2)


# In[ ]:


dls.train.show_batch(max_n=10, nrows=2)


# In[ ]:


paddy.summary(train_dir)


# I was using the model architecture like efficientnetB0, resnet34, resnet18, sometime using SGD or Adam, Everthing about trying let's try thats my mind says.

# In[ ]:


learn = vision_learner(dls, "efficientnet_b0", metrics=error_rate, opt_func=Adam)


# In[ ]:


lr_min,lr_steep = learn.lr_find(suggest_funcs=(minimum, steep))
print(f"Minimum/10: {lr_min:.2e}, steepest point: {lr_steep:.2e}")


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'learn.fine_tune')


# In[ ]:


learn = vision_learner(dls, "efficientnet_b0", metrics=error_rate, opt_func=Adam)
learn.fit_one_cycle(3, 9.12e-03)
learn.unfreeze()


# In[ ]:


lr_min,lr_steep = learn.lr_find(suggest_funcs=(minimum, steep))
print(f"Minimum/10: {lr_min:.2e}, steepest point: {lr_steep:.2e}")


# In[ ]:


learn.fit_one_cycle(12, lr_max=slice(1e-6,1e-4))


# In[ ]:


learn1 = vision_learner(dls, resnet34, metrics=error_rate)
learn1.fit_one_cycle(3, 3e-3)
learn1.unfreeze()
learn1.fit_one_cycle(12, lr_max=slice(1e-6,1e-4))


# In[ ]:


from fastai.callback.fp16 import *
learn2 = vision_learner(dls, resnet18, metrics=error_rate).to_fp16()
learn2.fine_tune(6, freeze_epochs=3)


# In[ ]:


samp_sub.head()


# In[ ]:


test = test_dir.ls()
test


# In[ ]:


dl = learn.dls.test_dl(test)


# In[ ]:


dl


# In[ ]:


pred = learn.get_preds(dl=dl)


# In[ ]:


pred


# In[ ]:


preds = learn.get_preds(dl=dl)[0].argmax(1).numpy()
preds[:5]


# In[ ]:


train_df


# In[ ]:


samp_sub


# In[ ]:


preds


# In[ ]:


sub = pd.DataFrame({'image_id':samp_sub.image_id, 'label': preds})
sub.head()


# In[ ]:


labels = os.listdir('../input/paddy-disease-classification/train_images')
labels


# In[ ]:


ss = pd.read_csv('../input/paddy-disease-classification/sample_submission.csv')
ss['label'] = preds
ss['label'] = ss['label'].replace([0,1,2,3,4,5,6,7,8,9], labels)
ss.to_csv("EfficientB0.csv",index=False)
ss.head()


# In[ ]:


dl = learn1.dls.test_dl(test)


# In[ ]:


pred1 = learn1.get_preds(dl=dl)


# In[ ]:


pred1 = learn1.get_preds(dl=dl)[0].argmax(1).numpy()


# In[ ]:


ss1 = pd.read_csv('../input/paddy-disease-classification/sample_submission.csv')
ss1['label'] = pred1
ss1['label'] = ss['label'].replace([0,1,2,3,4,5,6,7,8,9], labels)
ss1.to_csv("resnet34.csv",index=False)
ss1.head()


# In[ ]:


dl = learn2.dls.test_dl(test)


# In[ ]:


pred2 = learn2.get_preds(dl=dl)


# In[ ]:


pred2 = learn2.get_preds(dl=dl)[0].argmax(1).numpy()


# In[ ]:


ss1 = pd.read_csv('../input/paddy-disease-classification/sample_submission.csv')
ss1['label'] = pred2
ss1['label'] = ss['label'].replace([0,1,2,3,4,5,6,7,8,9], labels)
ss1.to_csv("resnet18.csv",index=False)
ss1.head()

