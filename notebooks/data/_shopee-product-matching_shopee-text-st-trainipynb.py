#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import os
import logging
from datetime import datetime
import gzip
import csv
import math

import cv2, matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import gc 

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


pip install ../input/sentence-transformers/sentence-transformers-1.1.0/


# In[ ]:


# DATA_PATH = '../input/'
DATA_PATH = '../input/shopee-product-matching/'


# In[ ]:


# f1 score metric
def getMetric(col):
    def f1score(row):
        n = len( np.intersect1d(row.target,row[col]) )
        return 2*n / (len(row.target)+len(row[col]))
    return f1score


# In[ ]:


train = pd.read_csv(DATA_PATH + 'train.csv')
train['image'] = DATA_PATH + 'train_images/' + train['image']
tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
train['target'] = train.label_group.map(tmp)


# In[ ]:


train = train.sort_values(by='label_group')
train['title'] = train['title'].str.lower()
train.head()


# In[ ]:


train_triplets_titles = pd.read_csv('../input/shopee-generate-data-for-triplet-loss/' + 'train_triplets_titles.csv')
train_triplets_titles.head()


# In[ ]:


train_triplets_titles.shape


# # tfidf

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())

print(X.shape)


# In[ ]:


X.toarray()


# In[ ]:


idx_x = np.where(X.toarray() > 0)
print(idx_x)
np.asarray(idx_x).shape


# In[ ]:


model = TfidfVectorizer(stop_words=None, binary=True, max_features=55000)
text_embeddings = model.fit_transform(train.title).toarray()
print('text embeddings shape',text_embeddings.shape)


# In[ ]:


import torch
text_embeddings = torch.from_numpy(text_embeddings)
text_embeddings = text_embeddings.cuda()


# In[ ]:


preds = []
CHUNK = 1024*2

print('Finding similar titles...')
CTS = len(train)//CHUNK
if len(train)%CHUNK!=0: CTS += 1
text_ids = None
    
for j in range( CTS ):
    
    a = j*CHUNK
    b = (j+1)*CHUNK
    b = min(b,len(train))
    print('chunk',a,'to',b)
    
    cts = torch.matmul(text_embeddings, text_embeddings[a:b].T).T
    cts = cts.data.cpu().numpy()
    for k in range(b-a):
        IDX = np.where(cts[k,]>0.6)[0]
        o = train.iloc[IDX].posting_id.values
        preds.append(o)
        
    del cts
    torch.cuda.empty_cache()


# In[ ]:


del text_embeddings
torch.cuda.empty_cache()

train['oof_text'] = preds


# In[ ]:


COMPUTE_CV = True

if COMPUTE_CV:
    train['f1'] = train.apply(getMetric('oof_text'),axis=1)
    print('CV score for baseline =',train.f1.mean())


# # Word2Vec

# In[ ]:


from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

vectors = KeyedVectors.load_word2vec_format("../input/glove2word2vec/glove_w2v.txt") # import the data file


# In[ ]:


from nltk.tokenize import word_tokenize
train_title_token = train['title'].apply(lambda x: word_tokenize(x))


# In[ ]:


text_embeddings = []
for title in tqdm_notebook(train_title_token[:]):
    title_feat = []
    for word in title:
        if word in vectors:
            title_feat.append(vectors[word])
    
    if len(title_feat) == 0:
        title_feat = np.random.rand(200)
    else:
        # max-pooling
        # mean-pooling
        # IDF
        # SIF
        title_feat = np.vstack(title_feat).max(0)
    text_embeddings.append(title_feat)
    # break


# In[ ]:


from sklearn.preprocessing import normalize

# l2 norm to kill all the sim in 0-1
text_embeddings = np.vstack(text_embeddings)
text_embeddings = normalize(text_embeddings)

import torch
text_embeddings = torch.from_numpy(text_embeddings)
text_embeddings = text_embeddings.cuda()


# In[ ]:


preds = []
CHUNK = 1024*4


print('Finding similar images...')
CTS = len(text_embeddings)//CHUNK
if len(text_embeddings)%CHUNK!=0: CTS += 1
for j in range( CTS ):
    
    a = j*CHUNK
    b = (j+1)*CHUNK
    b = min(b,len(train))
    print('chunk',a,'to',b)
    
    cts = torch.matmul(text_embeddings, text_embeddings[a:b].T).T
    cts = cts.data.cpu().numpy()
    for k in range(b-a):
        IDX = np.where(cts[k,]>0.93)[0]
        o = train.iloc[IDX].posting_id.values
        preds.append(o)
        
    del cts
    torch.cuda.empty_cache()


# In[ ]:


train['oof_w2v'] = preds


# In[ ]:


COMPUTE_CV = True

if COMPUTE_CV:
    train['f1'] = train.apply(getMetric('oof_w2v'),axis=1)
    print('CV score for baseline =',train.f1.mean())


# # Sentence Transformers

# In[ ]:


def combine_for_oof(row):
    x = np.concatenate([row.oof_text,row.oof_w2v])
    return np.unique(x)

# merge product proposal by tfidf and word2vec, we have positive and negative example.
# if two product in same group they are positive label.
train['oof'] = train.apply(combine_for_oof,axis=1)


# In[ ]:


train_pair = train.set_index('posting_id')


# In[ ]:


train_pair.head()


# In[ ]:


import torch
from torch.utils.data import DataLoader
from sentence_transformers import models, losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *


# In[ ]:


batch_size = 32
model_save_path = '/kaggle/working/models/training_shopee_title_embeddings-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# In[ ]:


get_ipython().system('mkdir /kaggle/working/models')


# In[ ]:


get_ipython().system('ls /kaggle/working/models')


# In[ ]:


from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

#Define the model. Either from scratch of by loading a pre-trained model
#model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
st_model = SentenceTransformer('stsb-roberta-base')


# In[ ]:


train_examples = []
title_pair = []
for row in tqdm_notebook(train_pair.iterrows()):
    for pair in row[1].oof:
        # not match self
        if pair == row[0]:
            continue
    
        if pair in row[1].target:
            lbl = 1.0
        else:
            lbl = 0.0
        title1 = row[1].title
        title2 = train_pair.loc[pair]['title']
            
        inputExample = InputExample( texts=[ row[1].title, train_pair.loc[pair]['title'] ], label=lbl)
        train_examples.append(inputExample)
        title_pair.append(
            [row[1].title, train_pair.loc[pair]['title'], lbl]
        )


# In[ ]:


title_pair = pd.DataFrame(title_pair, columns=['s1', 's2', 'label'])
title_pair = title_pair.sample(frac=1)
title_pair.head(5)


# In[ ]:


title_pair.shape


# In[ ]:


from sklearn.model_selection import train_test_split
train_inputs, test_inputs = train_test_split(train_examples,test_size=0.1,random_state=42)


# In[ ]:


print(len(train_inputs), len(test_inputs))


# In[ ]:


train_dataloader = DataLoader(train_inputs, shuffle=True, batch_size=32)
train_loss = losses.CosineSimilarityLoss(st_model)


# In[ ]:


from sentence_transformers import evaluation
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_inputs, name='shopee-titles-test')


# In[ ]:


print(st_model)


# In[ ]:


st_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator, evaluation_steps=500, output_path=model_save_path)


# In[ ]:


get_ipython().system('ls $model_save_path')


# In[ ]:


#model2 = SentenceTransformer(model_path)
#print(model2)


# In[ ]:


print(st_model)


# In[ ]:


st_model.evaluate(evaluator)


# In[ ]:


text_embeddings = model.encode(train.title)
print('text embeddings shape',text_embeddings.shape)


# In[ ]:


from sklearn.preprocessing import normalize

# l2 norm to kill all the sim in 0-1
text_embeddings = np.vstack(text_embeddings)
text_embeddings = normalize(text_embeddings)

import torch
text_embeddings = torch.from_numpy(text_embeddings)
text_embeddings = text_embeddings.cuda()


# In[ ]:


preds = []
CHUNK = 1024*2

print('Finding similar titles...')
CTS = len(train)//CHUNK
if len(train)%CHUNK!=0: CTS += 1
text_ids = None
    
for j in range( CTS ):
    
    a = j*CHUNK
    b = (j+1)*CHUNK
    b = min(b,len(train))
    print('chunk',a,'to',b)
    
    cts = torch.matmul(text_embeddings, text_embeddings[a:b].T).T
    cts = cts.data.cpu().numpy()
    for k in range(b-a):
        IDX = np.where(cts[k,]>0.93)[0]
        o = train.iloc[IDX].posting_id.values
        preds.append(o)
        
    del cts
    torch.cuda.empty_cache()


# In[ ]:


train['oof_bert'] = preds


# In[ ]:


if COMPUTE_CV:
    train['f1'] = train.apply(getMetric('oof_bert'),axis=1)
    print('CV score for baseline =',train.f1.mean())


# In[ ]:


text_embeddings.shape


# In[ ]:




