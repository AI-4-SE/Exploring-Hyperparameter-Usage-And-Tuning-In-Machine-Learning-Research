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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Exploratory data analysis: QnA 
# 
# Checkout what we have to submit?
# - The submission sample has 2 columns: *id* and *PredictionString*

# In[ ]:


sample_submission = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/sample_submission.csv")


# In[ ]:


sample_submission.sample(5)


# Looking at the train and test data that we have..

# In[ ]:


train_df = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/train.csv")
train_df.sample(2)


# In[ ]:


train_df["language"].value_counts()


# Training data is relatively small, with 746 `hindi` and 368 `tamil` examples. 
# 
# We have:
# - id
# - context 
# - question
# - answer_text
# - answer_start
# 
# Task: We have context available and we need to find the start and end span in that context that answers our given question. Similar example of such format is [SQUAD](https://rajpurkar.github.io/SQuAD-explorer/)

# In[ ]:


#test data
test_df = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/test.csv")
test_df.sample(5)


# In test data we have 5 sample data and we do not have *answer_text* and *answer_start* columns

# ### Metric
# 
# Evaluation is based on Jaccard score, which is an intersection over union (IoU) metric. It measures how many words from the context we picked correctly.
# 
# ### Baseline model
# 
# Using the Huggingface library and this repo on github:
# https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb

# In[ ]:


import transformers


# In[ ]:


get_ipython().system('ls "../input/"')


# In[ ]:


model_checkpoint = "deepset/xlm-roberta-large-squad2"
batch_size = 4


# In[ ]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# In[ ]:


train_df["num_tokens_context"] = train_df["context"].apply(lambda t: len(tokenizer(t)["input_ids"]))


# In[ ]:


train_df["num_tokens_context"]

