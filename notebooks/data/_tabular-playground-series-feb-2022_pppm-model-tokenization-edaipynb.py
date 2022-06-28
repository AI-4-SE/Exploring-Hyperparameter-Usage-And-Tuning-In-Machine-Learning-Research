#!/usr/bin/env python
# coding: utf-8

# The goal of this notebook is to evaluate how well does the vocabulary of each language model represent the patent data. If a model's tokenizer doesn't contain a word, it will split it into subwords. Therefore, we can look at the distribution of the number of tokens per word. This might be a good signal for whether a model is suitable for this competition. 
# 
# I created the notebook following a suggestion from the user @hengck23.

# In[ ]:


get_ipython().system(' pip install -q transformers sentencepiece')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
from transformers import AutoTokenizer
import transformers.utils.logging

transformers.utils.logging.disable_progress_bar()

import transformers



df_train = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/train.csv')

def plot_avg_tokens_per_word(df_train, model_name):
    df_train = df_train.copy(deep=True)
    
    if 'cocolm' in model_name:
        tokz = COCOLMTokenizer.from_pretrained(model_name)
    elif 'roberta' in model_name:
        tokz = transformers.AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    else:
        tokz = transformers.AutoTokenizer.from_pretrained(model_name)

        
    df_train['num_toks'] = df_train.anchor.apply(lambda x: len(tokz.convert_ids_to_tokens(tokz.encode(x.split(' '), is_split_into_words=True)))-2)
    df_train['num_words'] = df_train.anchor.apply(lambda x: len(x.split(' ')))
    df_train['tok_rate'] = df_train['num_toks'] / df_train['num_words']
    df_train['tok_rate'].hist()

    df_train['tok_rate'].hist()
    plt.xlabel('Number of Tokens per Word')
    plt.ylabel('Frequency')
    plt.title(f'model: {model_name}')
    
    avg_num_toks_per_word = (df_train['num_toks'] / df_train['num_words']).mean()
    return avg_num_toks_per_word

avg_tokens_per_word = {}


# In[ ]:


avg_tokens_per_word['bert-base-uncased'] = plot_avg_tokens_per_word(df_train, 'bert-base-uncased')


# In[ ]:


avg_tokens_per_word['anferico/bert-for-patents'] = plot_avg_tokens_per_word(df_train,'anferico/bert-for-patents')


# In[ ]:


avg_tokens_per_word['roberta-large'] = plot_avg_tokens_per_word(df_train,'roberta-large')


# In[ ]:


avg_tokens_per_word['microsoft/deberta-v3-small'] = plot_avg_tokens_per_word(df_train,'microsoft/deberta-v3-small')


# In[ ]:


avg_tokens_per_word['microsoft/deberta-v3-large'] = plot_avg_tokens_per_word(df_train,'microsoft/deberta-v3-large')


# In[ ]:


avg_tokens_per_word['allenai/scibert_scivocab_uncased'] = plot_avg_tokens_per_word(df_train,'allenai/scibert_scivocab_uncased')


# **Comparison of model tokenizers by average number of tokens per word**

# In[ ]:


pd.DataFrame.from_dict(avg_tokens_per_word, orient='index', columns=['avg_tokens_per_word']).sort_values(by='avg_tokens_per_word').plot(kind='bar')


# In[ ]:




