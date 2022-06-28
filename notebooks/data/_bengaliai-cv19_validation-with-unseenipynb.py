#!/usr/bin/env python
# coding: utf-8

# Hi every one. I've made you guys a validation split, which considered unseen graphemes.
# 
# It have only 1245 graphemes in training set, while all components are remains. All unseen graphemes are used in every fold while training.
# 
# Using my split, we will have approximately **38k SEEN** samples, and exacly **7578 UNSEEN** samples for validation in each fold.
# 
# Download the file `train_v2.csv` generated by this script and have a try on yourself!
# 
# Usage Example:
# ```
# df_train = pd.read_csv('path/to/train_v2.csv')
# n_fold = 5
# for fold in range(n_fold):
#     train_idx = np.where((df_train['fold'] != fold) & (df_train['unseen'] == 0))[0]
#     valid_idx = np.where((df_train['fold'] == fold) | (df_train['unseen'] != 0))[0]  # all unseen graphemes are in each fold
# ```

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# In[ ]:


df_train = pd.read_csv('../input/bengaliai-cv19/train.csv')


# In[ ]:


df_train.head(2)


# # Map grapheme to id

# In[ ]:


grapheme2idx = {grapheme: idx for idx, grapheme in enumerate(df_train.grapheme.unique())}
df_train['grapheme_id'] = df_train['grapheme'].map(grapheme2idx)


# In[ ]:


df_train.head(2)


# # StratifiedKFold On Grapheme

# In[ ]:


n_fold = 5
skf = StratifiedKFold(n_fold, random_state=42)
for i_fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train.grapheme)):
    df_train.loc[val_idx, 'fold'] = i_fold
df_train['fold'] = df_train['fold'].astype(int)


# In[ ]:


df_train.head(2)


# # Add Unseen Flag

# In[ ]:


df_train['unseen'] = 0
df_train.loc[df_train.grapheme_id >= 1245, 'unseen'] = 1


# In[ ]:


df_train.unseen.value_counts()


# In[ ]:


df_train.loc[df_train['unseen'] == 1, 'fold'] = -1


# In[ ]:


df_train['fold'].value_counts()


# In[ ]:


df_train.head(2)


# In[ ]:


df_train.to_csv('train_v2.csv', index=False)


# # Usage Example

# In[ ]:


n_fold = 5
for fold in range(n_fold):
    train_idx = np.where((df_train['fold'] != fold) & (df_train['unseen'] == 0))[0]
    valid_idx = np.where((df_train['fold'] == fold) | (df_train['unseen'] != 0))[0]

    df_this_train = df_train.loc[train_idx].reset_index(drop=True)
    df_this_valid = df_train.loc[valid_idx].reset_index(drop=True)
    
    #################################
    # Do training and validating here
    #################################
    
    break


# # Analysis

# In[ ]:


n_uniq_grapheme = df_this_train.grapheme_id.nunique()
n_uniq_root = df_this_train.grapheme_root.nunique()
n_uniq_vowel = df_this_train.vowel_diacritic.nunique()
n_uniq_diacritic = df_this_train.consonant_diacritic.nunique()

print(f'We have only {n_uniq_grapheme} grapheme in training data, but all {n_uniq_root} roots, {n_uniq_vowel} vowels, {n_uniq_diacritic} diacritics are remains')


# In[ ]:


n_uniq_grapheme = df_this_valid.grapheme_id.nunique()
n_uniq_root = df_this_valid.grapheme_root.nunique()
n_uniq_vowel = df_this_valid.vowel_diacritic.nunique()
n_uniq_diacritic = df_this_valid.consonant_diacritic.nunique()

print(f'While we have all {n_uniq_grapheme} grapheme in validation, and all {n_uniq_root} roots, {n_uniq_vowel} vowels, {n_uniq_diacritic} diacritics as well')


# In[ ]:


# We have 7578 unseen samples in validation set, which is approximately 16.4%
df_this_valid['unseen'].value_counts()


# In[ ]:




