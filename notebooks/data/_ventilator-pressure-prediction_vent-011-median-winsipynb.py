#!/usr/bin/env python
# coding: utf-8

# # Median wins against weighted mean
# 
# In this competition, we have seen public notebooks which compute a weighted arithmetic mean of other submissions. Here we use the same input as the currently top-scoring public notebook, but compute the median rather than the mean (and get a better score).
# 
# In a competition which is scored by mean absolute error (MAE), optimizing weights is a waste of time: The median is better and doesn't need any weights.
# 
# See https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/280573 for a discussion of the topic.
# 
# <font size="1">There is one exception to the rule: If you have only two inputs, a weighted arithmetic mean give a better result than the median.<font>

# In[ ]:


import pandas as pd
import numpy as np

files = ['../input/gb-vpp-pulp-fiction/median_submission.csv',
         '../input/basic-ensemble-of-public-notebooks/submission_median.csv',
         '../input/gaps-features-tf-lstm-resnet-like-ff/sub.csv']

sub = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')
sub['pressure'] = np.median(np.concatenate([pd.read_csv(f)['pressure'].values.reshape(-1, 1) for f in files], axis=1), axis=1)
sub.to_csv('submission.csv', index=False)
sub.head(5)


# In[ ]:




