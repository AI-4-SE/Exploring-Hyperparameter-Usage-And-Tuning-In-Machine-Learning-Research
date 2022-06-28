#!/usr/bin/env python
# coding: utf-8

# # This notebook provides:
# - Sampling from the training set
# - Generation of perfect and naive submission files
# - Scoring Function (Mean Jaccard)
# 
# Already generated samples between 100k - 600k are available below:
# 
# https://www.kaggle.com/datasets/sorkun/flm-train-samples
# 

# In[ ]:


import numpy as np 
import pandas as pd 
from tqdm.notebook import tqdm


# In[ ]:


#Creates submission file from train and match dfs
def create_submission(test_df, match_df):
    unique_ids=test_df["id"].unique()
    matches=[]
    for id_1 in tqdm(unique_ids,total=len(unique_ids)):
        match=id_1
        matched = match_df[match_df["id_1"]==id_1]["id_2"]
        for id_2 in matched:
            match = match + " " + id_2
#         print (match)
        matches.append(match)
    sub = pd.DataFrame(unique_ids,columns=["id"])
    sub["matches"]=matches
    return sub


# In[ ]:


#Creates naive submission using only same IDs
def create_naive_submission(test_df):
    unique_ids=test_df["id"].unique()
    sub = pd.DataFrame(unique_ids,columns=["id"])
    sub["matches"]=unique_ids
    return sub


# In[ ]:


# Scoring: Calculates mean Jaccard between two submissions
def evaluate_jaccard(df1, df2):
    if(len(df1)!=len(df2)):
        print("Error: Sizes are not equal")
        return None
    else:
        df1=df1.sort_values(by=['id'])
        df2=df2.sort_values(by=['id'])
        jaccard_list=[]
        for i in range(len(df1)):
            list1=df1.iloc[i]["matches"].split(" ")
            list2=df2.iloc[i]["matches"].split(" ")
            intersection = len(list(set(list1).intersection(list2)))
            union = (len(set(list1)) + len(set(list2))) - intersection
            jacc_sim=float(intersection) / union
            jaccard_list.append(jacc_sim)
        return np.mean(jaccard_list)


# In[ ]:


train_df = pd.read_csv("/kaggle/input/foursquare-location-matching/train.csv")
train_df.head()
len(train_df)


# In[ ]:


# Creates all possible matches, credit: https://www.kaggle.com/code/sudalairajkumar/flm-additional-match-pairs-data
match_df = pd.merge(train_df, train_df, on="point_of_interest", suffixes=('_1', '_2'))
match_df = match_df[match_df["id_1"]!=match_df["id_2"]]
# match_df = match_df.drop(["point_of_interest"], axis=1)
match_df["match"] = True
match_df.head()
len(match_df)


# In[ ]:


#Create SAMPLE data from 100k Training data (14068 matches)
train_df_100k=train_df.sample(n=100000,random_state=0)
match_df_100k = pd.merge(train_df_100k, train_df_100k, on="point_of_interest", suffixes=('_1', '_2'))
match_df_100k = match_df_100k[match_df_100k["id_1"]!=match_df_100k["id_2"]]
# match_df = match_df.drop(["point_of_interest"], axis=1)
match_df_100k["match"] = True
match_df_100k.head()
len(match_df_100k)


# In[ ]:


#Create submissions and evaluate score
sub_100k_naive=create_naive_submission(train_df_100k)
sub_100k=create_submission(train_df_100k,match_df_100k)
evaluate_jaccard(sub_100k_naive,sub_100k)


# In[ ]:


#Write CSV
train_df_100k.to_csv("train_df_100k.csv",index=False)
match_df_100k.to_csv("match_df_100k.csv",index=False)
sub_100k.to_csv("sub_100k.csv",index=False)
sub_100k_naive.to_csv("sub_100k_naive.csv",index=False)

