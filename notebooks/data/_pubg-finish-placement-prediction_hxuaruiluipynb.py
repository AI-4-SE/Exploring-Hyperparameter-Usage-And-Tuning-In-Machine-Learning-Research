#!/usr/bin/env python
# coding: utf-8

# # PUBG Finish Placement Prediction (Kernels Only)
# 
# ### In this notebook, I will use LightGBM to predict the final placement where the winner win this match with 4446966 different data. I will split it into four different patterns, which defined by the match patterns, solo, duo, squad and other custom matches. And I will combine them together in order to obtain the final result. As usual, I will use some necessary packages, like pandas to deal with dataset, numpy to process data, seaborn and marplotlib to draw graph, and most important package is lightbgm which is the model I used to predict.

# Firstly, import the packages

# In[ ]:


# if need please install them

#pip install pandas
#pip install numpy
#pip install seaborn
#pip install lightgbm
#pip install matplotlib


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
import matplotlib 
from matplotlib import pyplot as plt


# And then read the data, meanwhile learn the basic information about the dataset 

# In[ ]:


df_train = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')
df_test = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')
df_train


# In[ ]:


df_test


# In[ ]:


df_train.info()


# Below are features which contain in this dataset:
# * DBNOs - Number of enemy players knocked.
# * assists - Number of enemy players this player damaged that were killed by teammates.
# * boosts - Number of boost items used.
# * damageDealt - Total damage dealt. Note: Self inflicted damage is subtracted.
# * headshotKills - Number of enemy players killed with headshots.
# * heals - Number of healing items used.
# * Id - Player’s Id
# * killPlace - Ranking in match of number of enemy players killed.
# * killPoints - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
# * killStreaks - Max number of enemy players killed in a short amount of time.
# * kills - Number of enemy players killed.
# * longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
# * matchDuration - Duration of match in seconds.
# * matchId - ID to identify match. There are no matches that are in both the training and testing set.
# * matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.
# * rankPoints - Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.
# * revives - Number of times this player revived teammates.
# * rideDistance - Total distance traveled in vehicles measured in meters.
# * roadKills - Number of kills while in a vehicle.
# * swimDistance - Total distance traveled by swimming measured in meters.
# * teamKills - Number of times this player killed a teammate.
# * vehicleDestroys - Number of vehicles destroyed.
# * walkDistance - Total distance traveled on foot measured in meters.
# * weaponsAcquired - Number of weapons picked up.
# * winPoints - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.
# * groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
# * numGroups - Number of groups we have data for in the match.
# * maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
# * winPlacePerc - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.

# In[ ]:


df_train.describe()


# List all the independent values which are useful to predict

# In[ ]:


train_columns = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 
                  'heals', 'killPlace', 'killPoints', 'kills', 'killStreaks', 
                  'longestKill', 'maxPlace', 'numGroups', 'revives','rideDistance', 
                  'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 
                  'weaponsAcquired', 'winPoints']


# Write a function to iterate through all the columns of a dataframe and modify the data type in order to reduce memory usage.        

# In[ ]:


def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df


# In[ ]:


df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)


# Draw a heatmap to see each deatures' relationship

# In[ ]:


correlation = df_train.corr()
plt.figure(figsize=(25,25))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')
plt.title('Correlation between different fearures')


# In order to verify the number of data and see how many different matchs in thedataset, draw a histgraph to see it

# In[ ]:


def show_countplot(column):
    plt.figure(figsize=(20,4))
    sns.countplot(data=df_train, x=column).set_title(column)
    plt.show()


# In[ ]:


show_countplot('matchType')


# ## Solo Pattern
# 
# In the first pattern, I will analysis matchs whose pattern are solo, solo fpp, normal solo fpp, and normal solo.
# 
# Let me get those data fitting the patterns

# In[ ]:


train_solo = df_train.loc[df_train["matchType"] == "solo" ]
train_solo = train_solo.append( df_train.loc[df_train["matchType"] == "solo-fpp" ])
train_solo = train_solo.append( df_train.loc[df_train["matchType"] == "normal-solo-fpp" ])
train_solo = train_solo.append( df_train.loc[df_train["matchType"] == "normal-solo" ])
train_solo


# Than I define a function to split the train data to help the model better fit

# In[ ]:


def split_train_val(data, fraction):
    matchIds = data['matchId'].unique().reshape([-1])
    train_size = int(len(matchIds)*fraction)
    
    random_idx = np.random.RandomState(seed=2).permutation(len(matchIds))
    train_matchIds = matchIds[random_idx[:train_size]]
    val_matchIds = matchIds[random_idx[train_size:]]
    
    data_train = data.loc[data['matchId'].isin(train_matchIds)]
    data_val = data.loc[data['matchId'].isin(val_matchIds)]
    return data_train, data_val


# In[ ]:


x_train, x_train_test = split_train_val(train_solo, 0.91)


# Use LightGBM Model to do prediction

# In[ ]:


params = {
        "objective" : "regression", 
        "metric" : "mae", 
        "num_leaves" : 149, 
        "learning_rate" : 0.03, 
        "bagging_fraction" : 0.9,
        "bagging_seed" : 0, 
        "num_threads" : 4,
        "colsample_bytree" : 0.5,
        'min_data_in_leaf':1900, 
        'min_split_gain':0.00011,
        'lambda_l2':9
}
train_set = lgb.Dataset(x_train[train_columns], label=x_train['winPlacePerc'])
valid_set = lgb.Dataset(x_train_test[train_columns], label=x_train_test['winPlacePerc'])


# In[ ]:


model_solo = lgb.train(  params, 
                    train_set = train_set,
                    num_boost_round=9400,
                    early_stopping_rounds=200,
                    verbose_eval=100, 
                    valid_sets=[train_set,valid_set]
                  )


# In[ ]:


test_solo = df_test.loc[df_test["matchType"] == "solo" ]
test_solo = test_solo.append( df_test.loc[df_test["matchType"] == "solo-fpp" ])
test_solo = test_solo.append( df_test.loc[df_test["matchType"] == "normal-solo-fpp" ])
test_solo = test_solo.append( df_test.loc[df_test["matchType"] == "normal-solo" ])


# In[ ]:


test_solo


# The final result is below

# In[ ]:


pre_solo = model_solo.predict(test_solo[train_columns])
test_solo['winPlacePerc'] = pre_solo
test_solo


# In order to know the importance of each dependent deatures, I draw a graph to compare them.

# In[ ]:


plt.figure(figsize=(20,8))
lgb.plot_importance(model_solo, max_num_features=22)
plt.title("Featurertances with Solo pattern")
plt.show()


# As we can see, The relationship between each dependent value and my target is basically consistent with its importance.

# # Duo Pattern
# 
# In the second pattern, I will analysis the matchs whose patterns are duo, duo fpp, normal duo fpp, normal duo.
# 
# Then do the same thing above

# In[ ]:


train_duo = df_train.loc[df_train["matchType"] == "duo" ]
train_duo = train_duo.append( df_train.loc[df_train["matchType"] == "duo-fpp" ])
train_duo = train_duo.append( df_train.loc[df_train["matchType"] == "normal-duo-fpp" ])
train_duo = train_duo.append( df_train.loc[df_train["matchType"] == "normal-duo" ])
train_duo


# In[ ]:


x_train, x_train_test = split_train_val(train_duo, 0.91)
train_set = lgb.Dataset(x_train[train_columns], label=x_train['winPlacePerc'])
valid_set  = lgb.Dataset(x_train_test[train_columns], label=x_train_test['winPlacePerc'])


# In[ ]:


model_duo = lgb.train( params, 
                    train_set = train_set,
                    num_boost_round=9400,
                    early_stopping_rounds=200,
                    verbose_eval=100, 
                    valid_sets=[train_set,valid_set]
                  )


# In[ ]:


test_duo = df_test.loc[df_test["matchType"] == "duo" ]
test_duo = test_duo.append( df_test.loc[df_test["matchType"] == "duo-fpp" ])
test_duo = test_duo.append( df_test.loc[df_test["matchType"] == "normal-duo-fpp" ])
test_duo = test_duo.append( df_test.loc[df_test["matchType"] == "normal-duo" ])


# In[ ]:


pre_duo = model_duo.predict(test_duo[train_columns])
test_duo['winPlacePerc'] = pre_duo
test_duo


# In[ ]:


plt.figure(figsize=(20,8))
lgb.plot_importance(model_duo, max_num_features=30)
plt.title("Featurertances with Duo pattern")
plt.show()


# # Squad Pattern 
# 
# In the 3rd parts, I will analysis the matchs whose patterns are squad, squad fpp, normal squad, normal suqad fpp.
# 
# And also do the sam thing above

# In[ ]:


train_squad = df_train.loc[df_train["matchType"] == "squad" ]
train_squad = train_squad.append(df_train.loc[df_train["matchType"] == "squad-fpp" ])
train_squad = train_squad.append( df_train.loc[df_train["matchType"] == "normal-squad-fpp" ])
train_squad = train_squad.append( df_train.loc[df_train["matchType"] == "normal-squad" ])
train_squad


# In[ ]:


x_train, x_train_test = split_train_val(train_squad, 0.91)
train_set = lgb.Dataset(x_train[train_columns], label=x_train['winPlacePerc'])
valid_set  = lgb.Dataset(x_train_test[train_columns], label=x_train_test['winPlacePerc'])


# In[ ]:


model_squad = lgb.train( params, 
                    train_set = train_set,
                    num_boost_round=9400,
                    early_stopping_rounds=200,
                    verbose_eval=100, 
                    valid_sets=[train_set,valid_set]
                  )


# In[ ]:


test_squad = df_test.loc[df_test["matchType"] == "squad" ]
test_squad = test_squad.append( df_test.loc[df_test["matchType"] == "squad-fpp" ])
test_squad = test_squad.append( df_test.loc[df_test["matchType"] == "normal-squad-fpp" ])
test_squad = test_squad.append( df_test.loc[df_test["matchType"] == "normal-squad" ])


# In[ ]:


pre_squad = model_squad.predict(test_squad[train_columns])
test_squad['winPlacePerc'] = pre_squad
test_squad


# In[ ]:


plt.figure(figsize=(20,8))
lgb.plot_importance(model_squad, max_num_features=22)
plt.title("Featurertances with Squad pattern")
plt.show()


# # Others Pattern
# 
# In this part, I will analysis rest of matchs, which are custome matchs opened by third-parties.

# In[ ]:


train_others = df_train.loc[df_train["matchType"] == "flaretpp" ]
train_others = train_others.append(df_train.loc[df_train["matchType"] == "crashfpp" ])
train_others = train_others.append( df_train.loc[df_train["matchType"] == "flarefpp" ])
train_others = train_others.append( df_train.loc[df_train["matchType"] == "crashtpp" ])
train_others


# In[ ]:


x_train, x_train_test = split_train_val(train_others, 0.91)
train_set = lgb.Dataset(x_train[train_columns], label=x_train['winPlacePerc'])
valid_set  = lgb.Dataset(x_train_test[train_columns], label=x_train_test['winPlacePerc'])


# In[ ]:


model_others = lgb.train( params, 
                    train_set = train_set,
                    num_boost_round=9400,
                    early_stopping_rounds=200,
                    verbose_eval=100, 
                    valid_sets=[train_set,valid_set]
                  )


# In[ ]:


test_others = df_test.loc[df_test["matchType"] == "flaretpp" ]
test_others = test_others.append(df_test.loc[df_test["matchType"] == "crashfpp" ])
test_others = test_others.append( df_test.loc[df_test["matchType"] == "flarefpp" ])
test_others = test_others.append( df_test.loc[df_test["matchType"] == "crashtpp" ])


# In[ ]:


pre_others = model_others.predict(test_others[train_columns])
test_others['winPlacePerc'] = pre_others
test_others


# In[ ]:


plt.figure(figsize=(20,8))
lgb.plot_importance(model_others, max_num_features=22)
plt.title("Featurertances with Others pattern")
plt.show()


# ## Finally, I need to combine them together, and see the final result 

# In[ ]:


test_result = test_solo.append(test_duo).append(test_squad).append(test_others)


# In[ ]:


test_result = test_result.sort_index(ascending=True)


# In[ ]:


result = test_result[['Id','winPlacePerc']]
result


# In[ ]:


import os
os.getcwd()


# In[ ]:


result.to_csv('submission.csv',index=False)


# # conlusion 
# As we see, Whichever matchs players ateend, walk distance, kill place and damage dealt are very important. In light of the importance of tehm, player should do more exercises focus on them. Meanwhile, The official also should pay more attention to supervise those players who have lower kill palce but win the match for example. Furthermore, if I do more specifically, I will analysis the strength of groups in the duo and squad pattern, which maybe also has a higher importance in this model.  

# # Citation 
# - https://www.cnblogs.com/datasnail/p/9675410.html
# - https://www.jianshu.com/p/a7c71a4a3024
# - https://www.kaggle.com/kamalchhirang/5th-place-solution-0-0184-score
# - https://www.kaggle.com/amoeba3215/keras-nn-mlp
# - https://www.jianshu.com/p/2275f1fdd34f
# - https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/

# # License
# LICENSE:
# 
# MIT Liecnese
# 
# Copyright 2020(c) Huarui Lu
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
# 
# 

# In[ ]:




