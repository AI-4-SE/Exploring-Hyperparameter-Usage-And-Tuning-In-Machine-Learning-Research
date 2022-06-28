#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Let me import all the required library
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from itertools import takewhile
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.style as style
style.use('fivethirtyeight')


# ## Let's understand the data first

# In[ ]:


df = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/train.csv")


# In[ ]:


df.head()


# I can see here 5 columns: id, anchor, target, context and score. As fas as I know the objective of this competition is to predict similarity score between anchor and target. In the first row of the train dataframe I can see similarity score between anchor: 'abatement' and target: 'abatement of pollution' is 0.50. It seems like the word abatement is of great importance for similarity score. But it is not the case, look at the 4th row where target is 'eliminating process', here is no word abatement but still the similarity score is 0.50. Now I know that my general logic to find similarity score  is not going to work here.

# In[ ]:


# Let me check how many unique anchor are there and also unique target
len(df.anchor.unique()),len(df.target.unique())


#  There are 733 unique and anchor and 29340 unique target. This means each anchor corresponds to lots of targets. I came up with new beginner idea. What if I could make a separate dataframe of targets and score for each anchor. But wait I don't know about the score yet. 

# In[ ]:


df.score.unique()


# In[ ]:


len(df.anchor.unique()),len(df.target.unique()),len(df.score.unique())


# Now I know about the score, each anchor and target similarity is classified into one of the 5 scores(1,0.75,0.5,0.25 and 0.0). where 1 being the highest similarity score and 0 being the lowest. Ok now what? I was talking about creating separate dataframe of targets and scores for each anchor. There are 733 anchor. Is it reasonable to create 733 separate dataframe for each anchor? I don't think so. I came up with an idea. lets creat a function where we can access the dataframe of any anchor we want. for ex df_score(anchor_no). as we know our anchor_no range would be from 0 to 732. Do you know our df_score is going to have five columns; Can you guess what are those five columns??

# ## Let's define a function to generate a targets vs scores dataframe for each anchor

# In[ ]:


def df_score(anc_no): # range of anc_no is from 0 to 732
    anchor_list = df['anchor'].value_counts().index# list of 733 anchors
    train_anchor = df[df['anchor'] == anchor_list[anc_no]] # new dataframe for the input anchor
    
    score_1 = train_anchor[train_anchor['score']==1] # dataframe just for score 1
    score_2 = train_anchor[train_anchor['score']==0.75]# dataframe just for score 0.75
    score_3 = train_anchor[train_anchor['score']==0.5]# dataframe just for score 0.5
    score_4 = train_anchor[train_anchor['score']==0.25]# dataframe just for score 0.25
    score_5 = train_anchor[train_anchor['score']==0]# dataframe just for score 0
    
    #let's concat all the scores and give them a new columns names 'score_100,score_75...'
    df_score = pd.concat([score_1['target'], score_2['target'],score_3['target'],
                 score_4['target'],score_5['target']], axis=1,
                keys=['score_100', 'score_75','score_50','score_25','score_0'])
    
    #We have to sort the dataframe because the number of targets of each score columns 
    #are different,therefore there are some nan values,after sorting rows with nan values
    #are sent down.
    df_score['score_100'] = df_score['score_100'].sort_values(ascending=False).values
    df_score['score_75'] = df_score['score_75'].sort_values(ascending=False).values
    df_score['score_50'] = df_score['score_50'].sort_values(ascending=False).values
    df_score['score_25'] = df_score['score_25'].sort_values(ascending=False).values
    df_score['score_0'] = df_score['score_0'].sort_values(ascending=False).values    

    index = df_score.index
    index.name = 'Anchor: ' + anchor_list[anc_no] + ' VS Target Scores'
    
    return df_score


# ## Targets VS scores dataframe of each anchor

# Note: The input of function df_score() is index no. from anchor_list. You can access dataframe of any anchor by their corresponding index no. range is from 0-732.

# In[ ]:


anchor_list = df['anchor'].value_counts().index
anchor_list[:5]


# In[ ]:


df_score(0).head() # I can access any anchor by their index no. from anchor_list,


# Here we go. We got it. Thats what I want. Now I can see the relationship between Anchor ,targets and scores more clearly. Remember the anchor with highest counts are arranged first. when we enter df_score(0), we got anchor:'component composite coating' because it's value counts is the  highest. try df_score(732).

# ## Let's define a function to create Targets Vs Scores word cloud

# In[ ]:


def wc_target(anc_no):
    anchor_list = df['anchor'].value_counts().index
    train_anchor = df[df['anchor'] == anchor_list[anc_no]]
    
    score_1 = train_anchor[train_anchor['score']==1]
    score_2 = train_anchor[train_anchor['score']==0.75]
    score_3 = train_anchor[train_anchor['score']==0.5]
    score_4 = train_anchor[train_anchor['score']==0.25]
    score_5 = train_anchor[train_anchor['score']==0]
    
    
    df_score = pd.concat([score_1['target'], score_2['target'],score_3['target'],
                 score_4['target'],score_5['target']], axis=1,
                keys=['score_100', 'score_75','score_50','score_25','score_0'])
    
    df_score['score_100'] = df_score['score_100'].sort_values(ascending=False).values
    df_score['score_75'] = df_score['score_75'].sort_values(ascending=False).values
    df_score['score_50'] = df_score['score_50'].sort_values(ascending=False).values
    df_score['score_25'] = df_score['score_25'].sort_values(ascending=False).values
    df_score['score_0'] = df_score['score_0'].sort_values(ascending=False).values
    
    # here I am using takewhile from itertools that act as a while loop, 
    # I am using takewhile because there are lots of nan entries in df_score
    # using takewhile end the loop when nan entries started.
    wc_score100 = WordCloud(width = 800, height = 400, background_color="white").generate(" ".join(score for score in list(takewhile(lambda x:type(x)==str,list(df_score.score_100)))))
    wc_score75 = WordCloud(width = 800, height = 400, background_color="white").generate(" ".join(score for score in list(takewhile(lambda x:type(x)==str,list(df_score.score_75)))))
    wc_score50 = WordCloud(width = 800, height = 400, background_color="white").generate(" ".join(score for score in list(takewhile(lambda x:type(x)==str,list(df_score.score_50)))))
    wc_score25 = WordCloud(width = 800, height = 400, background_color="white").generate(" ".join(score for score in list(takewhile(lambda x:type(x)==str,list(df_score.score_25)))))
    wc_score0 = WordCloud(width = 800, height = 400, background_color="white").generate(" ".join(score for score in list(takewhile(lambda x:type(x)==str,list(df_score.score_0)))))
    
    fig = plt.figure(figsize = (40,40))
    plt.title('Anchor: ' + anchor_list[anc_no],fontsize=50)
    plt.axis('off')
    ims = [[wc_score100, "Target: Score 1.0"],
           [wc_score75, "Target: Score 0.75"],
           [wc_score50, "Target: Score 0.5"],
           [wc_score25, "Target: Score 0.25"],
           [wc_score0, "Target: Score 0"]]
    
    for a, b in enumerate(ims):
        fig.add_subplot(3,2, a+1)
        plt.imshow(b[0], interpolation='bilinear')
        plt.title(b[1], fontsize = 40)
        plt.axis("off")
   
    return plt.show()


# ## Targets Vs scores word cloud

# The most frequent occuring words are shown larger in the word cloud as shown below. component,coating are common words in all the scores

# In[ ]:


wc_target(0)


# I am a beginner, Please excuse me if you encounter any incovenience understanding the code.

# In[ ]:


wc_target(1)


# In[ ]:




