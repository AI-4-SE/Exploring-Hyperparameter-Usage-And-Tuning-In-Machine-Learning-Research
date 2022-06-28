#!/usr/bin/env python
# coding: utf-8

# # Descrption:
# 
# 1. what i am goin to do?
#     Try to understand the datasets and what analysis we can bring out of datasets
#     in the sense try to have a question and bring analyze each with set of setps to fiollow:
# 2. what questions to answer?
# 3. what analysis to apply to each question?

# ## 1. Importing packages

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# Below will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## 2. Read the datasets from the Database for bundesliga football league.

# In[ ]:


df = pd.read_json('/kaggle/input/german-football-scores/ranking_full_liga_3.json')


# ## 3. Display first 10 rows 

# In[ ]:


df.head(10)


# ### Data Description:
# 
# - `Season` : score of which year 
# - `md` : 
# - `position` : positional point gained
# - `team` : which team team played
# - `played` : number of matches or games played by a team
# - `won` : number of matches won
# - `draw` : number of times a team has finished a match with an even score or tie
# - `lost` : number of matches lost
# - `gf` : goal for team (Goals Scored)
# - `ga` : goal against (i.e., number of goals conceded by a team).
# - `gd` : goal difference (i.e., difference between GF and GA, and sometimes denoted by +/-)
# - `points` : Total scores (i.e., total number of points earned by a team after playing a certain number of games)., which is callculated using below formula:
#     `points = (3 * won) + draw` because 3 points are awarded for a win and 1 for a draw.

# ## What Data type is made up for each column 

# In[ ]:


df.dtypes


# ## 4. Statstical Analysis over the datasets

# In[ ]:


df.describe()


# ## Lets split the Season year from 20xx/20xx to 20xx
# - for understanding purpose.

# In[ ]:


season_li = []
def split_season(season):
    for i in df.season:
        season_li.append(i.split('/')[0])
split_season(df['season'])

df['Start_Season'] = season_li
df


# In[ ]:


df.Start_Season = df.Start_Season.astype(int)


# In[ ]:


df.dtypes


# ## Analyzing Top 10 teams during 2019 season

# In[ ]:


df[(df.position <= 10) & (df.Start_Season == 2019)].describe()


# We can make following conclusions:
# 1. Teams within top 10 positions has played: 
#     - On an average 19 to 20 matches with 8 matches won, with draw around 6, and lost 5 matches during 2019.
#     - On an average have scored 33 goals for team and 26 goal against.
# 2. Teams have scored with highest goals for team of 76 and 60 goal against.

# ## Teams count which secured 1st position from 2008 to 2020

# In[ ]:


top10 = df[df.position == 1]
top10.team.value_counts()


# - Team `MSV Duisburg` with 52 holds highest count and 
# - 2nd highest`SC Paderborn 07` with 40.
# irrespective of fall and rise of position over the year.

# ## Cleaning the dataset
# - removing `md` and `season` column

# In[ ]:


df.drop(columns=['season','md'],axis=1,inplace=True)
df


# In[ ]:


df.isnull().sum()


# ## Analyzing Duisburg team's point during 2019

# In[ ]:


df[(df.Start_Season == 2019) & (df.team=='MSV Duisburg')]


# ## Getting correlation between variables 

# In[ ]:


df.corr()


# In[ ]:




