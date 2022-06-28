#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# The purpose of this notebook is to gain deeper insights from football data by analyzing shot and pass statistics. <br>
# Only teams of Premier League, Ligue 1, Bundesliga, Serie A and La Liga will be analyzed. <br>
# 
# Auxiliary datasets and notebooks are listed below:
# * [2021-2022 Football Player Stats](https://www.kaggle.com/datasets/vivovinco/20212022-football-player-stats)
# * [2021-2022 Football Team Stats](https://www.kaggle.com/datasets/vivovinco/20212022-football-team-stats)
# 
# **If you're reading this, please upvote.**

# In[ ]:


get_ipython().system('pip install mplsoccer')
get_ipython().system('pip install highlight_text')

import pandas as pd
import matplotlib.pyplot as plt
from highlight_text import fig_text
import matplotlib as mpl
from mplsoccer.pitch import Pitch
import seaborn as sns

passes = pd.read_csv("../input/football-analytics/Passes.csv", delimiter=";")
shots = pd.read_csv("../input/football-analytics/Shots.csv", delimiter=";")


# # 2. Passes Analysis

# In[ ]:


#convert the data to match the mplsoccer statsbomb pitch
#to see how to create the pitch, watch the video here: https://www.youtube.com/watch?v=55k1mCRyd2k
passes['x'] = passes['x']*1.2
passes['y'] = passes['y']*.8
passes['endX'] = passes['endX']*1.2
passes['endY'] = passes['endY']*.8

passes


# In[ ]:


fig ,ax = plt.subplots(figsize=(13.5,8))
fig.set_facecolor('#22312b')
ax.patch.set_facecolor('#22312b')

#this is how we create the pitch
pitch = Pitch(pitch_type='statsbomb', orientation='horizontal',
              pitch_color='grass', line_color='white', figsize=(13, 8), stripe=True,
              constrained_layout=False, tight_layout=True)

#Draw the pitch on the ax figure as well as invert the axis for this specific pitch
pitch.draw(ax=ax)
plt.gca().invert_yaxis()

#Create the heatmap
kde = sns.kdeplot(
        passes['x'],
        passes['y'],
        shade = True,
        shade_lowest=False,
        alpha=.5,
        n_levels=10,
        cmap = 'magma'
)


#use a for loop to plot each pass
for x in range(len(passes['x'])):
    if passes['outcome'][x] == 'Successful':
        plt.plot((passes['x'][x],passes['endX'][x]),(passes['y'][x],passes['endY'][x]),color='green')
        plt.scatter(passes['x'][x],passes['y'][x],color='green')
    if passes['outcome'][x] == 'Unsuccessful':
        plt.plot((passes['x'][x],passes['endX'][x]),(passes['y'][x],passes['endY'][x]),color='red')
        plt.scatter(passes['x'][x],passes['y'][x],color='red')
        
plt.xlim(0,120)
plt.ylim(0,80)

plt.title('Messi Pass Map vs Real Betis',color='white',size=20)


# # 3. Shots Analysis

# In[ ]:


shots


# In[ ]:


fig, ax = plt.subplots(figsize=(13,8.5))
fig.set_facecolor('#22312b')
ax.patch.set_facecolor('#22312b')

#The statsbomb pitch from mplsoccer
pitch = Pitch(pitch_type='statsbomb', orientation='vertical',
              pitch_color='grass', line_color='white', stripe=True, figsize=(13, 8),
              constrained_layout=False, tight_layout=True, view='half')

pitch.draw(ax=ax)

plt.gca().invert_yaxis()

#plot the points, you can use a for loop to plot the different outcomes if you want
plt.scatter(shots.x, shots.y, c='red', s=100, alpha=.8)

plt.title('Barcelona Shot Chart vs Juventus', fontsize=24, color='w')

total_shots = len(shots)

fig_text(s=f'Total Shots: {total_shots}',
        x=.27, y =.67, fontsize=14,fontfamily='Andale Mono',color='w')
fig_text(s=f'xG: .85',
        x=.49, y =.67, fontsize=14,fontfamily='Andale Mono',color='w')
fig_text(s=f'Goals: 0',
        x=.68, y =.67, fontsize=14,fontfamily='Andale Mono',color='w')


# Well, obviously scatter plot does not plot the shot locations...
