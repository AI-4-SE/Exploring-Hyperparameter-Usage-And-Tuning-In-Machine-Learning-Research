#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Set up
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# # What do i want to analyze?
# ###### Only European League will be considered
# > * **Top 5 highst goal score (2016 ~ 2020)**
#     * viz - line chart: To see the performance of the players who are in top 5 Goal Score
#     over the years
#     
# > * **Who has the highst xG value (2016~2020)**
#     * viz - bar chart: comparing with other 5 most top players
#     
#     
# 

# 

# In[ ]:


df= pd.read_csv("../input/top-football-leagues-scorers/Data.csv")


# In[ ]:


df.describe()


# In[ ]:


df.head()


# In[ ]:


# Who has the most goals ever
df['Goals'].max()
df[["Player Names", "Goals", "Year"]][df["Goals"] == 37]
# messi 37 Goals


# In[ ]:


# Top 5 Goal Score (2016 ~ 2020)
df.groupby('Goals')['Goals'].count()
# 37, 36, 34, 33, 31 Goals are the highst ever 

df[["Player Names", "Goals", "Year", "Club", "League"]][df['Goals'] == 36]
# Immobile 36 Goals
# except Messi(already top 1 with 37 goals) and Vela (USA League will be not considered..)

df[["Player Names", "Goals", "Year"]][df['Goals'] == 34]
# Lewandowski 34 Goals

df[["Player Names", "Goals", "Year"]][df['Goals'] == 33]
# Mbappe 33 Goals

df[["Player Names", "Goals", "Year", "Club"]][df['Goals'] == 31]
# Aubameyang, Zlatan, Ronaldo 31 Goals

# top1 Lionel Messi
# top2 Ciro Immobile
# top3 Robert Lewandowski
# top4 Kylian Mbappe-Lottin
# top5 Pierre-Emerick Aubameyang, Zlatan Ibrahimovic, Cristiano Ronaldo


# In[ ]:


messi = df[df.loc[:, "Player Names"] == "Lionel Messi"]
sns.lineplot(data=df, x=messi["Year"], y=messi["Goals"])
# Legend never dies? Legend can die


# In[ ]:


# Top 5 highst goal score (2016 ~ 2020)
top5_goals_players_list = ["Lionel Messi", "Ciro Immobile", "Robert Lewandowski", "Kylian Mbappe-Lottin", "Pierre-Emerick Aubameyang", "Zlatan Ibrahimovic", "Cristiano Ronaldo"]
top5_goals_players = df[df["Player Names"].isin(top5_goals_players_list)]

plt.figure(figsize=(18, 12))
sns.barplot(data=df, x=top5_goals_players["Year"], y=top5_goals_players["Goals"], hue=top5_goals_players["Player Names"])


# #### The latest best player is lewandoski.
# #### Messi is dying but ronaldo is being better than messi.
# #### And it seems like zlatan got injured for 2 years (2017, 2018).
# #### There is such a huge decrease from 2019 to 2020,(Because of COVID-19?)

# In[ ]:


# Who has the highst xG value (2016~2020)
df.sort_values(by='xG', ascending=False).head(10)
top5_xG_list = ["Lieonel Messi", "Kylian Mbappe-Lottin", "Robert Lewandowski", "Edin Dzeko", "Cristiano Ronaldo"]
top5_xG = df[df.loc[:, "Player Names"].isin(top5_xG_list)]

plt.figure(figsize=(13, 8))
sns.barplot(data=df, x=top5_xG["Year"], y=top5_xG["xG"], hue=top5_xG["Player Names"])
plt.ylabel("xG Value")


# ### It makes sense that lewandowski has the highst xG value at 2020
