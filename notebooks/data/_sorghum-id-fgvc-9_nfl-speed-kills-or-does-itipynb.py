#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# As an introduction to this analysis it was inspired after reading this piece by PFF. https://www.pff.com/news/nfl-fast-wide-receivers-nfl-offenses
# 
# However several aspects caught my eye in the analysis conducted by PFF.
# * The use of average 40 time for the top 3 WR
# * Average depth of defensive backs for the play
#     
# These two aspects I thought could be improved by using the NFL Data Bowl tracking data.
# 
# **Average time for WR**
# * It seems unfair to use the average of the top three recievers when trying to calculate the impact of the fastest WRs in the league.
#     * Example:
#         * WR Group 1 40s (4.22, 4.57, 4.62) Avg 4.47
#         * WR Group 2 40s (4.38, 4.43, 4.52) Avg 4.44
#         * Group 1 has the much quicker WR who you likely paid premium draft capital for, however when averaged looks like they are in a slower group.
#     * Also are 40 times still valid once you consider player age and injuries. Can we use player speed from the tracking data?
# 
# **Average depth of defensive backs**
# * While a fair measure, the general aim of the super quick WRs is to streatch the defence.
#     * Due to this I will investigate the impact on the deepest Safeties in the formation, generally the FS and SS.

# # TL;DR
# 
# WR speed has minimal to no impact on the depth of safeties either pre or post snap.
# 
# This brings into question the value of using premium draft capital on WRs for speed with stretching the defence being the main aim.

# # Methodology
# 
# To try and remove as much bias from the results as possible we will add the following assumptions and restrictions to the data:
# * WR 40 times manaually extracted from Wikipedia or Google, pro day results are used if combine result is not avaliable
# * Will only include passing plays where the top three WR for the team are active in the play
# * Exclude 4th downs and passing downs over 10 yards
#     * To reduce situational variance where possible
# * Exclude passing plays in the Redzone (final 20 yards) and the first 10 yards. 
#     * Where DBs may be overly aggressive or limits on how far they can drop back
# * Exlcude passing plays from the final minue of the quarters. 
#     * Trying to avoid Hail Mary plays where the results will be heavily biased as DBs are generally start super deep
# * DB depth is measured at two time periods:
#     * Snap of the ball
#     * Time when a pass is thrown
# * WR tracking speed is measured in yards per second and is the median of their max speed for
#     * Go routes (consistent straight line WR route)
#     * Speed over the first 4 seconds to have allowed them to build up speed
#     * Median taken to exclude any potential tail anaomalies e.g. (running a slow go on a quick pass, or exlcude high speed when making a tackle / diving)

# # Initial Results
# 
# As we can see from the graphs below the impact from having a super fast WR has minimal impact on stretching the defence as neither safety depth changes at the snap of the ball, and safeties do not drop back significantly further.
# 
# The main outlier is Robert Foster for the Bills who had a blazing median top speed in game of 8.6 yards/ second who stretched the defence on average by 2 yards.
# 
# My initial takeaway:
# * Speed does not seem to be the solves everything solution. 
# * However maybe the WR are not being used all the time to stretch the defence. 
#     * What happens if I just limit it to plays where the fastest WR is on a GO route?

# ### Comparison of 40 time to Safety Depth at time of Ball Snap
# ![](https://i.imgur.com/tGW75NU.png?1)

# ### Comparison of WR Tracking Speed to Safety Depth at time of Ball Snap
# ![Speed to depth v1.png](attachment:4a314986-306e-4b43-919d-38d121f94740.png)

# ### Comparison of WR Tracking Speed to avg drop from Ball Snap to Pass Thrown
# ![Speed to drop v1.png](attachment:557ff0d2-8264-4251-9098-6aac9d265179.png)

# # Second Results - Only include plays where the fastest WR is on a GO route to stretch the defence
# 
# As we can see from the graphs below even when factoring in a WR on a GO route, speed seems to have little to no impact on stretching the field which is an interesting and suprising result. As a result I would be very cautious when spending premium draft capital on a WR solely due to a blazing 40 time.

# ### Comparison of 40 time to Safety Depth at time of Ball Snap
# ![40 to depth v2.png](attachment:1a6cb063-0ef7-43b8-9607-143ac91ba099.png)

# ### Comparison of WR Tracking Speed to Safety Depth at time of Ball Snap
# ![Speed to depth v2.png](attachment:4e3e88d5-7ca9-4829-8328-8863cb44b65f.png)

# ### Comparison of WR Tracking Speed to avg drop from Ball Snap to Pass Thrown
# ![Speed to drop v2.png](attachment:57a5437a-b5b5-4cff-beba-f6695781e919.png)

# In[ ]:





# # Appendix & Code

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import math

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

week1_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week1.csv')
week2_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week2.csv')
week3_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week3.csv')
week4_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week4.csv')
week5_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week5.csv')
week6_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week6.csv')
week7_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week7.csv')
week8_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week8.csv')
week9_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week9.csv')
week10_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week10.csv')
week11_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week11.csv')
week12_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week12.csv')
week13_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week13.csv')
week14_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week14.csv')
week15_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week15.csv')
week16_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/week16.csv')

frames = [week1_data,week2_data,week3_data,week4_data,week5_data,week6_data,week7_data,week8_data,week9_data,week10_data,week11_data,week12_data,week13_data,week14_data,week15_data,week16_data]
tracking_all = pd.concat(frames)
week1_data = tracking_all

game_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/games.csv')
player_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/players.csv')
plays_data = pd.read_csv('/Users/cgiess/Downloads/nfl-big-data-bowl-2021/plays.csv')



# only including tracking of WR and including the team
WR = week1_data[(week1_data['position'] == 'WR')&(week1_data['frameId'] == 1)]
WR_g = pd.merge(WR, game_data, how='inner', left_on=['gameId'], right_on=['gameId'])
WR_g['PlayerTeam'] = np.where(WR_g['team'] == 'away', WR_g['visitorTeamAbbr'], WR_g['homeTeamAbbr'])

# manually importing the comine 40 time from wikipedia or google
rec = ['Chad Williams', 'Christian Kirk', 'Larry Fitzgerald','Calvin Ridley','Julio Jones','Mohamed Sanu','John Brown','Michael Crabtree','Willie Snead','Kelvin Benjamin','Robert Foster','Zay Jones','Curtis Samuel','Devin Funchess','Jarius Wright','Allen Robinson','Anthony Miller','Taylor Gabriel','A.J. Green','John Ross','Tyler Boyd','Antonio Callaway','Jarvis Landry','Rashard Higgins','Amari Cooper','Cole Beasley','Michael Gallup','Courtland Sutton','DaeSean Hamilton','Emmanuel Sanders','Kenny Golladay','Marvin Jones','T.J. Jones','Davante Adams','Marquez Valdes-Scantling','Randall Cobb','DeAndre Hopkins','Demaryius Thomas','Will Fuller','Chester Rogers','Ryan Grant','T.Y. Hilton','Dede Westbrook','Donte Moncrief','Keelan Cole','Chris Conley','Sammy Watkins','Tyreek Hill','Brandin Cooks','Josh Reynolds','Robert Woods','Keenan Allen','Mike Williams','Tyrell Williams','Danny Amendola','DeVante Parker','Kenny Stills','Adam Thielen','Laquon Treadwell','Stefon Diggs','Chris Hogan','Josh Gordon','Julian Edelman','Michael Thomas','Ted Ginn',"Tre'Quan Smith",'Bennie Fowler','Odell Beckham','Sterling Shepard','Jermaine Kearse','Quincy Enunwa','Robby Anderson','Jordy Nelson','Seth Roberts','Alshon Jeffery','Jordan Matthews','Nelson Agholor','Antonio Brown','James Washington','JuJu Smith-Schuster','David Moore','Doug Baldwin','Tyler Lockett','Dante Pettis','Kendrick Bourne','Marquise Goodwin','Adam Humphries','Chris Godwin','Mike Evans','Corey Davis','Tajae Sharpe','Taywan Taylor','Josh Doctson','Maurice Harris','Paul Richardson']
time = [4.43, 4.47, 4.48, 4.43, 4.39, 4.67, 4.34, 4.54, 4.62, 4.61, 4.41, 4.45,4.31, 4.70, 4.42, 4.60, 4.50, 4.27, 4.50, 4.22, 4.58, 4.41, 4.77, 4.64,4.42, 4.49, 4.51, 4.54, 4.52, 4.41, 4.50, 4.46, 4.48, 4.56, 4.37, 4.46,4.57, 4.38, 4.32, 4.56, 4.64, 4.34, 4.39, 4.40, 4.59, 4.35, 4.43, 4.29,4.33, 4.52, 4.51, 4.56, 4.54, 4.43, 4.58, 4.45, 4.38, 4.49, 4.64, 4.46,4.50, 4.52, 4.52, 4.57, 4.37, 4.49, 4.52, 4.43, 4.48, 4.58, 4.45, 4.36,4.51, 4.44, 4.48, 4.46, 4.42, 4.48, 4.54, 4.54, 4.43, 4.48, 4.40,4.32, 4.68, 4.27, 4.53, 4.42, 4.53, 4.53, 4.55, 4.50, 4.50, 4.56, 4.40]
d = {'displayName': rec, 'time_40': time}
d_40 = pd.DataFrame(data=d)

# identifying the three most used WR for each team. 
WR_a = WR_g.groupby(['PlayerTeam','displayName']).agg({'frameId': 'sum'})
WR_a.reset_index( drop=False, inplace=True)
WR_a['rank'] = WR_a.groupby('PlayerTeam')['frameId'].rank(method="first", ascending=False)
WR_a2 = WR_a[WR_a['rank'].isin([1.0,2.0,3.0])]
WR_a2 = pd.merge(WR_a2, d_40, how='left', left_on=['displayName'], right_on=['displayName'])

#identifying the fasted 40 time for each team
T_40 = WR_a2.groupby('PlayerTeam').agg({'time_40': min})
T_40.reset_index( drop=False, inplace=True)

# i only want to include the impact for plays where all three main recievers are playing.
# Reasoning is it would be unfair to include snaps where the fastest reciever is not on the field
td = pd.merge(WR_g, WR_a2, how='inner', left_on=['PlayerTeam','displayName'], right_on=['PlayerTeam','displayName'])
td_w = td.groupby(['gameId','playId','PlayerTeam']).agg({'rank': 'sum'})
td_w.reset_index( drop=False, inplace=True)
td_3 = td_w[td_w['rank'] == 6.0]

# Trying to make sure the snaps used are as representive as possible so:
# excluding 4th downs
# snaps with over 10 to go 
# exlcuding potential hail marys in the final minute at the end of the quarters/ halves
con1 = (plays_data['down'].isin([1,2,3]))
con2 = (plays_data['gameClock'] >= '01:00:00')
con3 = (plays_data['yardsToGo'] <= 10)
p_r = plays_data[con1 & con2 & con3][['gameId','playId']]

td_3r = pd.merge(td_3, p_r, how='inner', left_on=['gameId','playId'], right_on=['gameId','playId'])

#standardising some of the tracking data so it is all from one viewpoint.
possession = pd.merge(plays_data, game_data, how='left', left_on=['gameId'], right_on=['gameId'])
possession = possession[['gameId','playId','possessionTeam','homeTeamAbbr','visitorTeamAbbr','yardlineNumber','yardlineSide']]
possession['possessionTeam_alt'] = "home"
possession.loc[possession.possessionTeam != possession.homeTeamAbbr, 'possessionTeam_alt'] = "away"

week1_data2 = pd.merge(week1_data, possession, how='left', left_on=['gameId','playId'], right_on=['gameId','playId'])
week1_data2['PlayerTeam'] = np.where(week1_data2['team'] == 'away', week1_data2['visitorTeamAbbr'], week1_data2['homeTeamAbbr'])
#week1_data.loc[week1_data['team'] == 'away', 'responsibility_role'] = 'visitorTeamAbbr'

W = week1_data2
W['Dir_rad'] = np.mod(90 - W.dir, 360) * math.pi/180.0
W['ToLeft'] = W.playDirection == "left"
W['TeamOnOffense'] = "home"
W.loc[W.possessionTeam != W.homeTeamAbbr, 'TeamOnOffense'] = "away"
W['IsOnOffense'] = W.team == W.TeamOnOffense # Is player on offense?
W['YardLine_std'] = 100 - W.yardlineNumber
W.loc[W.yardlineSide.fillna('') == W.possessionTeam,'YardLine_std'] = W.loc[W.yardlineSide.fillna('') == W.possessionTeam,  'yardlineNumber']
W['X_std'] = W.x
W.loc[W.ToLeft, 'X_std'] = 120 - W.loc[W.ToLeft, 'x'] 
W['Y_std'] = W.y
W.loc[W.ToLeft, 'Y_std'] = 160/3 - W.loc[W.ToLeft, 'y'] 
#W['Orientation_std'] = -90 + W.Orientation
#W.loc[W.ToLeft, 'Orientation_std'] = np.mod(180 + W.loc[W.ToLeft, 'Orientation_std'], 360)
W['Dir_std'] = W.Dir_rad
W.loc[W.ToLeft, 'Dir_std'] = np.mod(np.pi + W.loc[W.ToLeft, 'Dir_rad'], 2*np.pi)
W['dx'] = round(W['s']*np.cos(W['Dir_std']),2)
W['dy'] = round(W['s']*np.sin(W['Dir_std']),2)
W['X_std'] = round(W['X_std'],2)
W['Y_std'] = round(W['Y_std'],2)
#W['Orientation_rad'] = np.mod(W.o, 360) * math.pi/180.0
W['Orientation_rad'] = np.mod(-W.o + 90, 360) * math.pi/180.0
W['Orientation_std'] = W.Orientation_rad
W.loc[W.ToLeft, 'Orientation_std'] = np.mod(np.pi + W.loc[W.ToLeft, 'Orientation_rad'], 2*np.pi)
W['Orientation_deg_std'] = np.rad2deg(W['Orientation_std'])
W['MPH'] = W['s'] / 0.488889
Standardised  = W
Standardised['X_std_true'] = Standardised['X_std'] - (Standardised['YardLine_std']+10) # want to centre X axis on 0 for consistency
Standardised['Y_std_true'] = Standardised['Y_std']

# identifying safeties location at the snap of the ball
S = Standardised[(Standardised['position'].isin(['FS','SS']))&(Standardised['frameId'] == 11)]
S['db_rank'] = S.groupby(['PlayerTeam','gameId','playId'])['X_std_true'].rank(method="first", ascending=False)

# identifying safeties location when the ball is passed
S2 = Standardised[(Standardised['position'].isin(['FS','SS']))&(Standardised['event'] == 'pass_forward')]
S3 = S2[['gameId','playId','nflId','displayName','X_std_true']]
S3.rename(columns={'X_std_true': 'pass_X_std_true'}, inplace=True)

# excluding snaps which are in the redzone as safeties would not be lined up deep to reduce bias
# excluding snaps in the first 10 yards as defences may be playing them super aggressive and close to the line
# ranking safeties by their depth, deepest safety will be rank 1.0
S_no_red = S[(S['YardLine_std'] <= 80)&(S['YardLine_std'] >= 10)&(S['db_rank'].isin([1.0,2.0]))].groupby(['gameId','playId','nflId','displayName']).agg({'X_std_true': 'mean'})
S_no_red.reset_index( drop=False, inplace=True)
S_no_red = pd.merge(S_no_red, S3, how='left', on=['gameId','playId','nflId','displayName'])

td_3r_w_def = pd.merge(td_3r, S_no_red, how='inner', left_on=['gameId','playId'], right_on=['gameId','playId'])

T_depth =  td_3r_w_def.groupby(['PlayerTeam']).agg({'X_std_true': 'mean','pass_X_std_true': 'mean'})
T_depth.reset_index( drop=False, inplace=True)

# as an alternative to the 40 time, I also want to include in game speed.
# Reason as 40 time is speed at entry into the NFL, players may be older or slower due to injuries so the 40 time may no longer be representative

# to identify speed I am taking their max speed on a GO (straight line) route in the first 4 seconds of play
# I will eventually use their median top speed as their representive speed to exclude tail anomalies 
#    (e.g. slow for blocking plays, or postentially super quick if doing a tackling motion/ diving)
con1 = (week1_data['position'] == 'WR')
con2 = (week1_data['frameId'] >= 11)
con3 =(week1_data['frameId'] <= 50)
con4 =(week1_data['route'].isin(['GO']))
WR2 = week1_data[con1 & con2 & con3 & con4]

WR3 = WR2.groupby(['gameId','playId','nflId','displayName']).agg({'s': max})
WR3.reset_index( drop=False, inplace=True)
WR3['s_round'] = WR3['s'].round(0)

WR_ig = WR3.groupby(['nflId','displayName']).agg({'s':'median'})
WR_ig.reset_index( drop=False, inplace=True)

WR_e = pd.merge(WR_a2, WR_ig, how='left', left_on=['displayName'], right_on=['displayName'])
WR_e['s_rank'] = WR_e.groupby(['PlayerTeam'])['s'].rank(method="first", ascending=False)

T_40_2 = WR_e.groupby('PlayerTeam').agg({'time_40': min, 's': max})
T_40_2.reset_index( drop=False, inplace=True)

summary2 = pd.merge(T_depth, T_40_2, how='inner', left_on=['PlayerTeam'], right_on=['PlayerTeam'])
summary2['X_std_true_diff'] = summary2['pass_X_std_true'] - summary2['X_std_true']
summary2.rename(columns={'X_std_true': 'Avg Safety Depth At Snap', 'pass_X_std_true': 'Avg Safety Depth At Pass', 'time_40': 'Fastest WR 40 time', 's': 'Fastest WR speed, yards/second', 'X_std_true_diff': 'Avg Safety Drop Snap To Pass'}, inplace=True)
summary2


# In[ ]:


sns.lmplot(x='Fastest WR 40 time', y='Avg Safety Depth At Snap', data=summary2, ci=None, height=8.27, aspect=11.7/8.27)
plt.ylim(0,20)
plt.xlim(4.2,4.55)


# In[ ]:


sns.lmplot(x='Fastest WR speed, yards/second', y='Avg Safety Depth At Snap', data=summary2, ci=None, height=8.27, aspect=11.7/8.27)
plt.ylim(0,20)
plt.xlim(7.25,8.75)


# In[ ]:


sns.lmplot(x='Fastest WR speed, yards/second', y='Avg Safety Drop Snap To Pass', data=summary2, ci=None, height=8.27, aspect=11.7/8.27)
plt.ylim(0,8)
plt.xlim(7.25,8.75)


# In[ ]:


# Version 2, only include plays where GO route from the fastest player. 
# As no real correlation that faster players means the safeties are deeper
td_3r = pd.merge(td_3, p_r, how='inner', left_on=['gameId','playId'], right_on=['gameId','playId'])

GO_WR_S = WR_e[WR_e['s_rank'] == 1.0]

GO_WR = Standardised[(Standardised['route'] == 'GO')&(Standardised['frameId'] == 1)&(Standardised['position'] == 'WR')][['gameId','playId','PlayerTeam','displayName']]

GO_WR_1 = pd.merge(GO_WR, GO_WR_S, how='inner', on=['PlayerTeam','displayName'])
GO_WR_2 = pd.merge(GO_WR_1, td_3r, how='inner', on=['gameId','playId'])
GO_WR_3 = GO_WR_2[['gameId','playId']]

td_3r_w_def2 = pd.merge(td_3r_w_def, GO_WR_3, how='inner', on=['gameId','playId'])

T_depth2 =  td_3r_w_def2.groupby(['PlayerTeam']).agg({'X_std_true': 'mean','pass_X_std_true': 'mean'})
T_depth2.reset_index( drop=False, inplace=True)

summary3 = pd.merge(T_depth2, T_40_2, how='inner', left_on=['PlayerTeam'], right_on=['PlayerTeam'])
summary3['X_std_true_diff'] = summary3['pass_X_std_true'] - summary3['X_std_true']
summary3.rename(columns={'X_std_true': 'Avg Safety Depth At Snap', 'pass_X_std_true': 'Avg Safety Depth At Pass', 'time_40': 'Fastest WR 40 time', 's': 'Fastest WR speed, yards/second', 'X_std_true_diff': 'Avg Safety Drop Snap To Pass'}, inplace=True)


# In[ ]:


sns.lmplot(x='Fastest WR 40 time', y='Avg Safety Depth At Snap', data=summary3, ci=None, height=8.27, aspect=11.7/8.27)
plt.ylim(0,20)
plt.xlim(4.2,4.55)


# In[ ]:


sns.lmplot(x='Fastest WR speed, yards/second', y='Avg Safety Depth At Snap', data=summary3, ci=None, height=8.27, aspect=11.7/8.27)
plt.ylim(0,20)
plt.xlim(7.25,8.75)


# In[ ]:


sns.lmplot(x='Fastest WR speed, yards/second', y='Avg Safety Drop Snap To Pass', data=summary3, ci=None, height=8.27, aspect=11.7/8.27)
plt.ylim(0,8)
plt.xlim(7.25,8.75)

