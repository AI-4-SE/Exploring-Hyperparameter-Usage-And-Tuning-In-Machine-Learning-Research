#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# To Quote the great Mike Tyson - **"Everybody has a plan until they get punched in the mouth"**
# 
# This notebook will be investigating this effect in the NFL and seeing how well teams respond to their plan being derailed.
# 
# But punching is illegal in the NFL I hear you say? That is correct! 
# * I am defining a punch in the mouth as going over a touchdown down on the scoreboard
#     * Counting it as 8pts or more so they need two scores, or score and 2pt conversion
# 
#   
#   
# **Enter the hypothesis**
# 
# Here is an example of my hypothesis where you can have a strong team, however falls apart quicky when things go wrong.
# Hypothetical Team A:
# * Strong Running game
# * Very efficient play action pass scheme to take advantage of when the defence overcommits to the run
# * Defence built on rushing the passer and forcing turnovers in coverage
# 
# When things are going well Team A will be dominant on the field:
# * Rushing will wear down the defence and take time off the clock
# * Play action passing will be rolling and smashing efficiency metrics and the 
# * Opposition will struggle to keep up as the defence will shut down the passing game
# 
# But what happens when this team goes down by over a touchdown? 
# * The offence will need to shift more towards the pass to prevent running out of time
# * The defence will not be biting on the rush as much so play action is less effective
# * The opposition while ahead will likely rush the ball more which negates a strong pass rush and coverage
# 
# While most of the time this team will be smashing the efficiency metrics when beating up weak teams, there is potential that there could be a weakness which is exposed in the playoffs vs the strongest competition who may get ahead early.

# # TL;DR
# 
# The more successful teams in the league who make the Superbowl do seem to have the ability to overcome large deficits more often than those who just make the playoffs or miss out altogether.
# Comeback % from 8pt deficits:
# * Superbowl - 42%
# * Playoffs - 27%
# * Outside Playoffs - 14%
# 
# 
# In addition there seems to be a weak to moderate correlation (0.32, 0.37 if excluding 1 annomaly) between ability to comeback from a defeict and the base offence being built around the pass. I will count this correlation as important as there are many factors which go into comeback from deficit which this analysis has not been able to take into account which I would expect to also be significant e.g. defensive ability or passing efficiency.
# 
# If trying to build a team where the goal is to maximise the chances of making the superbowl I would start by building an offense which is built around the pass and is not reliant on the run to function sucessfuly. Ideally the base offense would be made up of 60-70% passing plays, I cannot confirm if over 75% is detrimental as no team has ever done it before.

# # Methodology
# 
# What counts as a comeback?
# * Team to go down by 8 or more points on a scoring play
# * A successful comeback is if they retake the lead at anypoint later in the game (they do not need to have won the game)
# 
# Assumptions in the analysis:
# * Data taken from the last 5 NFL seasons
# * Comeback % derived from the regular season only to ensure balance
# * QB scrambles have been counted as passing plays in this analysis even though generally counted as rushing plays
#     * Counted as passing plays as for the majority of QBs it is originally designed as a passing play but the pocket breaks down and they run
#     * This may slighly inflate the passing numbers for some teams with designed QB scrambles, e.g. BAL (Jackson), PHI (Hurts) and CAR (Newton)
# * A teams base offensive makeup has been defined as their behavior when leading by under a touchdown and not losing by more than a touchdown
#     * When losing by more than a touchdown teams skew towards pass, when leading they skew towards the run. 

# # Initial Results
# 
# When we plot comeback % against the % of passing plays in the base offense we can see a slight correlation between the two variables. 
# 
# This result makes sense as generally you need to be effective at passing the ball to comeback from a large points defecit, the closer passing is to your comfort offense the least you need to vary from your initial gameplan. 
# 
# The moderate correlation is also expected as there are other factors which have not been included in this analysis which I would expect to be significant e.g.
# * Strength of defense, the defense needs to be able to stop some points being scored otherwise the offense will never catch up
# * Passing is needed to make a comeback, but you need to be able to execute to make a positive gain
# 
# When we overlay the results with the final finsh in the season we can see that those teams who made the superbowl have a higher comeback % and generally are skewed towards being a pass heavy offense when under neutral game conditions.

# ![Pass to comeback %.png](attachment:5d1aec19-9e78-4925-bb55-aca613e25e8e.png)
# 
# Correlation 0.32 (0.37 if excluding SFO who had a comeback % of 100% in 2019)

# The majority of teams who make the superbowl have a base offense passing % above 60%, only 2 superbowl teams were under that threshold
# * SFO in 2019, 100% comeback, Base Offence - 55% passing plays
# * NE in 2018, 7% comeback, Base Offence - 58% passing plays
# 
# ![pass to comeback with ranking.png](attachment:4e837184-334b-4508-b794-40541b813840.png)

# Those who reach the superbowl have a comeback % of 42%, much higher than those who reach the playoffs 27% or those who miss the playoffs 14%
# 
# ![NFL anti fragile Season finsh.png](attachment:4aa6adbc-c101-4cc4-9e0c-15cc7e74d45e.png)

# **2017 NFL season**
# 
# Notable mentions:
# 
# The Pittsburg Steelers went into the playoffs with the highest comeback %, however they let the Jaxonville Jaguars get off to a 21 point lead early in the game, although still managed to bring the game back to 3pts near the end showing their ability to threaten a sizable comeback.
# 
# The Atlanta Falcons managed to upset the LA Rams by never giving the lead away in the first round of the playoffs. However as soon as they lost the lead to the Phillidelphia Eagles they were eliminated.
# 
# ![NFL anti fragile 2017 season.png](attachment:57d79fde-570b-43ef-867a-424d4ad5585d.png)

# **2018 NFL season**
# 
# Notable mentions:
# 
# The New England Patriots managed to make the Superbowl with excelent defence and never giving a sizable lead away in the playoffs, they were always ahead vs the San Diego Chargers, and only went behind by 4 vs Kansas City Chiefs. I wonder if they would have won the superbowl if they ever went behind by more than a touchdown?
# 
# ![NFL anti fragile 2018 season.png](attachment:1240dbe0-2457-44b3-82de-b54795df824a.png)

# **2019 NFL season**
# 
# Notable mentions:
# 
# Both of the teams who reached the superbowl were leaders in the comeback % metric. One impressive stat was that the San Fransisco 49ers always managed to retake the lead if they ever went more than a touchdown behind which can back up the notion that Kyle Shanahan is one of the leading offensive coaches in the league.
# 
# ![NFL anti fragile 2019 season.png](attachment:79a03154-52ac-49bd-92e6-bcb3534f249a.png)

# **2020 NFL season**
# 
# Notable mentions:
# 
# An almost repeat of the 2019 season where two of the teams with some of the highest comeback % managed to make it to the Superbowl.
# 
# ![NFL anti fragile 2020 season.png](attachment:facedcbd-af37-4693-93b0-f48d1dc0ee1b.png)

# **2021 NFL season**
# 
# Notable mentions:
# 
# The Cincinnati Bengals managed to turn their offense around very quicky from the prior struggles and were one of the leaders in comeback % with their new star players in Burrow and Chase. In the superbowl the Bengals showed their comeback skills by manging to come back and retaking the lead from a 14-3 defecit, however lost late in the 4th quarter to the Rams.
# 
# A surprising observation was how high the comeback % was for the Baltimore Ravens even though they did not make the playoffs, I wonder if this was related to the injuries to the RBs and having to embrase the passing offense on Lamar Jackson's shoulders.
# 
# The LA Rams initially may have looked like a surprise for the Superbowl, however to make they playoffs they needed for face similarly ranked Philadelphia Eagles, their next opponent was the injury ravaged Tampa Bay Buccaners. Before reaching the San Fransisco 49ers who had knocked out both Green Bay Packers and Dallas Cowbows.
# 
# ![NFL anti fragile 2021 season.png](attachment:8bc832de-815f-438c-96ee-de86ad3f2e27.png)

# # Next Steps
# 
# While we have been able to answer some of the questions as to how to make an offense more resiliant there are still some questions which need to be addressed in both defense and offensive passing efficiency. Sadly without tracking data we are unlikely to get the answers to these any time soon, so can only hope for these to be covered in future NFL Data Bowls.

# In[ ]:




