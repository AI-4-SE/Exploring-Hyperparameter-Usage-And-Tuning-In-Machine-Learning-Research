#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Introduction**
# 
# In the fall of 2016 I attended the first NFL Head Helth TECH Symposium as a graduate student, where I was introduced to the NFL's plans to implement a number of strategies for the improvement of player head health and safety. Since then I have closely followed the NFL and researcher's progress through the roadmap proposed while conducting my own investigations into head injury as well. I am excited for the opportunity to participate in an event that centers on head health and safety.
# 
# In order to propose a rule change with optimal efficacy, it is important to first understand the origins of concussion and head injuries. A lot of research has been conducted investigating the particular mechanism of concussion (rotational vs. linear accelerations of the head)[1-21], but current research, based on finite element modeling and experimental studies, has pinpointed brain shearing as one of the primary causes for acute brain damage[22-25]. Injurious brain shearing is due to a combination of both linear and angular accelerations and more often occurs as the result of direct impacts to the head as opposed to inertial loading from and impact to the body due to the compliance of the neck[26]. These direct impacts result in head accelerations that transmit a shear wave into the brain resulting in local displacements that can damage axons[22-25]. However, there is still limited data available on what acceleration levels produce acute injuries, so the rule changes proposed will be focused on limiting head accelerations overall.
# 
# The highest head accelerations tend to occur during events in which players have the highest changes in velocity (Delta V) because of a number of factors. First there is a larger amount of input energy being supplied to the system meaning overall, there is a greater amount of energy that will accelerate the head. Second, the higher the Delta V, often the more chaotic the event and the greater the chance the player will receive a direct blow to the head. Finally, these high Delta V events often occur when one player is unaware the hit is coming, preventing them from being able to properly position both their body and head to prevent injury. These high Delta V events are extremely common on NFL kickoffs and were a primary contributor to the higher injury rates; similar to the kickoff rule changes implemented in 2017, the rule changes proposed here will target these high Delta V events.
# 
# While it is important to evolve the rules of football to continuously improve player health and safety, these changes cannot be made in a vacuum. Any change will have an effect on the integrity of the sport and it is important to optimize this relationship between player safety and game integrity by anticipating these changes prior to rule implementation. An example of this is the change to the kickoff rule that prevents players from getting a running start prior to the ball being kicked. While the effect this rule has on head injuries has yet to be seen, it has limited the success of onside kicks. This rule change has altered how games are completed, but it has not drastically effect the overall product of the NFL, games are still exciting to watch, experience, and play.
# 
# The deliverable of this analysis is to present a rule modification to the NFL that will reduce the occurrence of concussions, and more specifically high Delta V events, on punt plays while still maintaining the integrity of the play. This means I must consider the purpose of the punt play in the NFL, field position. A punt allows the team with the ball to forgo one of their downs in order to "flip the field" on the opponent and make it more difficult for them to score by having to travel a further distance down the field. However one caveat is that the team receiving the punt has the opportunity to return the kick and counter this attempt to control field position. Both the kick and the return are important aspects to the overall flow and outcome of the game, and the effects rule changes will have on both of these elements are considered in this analysis.  

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML
import math
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
from plotly import figure_factory as FF
init_notebook_mode()


# In[ ]:


VideoReview = pd.read_csv('../input/video_review.csv')
VideoFootageInjury = pd.read_csv('../input/video_footage-injury.csv')
PlayInfo = pd.read_csv('../input/play_information.csv')
VidoeFootageControl = pd.read_csv('../input/video_footage-control.csv')


# To begin, I first look at the percentage of punt plays that result in a concussion over the 2016 and 2017 seasons.

# In[ ]:


RateOfHeadInjury = (len(VideoReview)/len(PlayInfo))*100
print ('Rate of Punt Play Head Injury: %.2f%%' % RateOfHeadInjury)


# **What Makes Punts Different**
# 
# While 0.55% is higher than that of normal plays (~0.5%), it is less than the kick-off concussion rate, meaning that there are some beneficial differences between punts and kick-offs even though they are "similar" plays. Again this is most likely due to the number of high Delta V events that occur on the play. One of the key differences between punts and kickoffs is that blockers and those blocking start within a few yards of one another at the beginning of a punt play. This means that they are often running with one another in the same direction and will have a lower Delta V when they impact one another as opposed kickoffs where the blockers are approaching the players they are blocking from the other direction. However, during a punt play there are multiple events that would result in high Delta V events like those in a kick-off. The first is unsuspecting blocks on defenseless players which often occurs when the player is focused on chasing down the returner and does not see the blocker coming from a different direction. The second is when someone has another player blocked into them and they are not anticipating the blow. I hypothesize that a large portion of injuries on punt plays will be a result of these two events.

# Now, I will consider the number of concussions that occurred on punt plays between the 2016 and 2017 seasons.

# In[ ]:


print (len(VideoReview))


# **Parsing Injury Plays to Those Relevant for Analysis**
# 
# I analyzed these videos of these thirty-seven injuries, looking for particular items. First does the injury occurr in a scenario that happens regularly on a play from line of scrimmage? This will differentiate injuries unique to punt plays. "Regular plays from line of scrimmage" include head injuries of linemen blocking prior to the punt being kicked, fake punt plays, and defensive players rushing. This removes nine plays, leaving twenty-eight for further analysis. Next, are there any plays where the injury was a result of an illegal play or would have been an illegal play with the implementation of this year's helmet rule? Rules have already been implemented to protect players in these situations, limiting their relevance to this analysis. This removes another nine plays, leaving nineteen for analysis. Finally, with two plays (Play IDs 1526 and 1262), the video was inadequate to determine the nature of the event surrounding the injury leaving seventeen plays. These results are summarized in the table below. 

# In[ ]:


table_data = [['Play ID', 'Unique Punt Injury', 'Injury Due to Illegal Play', 'Penalty Under New Helmet Rule'],
              ['3129', 'No', 'Yes', 'No'],
              ['1212', 'Yes', 'No', 'No'],
              ['905', 'Yes', 'No', 'No'], 
              ['2342', 'No', 'No', 'Yes'],
              ['3509', 'No', 'No', 'Yes'],
              ['3278', 'No', 'No', 'Yes'],
              ['2902', 'No', 'Yes', 'Yes'],
              ['2918', 'Yes', 'No', 'No'],
              ['3746', 'Yes', 'No', 'No'],
              ['3609', 'Yes', 'No', 'No'],
              ['2667', 'Yes', 'No', 'Yes'],
              ['3312', 'Yes', 'No', 'No'],
              ['1988', 'No', 'No', 'Yes'],
              ['1407', 'Yes', 'No', 'No'],
              ['733', 'No', 'Yes', 'Yes'],
              ['2208', 'No', 'No', 'Yes'],
              ['2792', 'No', 'No', 'Yes']]
# Initialize a figure with FF.create_table(table_data)
#figure = py.figure_factory.create_table(table_data, height_constant=60)
figure = FF.create_table(table_data)
iplot(figure)


# **Results of Relevant Injury Plays**
# 
# The next analysis looked at whether or not the injuries were result of an unsuspecting block on a defenseless player [Illustrated Here](http://a.video.nfl.com//films/vodzilla/153258/61_yard_Punt_by_Brett_Kern-g8sqyGTz-20181119_162413664_5000k.mp4). Five of the seventeen injuries (30%) were a result of unsuspecting blocks on defenseless players, an event that results in a Delta V much larger than other blocks during a punt because it is often due to gunners or other members of the punt team that have turned back up field in pursuit of the returner and are blocked by players traveling downfield. Next, the number of injurious plays that were a result of players being blocked into another player was analyzed [Illustrated Here](http://a.video.nfl.com//films/vodzilla/153240/Punt_by_Thomas_Morstead-eZpDKgMR-20181119_154525222_5000k.mp4). Five of the seventeen injuries (30%) were a result of a player being blocked into another unsuspecting player. Again, these are plays that result in a high Delta V because often the player being impacted is traveling in a different direction from the impactor or is stationary. Six of the remaining seven injuries are due to the "spread" of a punt play, or the idea that since the play occurs over such a large portion of the field, players are able to stop, readjust direction and accelerate to converge on a single point over large distances unhindered, [Illustrated Here](http://a.video.nfl.com//films/vodzilla/153259/41_yard_Punt_by_Toby_Baker-gyQgJXCY-20181119_162457711_5000k.mp4). An important distinction here is that these injuries do not appear to occur due to the fact that players are accelerating down the field, instead they are able to slow down near the returner and still have more open space to accelerate than during a non-punt play. The final play was an anomaly in which the gunner tripped and fell head first into the feet of the returner, causing him to be kicked in the head. Furthermore, there were multiple plays where a defenseless player was blocked into another player, for these, the most prominent event was used for analysis. These results are summarized in the table below.

# In[ ]:


table_data = [['Play ID', 'Block on Defenseless Player', 'Blocked into Another Player', 'Injury from Play Spread'],
              ['2587', 'Yes', 'No', 'No'],
              ['1045', 'Yes', 'Yes', 'No'],
              ['3663', 'No', 'Yes', 'No'],
              ['3468', 'No', 'Yes', 'No'], 
              ['1976', 'Yes', 'Yes', 'No'],
              ['2341', 'Yes', 'Yes', 'No'],
              ['2764', 'Yes', 'No', 'No'],
              ['1088', 'Yes', 'No', 'No'],
              ['2792', 'Yes', 'No', 'No'],
              ['1683', 'Yes', 'No', 'No'],
              ['538', 'No', 'No', 'No'],
              ['3630', 'No', 'No', 'Yes'],
              ['2489', 'No', 'No', 'Yes'],
              ['183', 'No', 'No', 'Yes'],
              ['1526', 'No', 'No', 'Yes'],
              ['2072', 'No', 'No', 'Yes'],
              ['602', 'No', 'No', 'Yes']]
# Initialize a figure with FF.create_table(table_data)
#figure = py.figure_factory.create_table(table_data, height_constant=60)
figure = FF.create_table(table_data)
iplot(figure)


# **Rule Modification Suggestions and Discussion**
# 
# Based on my analysis, there are two rule modifications I would like to propose. The first is to add an amendment to Rule 12, Section 2, Article 7, Part a, that defines players considered in a defenseless posture. The amendment would be a 13th clause that states "A member of the kicking team in a punt or kickoff formation who has begun to turn back up field in pursuit of the returner." This will address injuries caused by unsuspecting blocks on defenseless players, allowing blocks to be made as long as there is not "unnecessary contact" as determined by the official. This will have minimal impact on the integrity of the punt play for two reasons; first, often these blocks are on players who are too distant to make a legitimate play on the returner, second, these players can still be blocked, just not with the extraneous force that will result in injury. The second is an amendment to Rule 12, Section 1, Article 1, that lists illegal blocks. The amendment would state "r. blocking a player into another player 10 yards beyond the line of scrimmage following a punt". This would require players to be more cognizant of blocks they are making and blocks that are occurring around them but have a minimal impact on the overall course of the punt play because it does not hinder the ability to make successful blocks. By making these two rule modifications, over half of the current legal punt specific plays that result in injuries in the 2016 and 2017 seasons would be deemed illegal. One final point is focused around the injuries that are not a result of a defenseless posture or one player being blocked into another. In a majority of these injuries, defending players had traveled down the field, slowed down, adjusted course, and then reaccelerated in pursuit of the returner. This indicates that changes to the initial formation, or procedure prior to players traveling down field would had limited effect on reducing head injuries.

# **Conclusions**
# 
# - Goal is to reduce the occurrence of high Delta V events while maintaining the integrity of punt plays.
# - High Delta V events often are the result of two things: Unexpected blocks on defenseless players and players being blocked into other players.
# - These two events occurred on 60% of the relevant punt specific plays that result in concussions.
# - The other 40% of injuries are due the amount of open space available on punt plays, not necessarily the initial formations or coverage.
# - **Rule Modification 1:** Amend Rule 12, Section 2, Article 7, Part a, to add "A member of the kicking team in a punt or kickoff formation who has begun to turn back up field in pursuit of the returner." as a defenseless posture.
# - **Rule Modification 2:** Amend Rule 12, Section 1, Article 1, to include "blocking a player into another player 10 yards beyond the line of scrimmage following a punt" as an Illegal Block.

# References:
# 1. Gurdjian E, Webster J. Linear acceleration causing shear in the brain stem in trauma of the central nervous system. Mental Advances in Disease. 1945;24:28.
# 2. Gurdjian E, Webster J, Lissner H. Observations on the mechanism of brain concussion, contusion, and laceration. Surgery, gynecology & obstetrics. 1955;101(6):680.
# 3. Gurdjian ES, Lissner H, Evans FG, Patrick L, Hardy W. Intracranial pressure and acceleration accompanying head impacts in human cadavers. Surgery, gynecology & obstetrics. 1961;113:185.
# 4. Gurdjian ES, Lissner H, Patrick L. Concussion: mechanism and pathology. Paper presented at: Proceedings: American Association for Automotive Medicine Annual Conference 1963.
# 5. Ono K, Kikuchi A, Nakamura M, Kobayashi H, Nakamura N. Human Head Tolerance to Sagittal Impact—Reliable Estimation Deduced from Experimental Head Injury Using Subhuman Primates and Human Cadaver Skulls. SAE Transactions. 1980:3837-3866.
# 6. Eiband AM. Human tolerance to rapidly applied accelerations: a summary of the literature. 1959.
# 7. Gadd CW. Use of a weighted-impulse criterion for estimating injury hazard. SAE Technical Paper; 1966. 0148-7191.
# 8. Mertz HJ, Prasad P, Nusholtz G. Head injury risk assessment for forehead impacts. Stapp Car Crash J. 1996.
# 9. Nusholtz GS, Glascoe LG, Wylie EB. Cavitation during head impact. SAE Technical Paper; 1997. 0148-7191.
# 10. Nusholtz GS, Wylie B, Glascoe LG. Cavitation/boundary effects in a simple head impact model. Aviat Space Envir Md. 1995;66(7):661-667.
# 11. Nusholtz GS, WYLIE EB, GLASCOE LG. Internal cavitation in simple head impact model. J Neurotrauma. 1995;12(4):707-714.
# 12. Cullen DK, Harris JP, Browne KD, et al. A porcine model of traumatic brain injury via head rotational acceleration. Injury Models of the Central Nervous System: Methods and Protocols. 2016:289-324.
# 13. Browne KD, Chen X-H, Meaney DF, Smith DH. Mild traumatic brain injury and diffuse axonal injury in swine. J Neurotrauma. 2011;28(9):1747-1755.
# 14. Gennarelli T, Adams J, Graham D. Acceleration induced head injury in the monkey. I. The model, its mechanical and physiological correlates. Experimental and Clinical Neuropathology: Springer; 1981:23-25.
# 15. Adams J, Graham D, Gennarelli T. Acceleration induced head injury in the monkey. II. Neuropathology. Experimental and Clinical Neuropathology: Springer; 1981:26-28.
# 16. Gennarelli T, Ommaya A, Thibault L. Comparison of translational and rotational head motions in experimental cerebral concussion. Paper presented at: Proc. 15th Stapp Car Crash Conference 1971.
# 17. Gennarelli TA, Thibault L, Ommaya AK. Pathophysiologic responses to rotational and translational accelerations of the head. SAE Technical Paper; 1972. 0148-7191.
# 18. Gennarelli T, Segawa H, Wald U, Czernicki Z, Marsh K, Thompson C. Physiological response to angular acceleration of the head. Head injury: basic and clinical aspects. 1982;1982:129-140.
# 19. Thibault LE, Gennarelli TA. Biomechanics of diffuse brain injuries. SAE Technical Paper; 1985.
# 20. King AI, Yang KH, Zhang L, Hardy W, Viano DC. Is head injury caused by linear or angular acceleration. Paper presented at: IRCOBI conference 2003.
# 21. Margulies SS, Thibault LE. A proposed tolerance criterion for diffuse axonal injury in man. J Biomech. 1992;25(8):917-923.
# 22. Meaney DF, Morrison B, Bass CD. The mechanics of traumatic brain injury: a review of what we know and what we need to know for reducing its societal burden. J Biomech Eng. 2014;136(2):021008.
# 23. Takhounts EG, Craig MJ, Moorhouse K, McFadden J, Hasija V. Development of brain injury criteria (BrIC). Stapp Car Crash J. 2013;57:243.
# 24. Giudice JS, Alshareef A, Forman J, Panzer MB. Measuring 3D Brain Deformation During Dynamic Head Motion Using Sonomicrometry. Paper presented at: IRCOBI Conference Proceedings 2017.
# 25. Hardy WN, Foster CD, Mason MJ, Yang KH, King AI, Tashman S. Investigation of head injury mechanisms using neutral density technology and high-speed biplanar X-ray. Stapp Car Crash J. 2001;45:337-368.
# 26. Eckersley CP, Nightingale RW, Luck JF, Bass CR. Effect of Neck Musculature on Head Kinematic Response Following Blunt Impact. IRCOBI Conference Proceedings 2017.
