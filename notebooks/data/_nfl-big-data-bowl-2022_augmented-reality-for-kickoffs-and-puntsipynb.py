#!/usr/bin/env python
# coding: utf-8

# # Zone Segmentation - Augmented Reality for Kickoffs and Punts
# 
# <img src="https://github.com/JohnM-TX/misc/raw/main/combo4_adobe.gif" width="700">

# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\n\ndiv.h1 {\n    font-size: 32px; \n    margin-bottom:2px;\n}\ndiv.h2 {\n    background-color: steelblue; \n    color: white; \n    padding: 8px; \n    padding-right: 300px; \n    font-size: 24px; \n    max-width: 1500px; \n    margin-top: 50px;\n    margin-bottom:4px;\n}\ndiv.h3 {\n    color: steelblue; \n    font-size: 18px; \n    margin-top: 4px; \n    margin-bottom:8px;\n}\ndiv.h4 {\n    font-size: 15px; \n    margin-top: 20px; \n    margin-bottom: 8px;\n}\nspan.note {\n    font-size: 5; \n    color: gray; \n    font-style: italic;\n}\nspan.lists {\n    font-size: 16; \n    color: dimgray; \n    font-style: bold;\n    vertical-align: top;\n}\nspan.captions {\n    font-size: 5; \n    color: dimgray; \n    font-style: italic;\n    margin-left: 130px;\n    vertical-align: top;\n}\nhr {\n    display: block; \n    color: gray;\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}\nhr.light {\n    display: block; \n    color: lightgray;\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}\ntable.dataframe th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n}\ntable.dataframe td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 14px;\n    text-align: center;\n} \ntable.rules th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 14px;\n}\ntable.rules td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 13px;\n    text-align: center;\n} \ntable.rules tr.best\n{\n    color: green;\n}\n\n</style>\n')


# The NFL Big Data Bowl 2022 has provided an opportunity for the data science community to examine special teams plays. We examined data and researched past and present NFL operations to see where we could best contribute. Augmented reality(AR) has been a big part of sports and is of growing importance for the NFL as they continually evolve.
# 
# We present an implementation of Zone Segmentation, which shows accessibility and control of space during a play. We believe this addition to game video and the analysis it enables will benefit viewers and professionals alike.
# 
# 
# Our report contains the following sections:
# - The Benefits of Successful AR
# - Design
# - Methodology
# - Additional Analysis
# - Conclusion
# 
# Thank you for this opportunity!
# John Miller
# Uri Smashnov

# <a id='bg'></a>
# <div class="h2">The Benefits of Successful AR</div>
# 

# The NFL has been using AR for decades to **increase fan engagement**. One of the most enduring examples is the telestrator
# , first used in the NFL by John Madden at the 1982 Super Bowl. The legendary coach and iconic commentator used the electronic chalkboard to explain plays with X’s and O’s, and it has since been used countless times to help viewers understand the game. Madden also demonstrated how AR can improve the entertainment aspect of watching a game as he delighted fans with things like excessive doodling and tips on how to carve a turducken.
# 
# <img src="https://github.com/JohnM-TX/misc/raw/main/madden_ensemble.png" width="700">

# AR also provides cues during a game to **help viewers quickly understand** what is happening and what might happen next. The yellow first down line is a perfect example here. Fans at all levels of experience benefit from seeing this information right on the field. It greatly reduces the amount of brain work that’s required without the line --  looking at the down marker, adjusting for camera angle, looking back to the ball’s location, all while trying to keep one’s eye on the action. The line and it’s associated graphics are helpful and unobtrusive, giving them universal appeal.
# 
# <img src="https://github.com/JohnM-TX/misc/raw/main/super-bowl-lines.png" width="700">

# Investing in AR is more important to a sports league’s success than ever before. A 2019 study on engaging sports fans found that the visual appeal of AR is "very influential" on the degree of positive word of mouth. [1] In the NFL, coaches and teams now rely heavily on Next Gen Stats and analytics. Fans with a high appetite for analysis subscribe to NFL Game Pass and watch Thursday Night Football on Prime Video. And as the NFL expands its presence in eight countries later this year, attracting new fans to the game will partly depend on how easily they can understand it.

# <a id='bg'></a>
# <div class="h2">Design</div>

# Our contribution to AR for football, video zone segmentation,  is based on bringing viewers the benefits listed above for kickoff and punt returns. We used the following principles to guide our design:
# -	Help viewers interpret what they see during a return
# -	Adapt to dynamic broadcast camera views in addition to the more stable NFL coaches view
# -	Take advantage of “preattentive attributes” of the human brain: color, shape, and spatial positioning
# -	Minimize any distraction from added graphics
# 
# With these principles in mind, we formed our initial concept of helping viewers see where each team controls space on the field. We use the term “controlling space” in a general sense. It can also be thought of as accessibility. Some examples:
# -	If two players are running toward a space, the player who will reach it first controls it
# -	If one or more players is blocking another player, each player controls the space on his side of the block (generally behind him)
# -	The returner can only control the space where he is likely to gain first access. 
# 
# The picture below shows how segmenting zones between teams helps viewers interpret the action. The left-hand frame shows how a viewer might interpret this scene where the Cowboys are returning a punt. The three red arrows show a defender, a blocker, and the returner with the ball. It appears that the three players will reach the X at the same time, or the blocker in the middle might turn and intercept his opponent to protect the ball carrier.
# 
# <img src="https://github.com/JohnM-TX/misc/raw/main/cowboys.png" width="1200">
# 
# But this interpretation would be wrong. The sideline perspective from the camera distorts the actual scale by quite a bit. Notice the yellow arrows showing 30 feet downfield and laterally. One is twice as long as the other as drawn on the image. Looking at the middle frame, we see the reality of the field. The defender is nowhere near the X and would miss the defender by a large margin if he were to run at that angle. 
# 
# The picture in the right-hand frame shows zone segmentation with the proper perspective. All three players are running at a shallow angle  with the ball carrier working toward the sideline. The shading between the defender and blocker correctly shows the space closest to them and suggests a likely angle of interception. To be fair, the human brain is pretty good at judging perspective when watching the video. But it takes mental work to go back and forth, judging rates of closure and the changing positions on the screen as the camera pans and zooms. The colored zones give a nice visual cue and allow viewers to take in additional aspects of the game.
# 

# Here’s another example where zone partitioning shows what’s happening. In this play, NFL veteran Andre Roberts enters the screen from the right as he returns a kick. You can see in the video below how Roberts moves to his left, runs through a mass of players, and then cuts back right to break away. Did his choice make sense?

# In[ ]:


from IPython.display import Video

display(Video("https://github.com/JohnM-TX/misc/raw/main/chargers_coach_clip.mp4", width=900))


# Zone segmentation suggests that the choice made sense. In the picture below, the left frame shows the overhead view. Roberts, shown with the higher blue underline, has more room to his left than what appeared before. And his blockers have set up a nice seam for him to exploit. Looking at the original sideline view on the right with zone segmentation, we see a path very similar to the one Roberts actually took.
# 
# 
# <img src="https://github.com/JohnM-TX/misc/raw/main/chargers_combo.png" width="1000">

# <a id='bg'></a>
# <div class="h2">Methodology</div>

# We used the following methodology to create on-screen effects and play statistics:
# 
# 1. Find players in the image using Object Detection
# 2. Match player coordinates in the image to x-y locations and player identities provided by Next Gen Stats.
# 3. Produce the zones accessible to each player, properly adjusted for the camera’s perspective
# 
# Here are the details for each step.

# #### Step 1.	Find players in the image using Object Detection
#   -	We start by feeding in the video and processing each frame separately. NFL video is filmed at 60 frames per second, and we typically used clips of around 8-seconds, which is about 500 frames per clip. 
#   -	Identify players on the field and their precise location at the plane of the playing field. We use a Yolo (You Only Look Once) deep learning model trained on detecting people and faces in crowds.  [2]
#   -	Identify helmets on the field using a second Yolo model. This model is trained using NFL data provided for the Helmet Assignment challenge. The model detects helmets in the image and classifies them as on-field helmets or sideline helmets.  [3]
#   -	Calculate the amount of overlap between bounding boxes of players and helmets and choose the most likely player locations.
# 
# 
# <img src="https://github.com/JohnM-TX/misc/raw/main/yolos.png" width="800">
# 
# 
# The output of this stage is a table of player locations with x-y coordinates of the boxes that outline each player in the given frame. The table also contains bounding box coordinates for each player’s helmet, which we use as an approximate baseline for the next step. 

# In[ ]:


import datatable as dt

dt.fread('../input/big-data-bowl-2022-jm-us/person_df_all_cowboys_steelers.csv')[:7, :]


# #### Step 2.	Match player coordinates in the image to x-y locations provided by Next Gen Stats. 
# 
# Here we match image data and NGS tracking data using a series of models developed by Kippei Matsuda, winner of the NFL Helmet Assignment challenge. [4]
#   -	Use baseline helmet locations to provide an interpolated version of the tracking data. NGS tracking data registers locations 10 times per second and so we fill in the blanks to match the video data which has locations 60 times per second.
#   -	Convert helmet locations from a frame in image space to a 2-d overhead perspective. This is done using a Convolutional Neural Network (CNN) trained to predict x-y locations based on the camsera perspective for each frame. This step provides a translation matrix to be used for image dewarping as shown earlier.
#   -	Match the converted helmet locations with a subset of the x-y points listed in the tracking data. An iterative method is used to match the points. The tracking data is used as the true locations for players and fills in any players hidden or missing up to this point.
#   -	Apply player tracking through the series of frames and reassign players as needed. Tracking is done by matching closest bounding boxes from one frame to the next with a intersection-over-union metric.
# 
# The outputs of this step are the updated coordinates, player identities and speeds, and the transformation matrix used for image dewarping. The figure below from Matsuda shows the iterative method used to match player location.
# 
# <img src="https://github.com/JohnM-TX/misc/raw/main/step2_matsuda.png" width="700">

# #### Step 3.	Produce the zones accessible to each player, properly adjusted for the camera’s perspective. 
#   - Take each frame and apply the transformation matrix calculated in the previous step to create an array representing the dewarped image. Use precalculated transformation matrices as a backup or if processing speed is a concern.
#   - Use the dewarped array to calculate Voronoi partitions. We use fast GPU-enabled code to apply regions directly onto the image array. Player partitions are then mapped by team to produce team partitions.
#   - Reverse the perspective transformation to restore the original camera perspective. The result is the original image array with an overlay of space controlled by each team.
# 
# The figure below shows a simplified example of creating the zones. 
# 
# <img src="https://github.com/JohnM-TX/misc/raw/main/step3.png" width="800">
# 
# 
# Finally, we assemble the iamge frames back into a video at the original resolution and frame rate. 
# 
# The appendix contains the code we used to implement the methodology.

# <a id='bg'></a>
# <div class="h2">Additional Analysis</div>

# In this section we present additional dimensions of analysis possible with video zone segmentation.
# 
# **Field control percentage**
# 
# So far we have talked mostly about individual plays and the benefits to viewers of all levels watching  a game. Coaches, players, other pros, and avid fans are also interested in statistics and analysis that show trends and correlations. The chart below shows field control percentage for a punt return. In this case, the play design was based on drawing the kicking team to one side of the field and then passing the ball laterally to the other side. The advantage of such a play is that the kicking team might control the largest part of the field, but it will be the part without the ball. The relative percentages are as interesting as the absolutes, if not more so. You can see how how the field percentage changed as the play progessed and correlate it with the video (link below). 

# In[ ]:


# Use labeled chart instead

# from bokeh.models.formatters import NumeralTickFormatter
# import hvplot.pandas

# ctrl_df = dt.fread('../input/big-data-bowl-2022-jm-us/ctrl_df.csv').to_pandas()
# chart_opts = dict(
#     xlabel='Video Time (s)',
#     ylabel='Home Team Field Control Pct',
#     yformatter=NumeralTickFormatter(format='0 %'),
#     yticks = 5,
# )
# ctrl_df.hvplot(x='clip_time', y='home_ctrl', kind='line', **chart_opts)


# <img src="https://github.com/JohnM-TX/misc/raw/main/lineplot_combo.png" width="700">
# 
# A video of the play can be seen at <a href="https://www.youtube.com/watch?v=kn8VaYulTCU" target="_blank">Cowboys Punt Return</a>, courtesy of the NFL.
# 
# <br>
# Charts like this one could be helpful as a team watches game video and discusses their execution. The charts and segmented videos could also be used by players as they review the game individually and reflect on multiple plays. It's important to note that metrics like Fiel Control Percentage should be used in context. Football is a complex game and it wouldn't make sense to compare these numbers across teams or games. We think it works more as a quantitative insight to supplement the analog data from video and player experience. 
# 

# **Combinations with other elements of NGS data**
# 
# Any element captured by NGS data can be matched to video at the proper time and place using the method above. Speed of the ball carrier is one that might be useful for kickoff and punt returns, especially for longer runs. Speed and direction are also useful for velocity-adjusted Voronoi partitions. [5] Other metrics such as distances between specific players or time elapsed between events can also be valuable when viewed in the context of video. Super Bowl champion Usama Young provides excellent insight in a 2022 Big Data Bowl film session. He talks aobut how players on a punt have to disperse to their lanes of responsibility and play off their partners at the best interval to contain a returner. [6] 

# <a id='bg'></a>
# <div class="h2">Conclusion</div>

# Implementing zone segmentation would likely make the most sense for distribution channels such as NFL Game Pass, NFL Red Zone, and other post-game applications. In our experience it takes about 3-4 minutes to process a 10-second video clip, Post-processing for accuracy and applying the segmentation to relevant plays would take additional time.
# 
# We believe that the use of zone segmentation would be a useful addition to the NFLs suite of AR tools. It can contribute to fan engagement and provide additional understanding when viewing special teams plays, especially with a sideline perspective.  It can also provide players and coaches with additonal information in context when reviewing game video.

# <img src="https://github.com/JohnM-TX/misc/raw/main/combo4_adobe.gif" width="700">

# <a id='bg'></a>
# <div class="h2">Appendix</div>

# #### About the authors
# 
# John Miller is a Customer Data Scientist at H2O.ai. He is currently helping a large hospital system use AI to provide better healthcare. John is a devoted Kaggler and sports fan. He has presented models and findings in previous Kaggle challenges to the NFL and MLB. 
# 
# Uri Smashnov also works as a Customer Data Scientist at H2O.ai. He is currently helping several customers in finance and telecom to use AI to solve a wide variety of use cases. Uri is a football and MMA fan and enjoys the athletics and strategy of the football game. Uri has many friends and relatives living abroad, and has witnessed increased interest in NFL football outside of the US.

# #### Citations
# 
# [1] *A new reality: Fan perceptions of augmented reality readiness in sport marketing*
# 
# www.sciencedirect.com/science/article/abs/pii/S0747563219304509
# 
# [2] *CrowdHuman: A Benchmark for Detecting Human in a Crowd*
# 
# crowdhuman.org
# 
# [3] *NFL_yolov5_helmet_detection_models* 
# 
# www.kaggle.com/georgeteo89/nfl-yolov5-helmet-detection-models
# 
# [4] *NFL Health & Safety - Helmet Assignment*
# 
# www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/284975
# 
# 
# [5] *Wide Open Spaces: A statistical technique for measuring space creation in professional soccer*
# 
# http://www.lukebornn.com/papers/fernandez_ssac_2018.pdf
# 
# [6] *2022 Big Data Bowl film session -- Usama Young on punt plays*
# 
# https://www.youtube.com/watch?v=gpfDQ0tNUUg&t=2193s
# 

# #### Implementation
# 
# Our code is included in the attached dataset as vid_seg_pipeline.py. We made the code modular to the extent possible in the interest of readability and future changes.

# #### Other
# 
# Devin Hester doing his thing.
# 
# <img src="https://github.com/JohnM-TX/misc/raw/main/devin-hester-qa.jpg" width="700">

#     
