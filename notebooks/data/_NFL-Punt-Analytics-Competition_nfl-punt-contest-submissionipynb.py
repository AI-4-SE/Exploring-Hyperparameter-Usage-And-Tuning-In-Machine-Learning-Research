#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
I have used only spreadsheet tools and website publicly available tools to do a simple analysis

Also, this is the first contest in which I am actively participating and therefore creating a kernel

So, I am just going to use these comments of this this kernel to explain my analysis.

Concussion Frequency
37 concussions
6584 games (post season excluded due to missing concussion data)
1 in every 178 games
0.56% chance in each game

Pre-Season
12 Pre-Season
736 games
1 in 61 games, 1.6% chance in each game

Regular Season
25 Regular Season
5848 games
1 in 234 games, 0.43% chance in each game

So it seems pretty clear that there is substantially higher probability of consussion in pre-season games.
As such, my recommendations will center on that issue.
But first, what about the statistical significance of the difference in probability bwtween pre-season
and regular-season? To address this I use a z score calculator as shown below:

https://www.socscistatistics.com/tests/ztest/Default2.aspx

Z Score Calculator for 2 Population Proportions
Success!

You'll find the values for z and p below. Blue means your result is significant, red means it's not.

Sample 1 Proportion (or total number)
12

Sample 1 Size (N1)
736

Sample 2 Proportion (or total number)
25

Sample 2 Size (N2)
5848

Significance Level:
0.01 chosen
0.05
0.10

One-tailed or two-tailed hypothesis?:
One-tailed
Two-tailed chosen

The value of z is 4.1144. The value of p is < .00001. The result is significant at p < .01.

So the above seems to indicate that the difference is statistically significant, assuming
I used the tool properly.  That is, the possibility of this much difference has a less than 1% chance
of occuring randomly.

So I have chosen to base my recommendations on this one finding.

I don't see a way to reliably evaluate the efficacy of rule changes so I am instead recommending 
to the NFL that they consider using pre-season games as a test-bed for potential rule changes.  
That is, whatever rule changes are highest ranked (along with variations thereof) could potentially
be tested during the pre-season, where the like would have a magnified effect, allowing evaluation 
after fewer games played.
"""

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



# In[ ]:




