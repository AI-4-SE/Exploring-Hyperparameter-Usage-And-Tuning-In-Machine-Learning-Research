#!/usr/bin/env python
# coding: utf-8

# # <b>1 <span style='color:#23a9f7'>|</span> Introduction</b>
# ![](https://www.hcube.io/files/plotly-logo-01-stripe.png)
# 
# ### What to Expect?
# In this notebook, I'm gonna be showing techniques on how to visualize categorical data using Plotly - a tutorial
# 
# ### What is Plotly?
# Plotly's Python graphing library makes interactive, publication-quality graphs. Examples of how to make line plots, scatter plots, area charts, bar charts, error bars, box plots, histograms, heatmaps, subplots, multiple-axes, polar charts, and bubble charts. - [Plotly](https://plotly.com/python/)
# 
# ### What do you mean by Categorical Data?
# Categorical variables represent types of data which may be divided into groups. Examples of categorical variables are race, sex, age group, and educational level. While the latter two variables may also be considered in a numerical manner by using exact values for age and highest grade completed, it is often more informative to categorize such variables into a relatively small number of groups. - [Yale](http://www.stat.yale.edu/Courses/1997-98/101/catdat.htm)<br><br>
# 
# ##### So, without further ado, lets get started

# # <b>2 <span style='color:#23a9f7'>|</span> Importing Libraries</b>

# In[ ]:


import numpy as np 
import pandas as pd 

from plotly.offline import iplot, init_notebook_mode
import plotly.express as px

init_notebook_mode(connected=True)

tips = px.data.tips()


# # <b>3 <span style='color:#23a9f7'>|</span> Stripplot</b>
# A strip plot is a graphical data anlysis technique for summarizing a univariate data set. - [NIST](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/striplot.htm)

# In[ ]:


fig = px.strip(tips, x='day', y='tip', color='day')
fig.show()


# # <b>4 <span style='color:#23a9f7'>|</span> Boxplot</b>
# A box plot (or box-and-whisker plot) shows the distribution of quantitative data in a way that facilitates comparisons between variables or across levels of a categorical variable. - [Seaborn](https://seaborn.pydata.org/generated/seaborn.boxplot.html)

# In[ ]:


fig = px.box(tips, x="time", y="total_bill")
fig.show()


# ### Display the underlaying data

# In[ ]:


fig = px.box(tips, x="time", y="total_bill", points='all')
fig.show()


# ### Adding another varible to compare

# In[ ]:


fig = px.box(tips, x="time", y="total_bill", color='smoker')
fig.show()


# ### Styled box plot

# In[ ]:


fig = px.box(tips, x="time", y="total_bill", color="smoker",
             notched=True, # used notched shape
             title="Box plot of total bill"
            )
fig.show()


# # <b>5 <span style='color:#23a9f7'>|</span> Violin Plot</b>
# A violin plot depicts distributions of numeric data for one or more groups using density curves. The width of each curve corresponds with the approximate frequency of data points in each region. Densities are frequently accompanied by an overlaid chart type, such as box plot, to provide additional information. - [Chartio](https://chartio.com/learn/charts/violin-plot-complete-guide/)

# In[ ]:


fig = px.violin(tips, x='time', y="total_bill")
fig.show()


# ### Including the data points

# In[ ]:


fig = px.violin(tips, x='time', y="total_bill", points='all')
fig.show()


# ### Adding boxplot

# In[ ]:


fig = px.violin(tips, x='time', y="total_bill", points='all', box=True)
fig.show()


# ### Adding another variable

# In[ ]:


fig = px.violin(tips, x='time', y="total_bill", points='all', box=True, color='sex')
fig.show()


# ### Overlay the violin plots

# In[ ]:


fig = px.violin(tips, y="tip", violinmode='overlay', color='smoker')
fig.show()


# # <b>6 <span style='color:#23a9f7'>|</span> Histogram</b>
# A histogram is a graphical representation that organizes a group of data points into user-specified ranges. - [Investopedia](https://www.investopedia.com/terms/h/histogram.asp#:~:text=A%20histogram%20is%20a%20graphical,into%20logical%20ranges%20or%20bins.)

# In[ ]:


fig = px.histogram(tips, x="day", category_orders=dict(day=["Thur", "Fri", "Sat", "Sun"]), color='day')
fig.show()


# ### Adding another variable to compare

# In[ ]:


fig = px.histogram(tips, x="day", color='time', category_orders=dict(day=["Thur", "Fri", "Sat", "Sun"]))
fig.show()


# # <b>7 <span style='color:#23a9f7'>|</span> Conclusion</b>
# Cool! After you've made all those graphs, hopefully you got the hang of it and got the idea on how Plotly works! Now its your turn and using Plotly to your exploratory data analysis so that it would look pretty and it will be interactive!

# # <b>8 <span style='color:#23a9f7'>|</span> Authors Message</b>
# * If you find this helpful, I would really appreciate the upvote!
# * If you see something wrong please let me know.
# * And lastly Im happy to hear your thoughts about the notebook for me to also improve!
