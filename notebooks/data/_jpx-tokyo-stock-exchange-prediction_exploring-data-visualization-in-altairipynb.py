#!/usr/bin/env python
# coding: utf-8

# ![AltairvsSeaborn.jpg](attachment:06f2c82d-1e4f-4207-a731-3a9ccdb04d82.jpg)

# <p style="font-family: Arials; font-size: 18px;text-align: center;; font-style: normal;line-height:1.3">Seaborn is based on Matplotlib and provides a high-level interface for building informative statistical visualizations. However, there is an alternative to Seaborn. This library is called ‘Altair’, an open-source Python library built for statistical data visualization. According to the official documentation, it is based on the Vega and Vega-lite language. Using Altair we can create interactive data visualizations through bar chart, histogram, scatter plot and bubble chart, grid plot and error chart, etc. similar to the Seaborn plots.<br></p>
# <br>
# <p style="font-family: Arials; font-size: 18px;text-align: center;; font-style: normal;line-height:1.3">In this notebook, we will compare Seaborn to Altair. For this comparison, we will create the same set of visualizations using both libraries.</p>

# <h3 style="font-family: Arial;background-color:#ffc2d1;color:black;text-align: center;padding-top: 5px;padding-bottom: 5px;border-radius: 15px 50px;letter-spacing: 2px;font-size: 20px"><strong><centre>Importing Libraries & Packages 📚 </centre></strong></h3>

# pip install altair 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style("white")
import altair as alt
import warnings
warnings.filterwarnings("ignore")


# <h3 style="font-family: Arial;background-color:#ffc2d1;color:black;text-align: center;padding-top: 5px;padding-bottom: 5px;border-radius: 15px 50px;letter-spacing: 2px;font-size: 20px"><strong><centre>Importing dataset 📝 </centre></strong></h3>

# <p style="font-family: Arials; font-size: 18px;text-align: center;; font-style: normal;line-height:1.3">We will use the ‘mpg’ or the ‘miles per gallon’ dataset from the seaborn dataset library to generate these different plots. This famous dataset contains 398 samples and 9 attributes for automotive models of various brands. </p>

# In[ ]:


df = sns.load_dataset('mpg')
df.shape


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


df.isna().sum()


# <p style="font-family: Arials; font-size: 18px;text-align: center;; font-style: normal;line-height:1.3">This dataset is simple and has a nice blend of both categorical and numerical features. We can now plot our charts for comparison.</p>

# <h3 style="font-family: Arial;background-color:#ffc2d1;color:black;text-align: center;padding-top: 5px;padding-bottom: 5px;border-radius: 15px 50px;letter-spacing: 2px;font-size: 20px"><strong><centre>Scatter Plot 📊</centre></strong></h3>

# # Scatter Plot with Altair

# In[ ]:


alt.Chart(df).mark_point().encode(alt.Y('mpg'),alt.X('horsepower'),alt.Color('origin'),alt.OpacityValue(0.7),size='displacement')


# # Scatterplot with Seaborn

# In[ ]:


sns.relplot(y='mpg',x='horsepower',data=df,kind='scatter',size='displacement',hue='origin',aspect=1.2);


# <h3 style="font-family: Arial;background-color:#ffc2d1;color:black;text-align: center;padding-top: 5px;padding-bottom: 5px;border-radius: 15px 50px;letter-spacing: 2px;font-size: 20px"><strong><centre>Line Chart 📉</centre></strong></h3>

# # Line Chart with Altair

# In[ ]:


alt.Chart(df).mark_line().encode(
    alt.X('horsepower'),
    alt.Y('acceleration'),
    alt.Color('origin')
    )


# # Line Chart with Seaborn

# In[ ]:


sns.lineplot(data=df, x='horsepower', y='mpg',hue='origin')


# <h3 style="font-family: Arial;background-color:#ffc2d1;color:black;text-align: center;padding-top: 5px;padding-bottom: 5px;border-radius: 15px 50px;letter-spacing: 2px;font-size: 20px"><strong><centre>Bar and Count Plots 📊</centre></strong></h3>

# # Bar and Count plots with Altair

# In[ ]:


plot=alt.Chart(df).mark_bar(size=40).encode(
    alt.X('cylinders'),
    alt.Y('mpg'),
    alt.Color('origin')

)
plot.properties(title='cylinders vs mpg')


# In[ ]:


alt.Chart(df).mark_bar().encode(
    x='origin',
    y='count()',
    column='cylinders:Q',
    color=alt.Color('origin')
).properties(
    width=100,
    height=100    
)


# # Bar and Count plots with Seaborn

# In[ ]:


sns.catplot(x='cylinders',y='mpg', hue="origin", kind="bar", data=df,palette='magma_r')


# In[ ]:


g = sns.FacetGrid(df, col="cylinders", height=4, aspect=.5,hue='origin',palette='magma_r')
g.map(sns.countplot, "origin", order = df['origin'].value_counts().index)


# <h3 style="font-family: Arial;background-color:#ffc2d1;color:black;text-align: center;padding-top: 5px;padding-bottom: 5px;border-radius: 15px 50px;letter-spacing: 2px;font-size: 20px"><strong><centre>Histogram 📊</centre></strong></h3>

# # Histogram with Altair

# In[ ]:


alt.Chart(df).mark_bar().encode(
    alt.X("model_year:Q", bin=True),
    y='count()',
).configure_mark(
    opacity=0.7,
    color='cyan'
)


# # Histogram with Seaborn

# In[ ]:


sns.displot(df, x='model_year', aspect=1.2)


# <h3 style="font-family: Arial;background-color:#ffc2d1;color:black;text-align: center;padding-top: 5px;padding-bottom: 5px;border-radius: 15px 50px;letter-spacing: 2px;font-size: 20px"><strong><centre>Strip Plot 📊</centre></strong></h3>

# # Strip plot with Altair

# In[ ]:


alt.Chart(df).mark_tick(filled=True).encode(
    x='horsepower:Q',
    y='cylinders:O',
    color='origin'
)


# # Strip plot with Seaborn

# In[ ]:


sns.set_style("white")
ax = sns.stripplot(y="horsepower", x="cylinders", data=df)


# <h3 style="font-family: Arial;background-color:#ffc2d1;color:black;text-align: center;padding-top: 5px;padding-bottom: 5px;border-radius: 15px 50px;letter-spacing: 2px;font-size: 20px"><strong><centre>Bubble Plot 📉</centre></strong></h3>

# # Bubble plot with Altair

# In[ ]:


alt.Chart(df).mark_point(filled=True).encode(
    x='horsepower',
    y='mpg',
    size='acceleration',
    color='origin'
)


# # Bubble plot with Seaborn

# In[ ]:


sns.set(rc={'figure.figsize':(10,7)})
sns.set_style("white")
sns.scatterplot(data=df, x="horsepower", y="mpg", size="acceleration", hue='origin',legend=True, sizes=(10, 500))


# <h3 style="font-family: Arial;background-color:#ffc2d1;color:black;text-align: center;padding-top: 5px;padding-bottom: 5px;border-radius: 15px 50px;letter-spacing: 2px;font-size: 20px"><strong><centre>Interactive Plot 📉</centre></strong></h3>

# # Interactive plots with Altair

# In[ ]:


select = alt.selection(type='interval')
values = alt.Chart(df).mark_point().encode(
    x='horsepower:Q',
    y='mpg:Q',
    color=alt.condition(select, 'origin:N', alt.value('lightgray'))
).add_selection(
    select
)

bars = alt.Chart(df).mark_bar().encode(
    y='origin:N',
    color='origin:N',
    x='count(origin):Q'
).transform_filter(
    select
)
values & bars


# <p style="font-family: Arials; font-size: 18px;text-align: center;; font-style: normal;line-height:1.3">We saw various types of plots with Seaborn and Altair. Both the data visualization libraries – Seaborn and Altair seem equally powerful. Syntax of Seaborn is a little simpler to write and easier to understand when compared to Altair; whereas data visualizations in Altair seem a little more pleasant and eye-catching when compared to the Seaborn plots. The ability to generate interactive visualizations is another advantage offered by Altair. Therefore, choosing either one of these depends on personal preferences and visualization requirements. Ideally, both libraries are self-sufficient to handle most of the data visualizations requirements. </p>

# 📌  Details of the libraries and model setup are discussed in my article https://www.analyticsvidhya.com/blog/2021/10/exploring-data-visualization-in-altair-an-interesting-alternative-to-seaborn/ published on AnalyticsVidhya.

# <h3 style="font-family: Arial;background-color:#ffc2d1;color:black;text-align: center;padding-top: 5px;padding-bottom: 5px;border-radius: 15px 50px;letter-spacing: 2px;font-size: 20px"><strong><centre>If you found this notebook useful, please Upvote. Thanks!  </centre></strong></h3>
