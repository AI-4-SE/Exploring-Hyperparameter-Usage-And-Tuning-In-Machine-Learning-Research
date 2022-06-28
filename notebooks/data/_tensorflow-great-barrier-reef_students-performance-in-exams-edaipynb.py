#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

sns.set_style("darkgrid")


# In[ ]:


data=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# # math score:

# In[ ]:


f,(ax_box, ax_hist) = plt.subplots(2,figsize=(16,8),sharex=True,gridspec_kw={"height_ratios": (.50, .85)})
sns.boxplot(x='math score',data=data,color='#ff1a1a',ax=ax_box)
sns.histplot(data = data,x = 'math score',bins=20,edgecolor='black',color='#ff1a1a',kde=True,ax=ax_hist)
plt.title('Math score data distribution',color='black',size=25)
plt.show()


# ###  **`Observation:-`  data is skewed to the left, that means mean is less than the median**
# ### **As you can see in math score have some outlires.**
# ### Mean value is: 66

# # reading score:

# In[ ]:


f,(ax_box, ax_hist) = plt.subplots(2,figsize=(16,10),sharex=True,gridspec_kw={"height_ratios": (.50, .85)})
sns.boxplot(x='reading score',data=data,color='#0000ff',ax=ax_box)
sns.histplot(data = data,x = 'reading score',bins=20,edgecolor='black',color='#0000ff',kde=True,ax=ax_hist)
plt.title('Reading score data distribution',color='black',size=25)
plt.show()


# ### **`Observation:-` As you can see in reading score have some outlires.**
# ### Mean value is: 69

# In[ ]:


data['reading score'].mean()


# # writing score:

# In[ ]:


f,(ax_box, ax_hist) = plt.subplots(2,figsize=(16,10),sharex=True,gridspec_kw={"height_ratios": (.50, .85)})
sns.boxplot(x='writing score',data=data,color='#cc0066',ax=ax_box)
sns.histplot(data = data,x = 'writing score',bins=20,edgecolor='black',color='#cc0066',kde=True,ax=ax_hist)
plt.title('Writing score data distribution',color='black',size=25)
plt.show()


# In[ ]:


data['writing score'].mean()


# ### **`Observation:-` As you can see in writing score have some outlires.**
# ### Mean value is: 68

# In[ ]:


f,ax=plt.subplots(1,3,figsize=(20,10))

sns.violinplot(x=data['math score'], data=data,split=True,ax=ax[0],color='#ff0000')
ax[0].set_title('math score data distribution',color='black',size=20)
sns.violinplot(x=data['writing score'], data=data,split=True,ax=ax[1],color='#bf4080')
ax[1].set_title('writing score data distribution',color='black',size=20)
sns.violinplot(x=data['reading score'], data=data,split=True,ax=ax[2],color='#6666ff')
ax[2].set_title('reading score data distribution',color='black',size=20)
plt.show()


# # Categorical Variables

# In[ ]:


data.info()


# In[ ]:


data.head()


# # Gender:

# In[ ]:


gender_counts = data['gender'].value_counts()
fig = go.Figure(data=[go.Pie(labels=gender_counts.index, values=gender_counts, opacity=0.9)])
fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
fig.update_layout(title_text='Distribution of the Gender', title_x=0.5, title_font=dict(size=20))
fig.show()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,10))
sns.countplot(x=data['gender'],data=data,palette ='bright',ax=ax[0],saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=20)
    
plt.pie(x=data['gender'].value_counts(),labels=['Male','Female'],explode=[0,0.1],autopct='%1.1f%%',shadow=True,colors=['#ff4d4d','#ff8000'])
plt.show()


# In[ ]:


data['gender'].value_counts()


# ### `Observation:-` As you can see female students are 518 `48%` and male students are 482 `52%`

# In[ ]:


Group_data=data.groupby('gender')
Group_data.get_group('male')


# In[ ]:


print(Group_data['writing score'].mean().index)
print(Group_data['writing score'].mean().values)


# In[ ]:


#here i am passing index for lable and values for value

f,ax=plt.subplots(1,3,figsize=(20,8))
sns.barplot(x=Group_data['math score'].mean().index,y=Group_data['math score'].mean().values,palette = 'mako',ax=ax[0],saturation=0.95)
ax[0].set_title('Math score',color='black',size=20)

for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=20)

sns.barplot(x=Group_data['reading score'].mean().index,y=Group_data['reading score'].mean().values,palette = 'flare',ax=ax[1],saturation=0.95)
ax[1].set_title('Reading score',color='black',size=20)

for container in ax[1].containers:
    ax[1].bar_label(container,color='black',size=20)

sns.barplot(x=Group_data['writing score'].mean().index,y=Group_data['writing score'].mean().values,palette = 'crest',ax=ax[2],saturation=0.95)
ax[2].set_title('Writing score',color='black',size=20)

for container in ax[2].containers:
    ax[2].bar_label(container,color='black',size=20)


# ### **`Observation:-` We can see that male has better performance on math field, but worse on reading and writing. Secondly, see the performance of ethnicity.**

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,10))
sns.countplot(x=data['gender'],data=data,hue='race/ethnicity',palette = 'bright',saturation=0.95,ax=ax[0])
ax[0].set_title('Male and Female from differents Groups ',color='black',size=25)
for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=16)
    
sns.countplot(x=data['gender'],data=data,hue='parental level of education',palette = 'bright',saturation=0.95,ax=ax[1])
ax[1].set_title('Male and Female vs level of education ',color='black',size=25)
for container in ax[1].containers:
    ax[1].bar_label(container,color='black',size=16)    


# ### `Observation:-`
# * ### Most of Female and Male students belonging from Group C and Group D and very less in Group A
# * ### Most of Female and Male students belonging from some collage and associate's degree and very less in master's degree

# # race/ethnicity:

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,10))
sns.countplot(x=data['race/ethnicity'],data=data,palette = 'bright',ax=ax[0],saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=20)
    
plt.pie(x = data['race/ethnicity'].value_counts(),labels=data['race/ethnicity'].value_counts().index,explode=[0.1,0,0,0,0],autopct='%1.1f%%',shadow=True)
plt.show()    


# In[ ]:


data['race/ethnicity'].value_counts()


# ### `Observation:-`As you can see most of the student belonging from Group C, group D and students less in                                          groupA.

# In[ ]:


Group_data2=data.groupby('race/ethnicity')
Group_data2.get_group('group C')


# In[ ]:


Group_data2['math score'].mean()


# In[ ]:


f,ax=plt.subplots(1,3,figsize=(20,8))
sns.barplot(x=Group_data2['math score'].mean().index,y=Group_data2['math score'].mean().values,palette = 'mako',ax=ax[0])
ax[0].set_title('Math score',color='#005ce6',size=20)

for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=15)

sns.barplot(x=Group_data2['reading score'].mean().index,y=Group_data2['reading score'].mean().values,palette = 'flare',ax=ax[1])
ax[1].set_title('Reading score',color='#005ce6',size=20)

for container in ax[1].containers:
    ax[1].bar_label(container,color='black',size=15)

sns.barplot(x=Group_data2['writing score'].mean().index,y=Group_data2['writing score'].mean().values,palette = 'coolwarm',ax=ax[2])
ax[2].set_title('Writing score',color='#005ce6',size=20)

for container in ax[2].containers:
    ax[2].bar_label(container,color='black',size=15)


# ### `Observation:-`
# ### `group E` has best performance for all the fields
# ### `group A` is the worst
# 

# In[ ]:


f,ax=plt.subplots(1,1,figsize=(16,10))
sns.countplot(x=data['race/ethnicity'],data=data,palette = 'flare',hue='gender',saturation=0.95)
ax.set_title('race/ethnicity vs Male and Female:',color='black',size=25)
for container in ax.containers:
    ax.bar_label(container,color='black',size=20)


# ### `Observation:-` majority students present in Group C and Group D

# In[ ]:


data[['race/ethnicity','parental level of education']].value_counts().sort_index()


# In[ ]:


f,ax=plt.subplots(1,1,figsize=(16,10))
sns.countplot(x=data['race/ethnicity'],data=data,palette = 'Blues',hue='parental level of education',saturation=0.95)
ax.set_title('race/ethnicity vs parental level of education:',color='black',size=25)
for container in ax.containers:
    ax.bar_label(container,color='black',size=20)
plt.show()


# In[ ]:


f,ax=plt.subplots(1,1,figsize=(16,10))
sns.countplot(x=data['race/ethnicity'],data=data,palette = 'ch:s=.25,rot=-.25',hue='lunch',saturation=0.95)
for container in ax.containers:
    ax.bar_label(container,color='black',size=20)


# In[ ]:


f,ax=plt.subplots(3,1,figsize=(20,16))
sns.boxplot(data =data, x='race/ethnicity', y='math score',ax=ax[0],palette='flare',saturation=0.95)
ax[0].set_title('Math & Reading & Writing score respect to race/ethnicity ',color='black',size=25)
sns.boxplot(data =data, x='race/ethnicity', y='reading score',ax=ax[1],palette='light:#5A9',saturation=0.95)
sns.boxplot(data =data, x='race/ethnicity', y='writing score',ax=ax[2],palette='ch:s=.25,rot=-.25',saturation=0.95)
plt.show()


# # parental level of education:

# In[ ]:


data['parental level of education'].unique()


# In[ ]:


f,ax=plt.subplots(1,1,figsize=(16,10))
sns.countplot(y=data['parental level of education'],data=data,palette = 'OrRd',saturation=0.95)
ax.set_title('race/ethnicity vs Male and Female:',color='black',size=25)
for container in ax.containers:
    ax.bar_label(container,color='black',size=20)


# In[ ]:


f,ax=plt.subplots(3,1,figsize=(25,15))
sns.violinplot(x=data['parental level of education'], y=data['math score'],data=data,split=True,palette='flare',saturation=0.95,ax=ax[0])
ax[0].set_title('Math & Reading & Writing score respect to level of education ',color='black',size=25)
sns.violinplot(x=data['parental level of education'], y=data['reading score'],data=data,split=True,palette='flare',saturation=0.95,ax=ax[1])
sns.violinplot(x=data['parental level of education'], y=data['writing score'],data=data,split=True,palette='flare',saturation=0.95,ax=ax[2])


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,8))
sns.countplot(x=data['parental level of education'],data=data,palette = 'bright',hue='test preparation course',saturation=0.95,ax=ax[0])
ax[0].set_title('Students vs test preparation course ',color='black',size=25)
for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=20)
    
sns.countplot(x=data['parental level of education'],data=data,palette = 'bright',hue='lunch',saturation=0.95,ax=ax[1])
for container in ax[1].containers:
    ax[1].bar_label(container,color='black',size=20)    


# ### `Observation:-`Most students did not take the test preparation course.

# # Relationship:

# In[ ]:


data['lunch'].unique()


# In[ ]:


#sns.pairplot(data, hue ='race/ethnicity',height=5)
sns.pairplot(data, hue ='parental level of education',height=5,palette = 'bright')


# In[ ]:


sns.heatmap(data.corr(),annot=True,cmap='icefire',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(16,10)
#fig.title('corelation between math, reading and writing scores')
plt.title('Corelation between math, reading and writing scores',color='black',size=25)
plt.show()


# ## As you can see reading and writing score highly corelated

# In[ ]:


plt.figure(figsize = [10,7])
fig,axes = plt.subplots(2,2, figsize = [18,10])
axes[0,0].set_title('Relationship between math and reading score',color='black',size=25)
sns.scatterplot(x = 'math score',y = 'reading score',hue = 'parental level of education',data = data, ax =axes[0,0],palette = 'deep')
sns.scatterplot(x = 'math score',y = 'reading score',hue = 'race/ethnicity',data = data, ax =axes[0,1],palette = 'bright')
sns.scatterplot(x = 'math score',y = 'reading score',hue = 'lunch',data = data, ax =axes[1,0],palette = 'dark')
sns.scatterplot(x = 'math score',y = 'reading score',hue = 'test preparation course',data = data, ax =axes[1,1],palette = 'pastel')


# In[ ]:


plt.figure(figsize = [10,7])
fig,axes = plt.subplots(2,2, figsize = [18,10])
axes[0,0].set_title('Relationship between math and writing score',color='black',size=25)
sns.scatterplot(x = 'math score',y = 'writing score',hue = 'parental level of education',data = data, ax =axes[0,0],palette = 'deep')
sns.scatterplot(x = 'math score',y = 'writing score',hue = 'race/ethnicity',data = data, ax =axes[0,1],palette = 'bright')
sns.scatterplot(x = 'math score',y = 'writing score',hue = 'lunch',data = data, ax =axes[1,0],palette = 'dark')
sns.scatterplot(x = 'math score',y = 'writing score',hue = 'test preparation course',data = data, ax =axes[1,1],palette = 'pastel')


# In[ ]:


fig = px.scatter(data, x="math score", y="writing score", trendline="ols",
                 labels={"math score": "Math Score",
                         "writing score": "Writing Score"})
fig.update_layout(title_text='Relationship between Math and Writing Scores',
                  title_x=0.5, title_font=dict(size=20))
fig.data[1].line.color = 'red'
fig.show()


# In[ ]:


fig = px.scatter(data, x="math score", y="reading score", trendline="ols",
                 labels={"math score": "Math Score",
                         "reading score": "Reading Score"})
fig.update_layout(title_text='Relationship between Math and reading Scores',
                  title_x=0.5, title_font=dict(size=20))
fig.data[1].line.color = 'red'
fig.show()


# In[ ]:


fig = px.scatter(data, x="writing score", y="reading score", trendline="ols",
                 labels={"writing score": "Writing Score",
                         "reading score": "Reading Score"})
fig.update_layout(title_text='Relationship between writing and reading Scores',
                  title_x=0.5, title_font=dict(size=20))
fig.data[1].line.color = 'red'
fig.show()


# In[ ]:


#sns.scatterplot(data=data, x="math score", y="writing score",trendline="ols")
sns.regplot(x="math score", y="writing score", data=data);


# ## Conclusion:
# * ### Students who completed test preparation courses get higher score.
# * ### The score of female in reading and writing is higher than male, but the score of male in math is higher than female.
# * ### The score of student whose parents possess master and bachelor level education are higher than others.
# * ### The score of students who were provided standard lunch is higher than those who were provided free/reduced lunch.

# In[ ]:




