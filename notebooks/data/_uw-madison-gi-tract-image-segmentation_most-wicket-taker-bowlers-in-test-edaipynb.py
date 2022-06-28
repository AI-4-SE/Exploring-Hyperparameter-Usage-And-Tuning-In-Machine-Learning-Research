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


data=pd.read_csv('../input/most-wickets-in-test-cricket/Most Wickets in Test Cricket .csv')


# <div class="alert alert-block"; style= "background-color:#f2f2f2;">
# <h1 style="color:#0099ff; font-size:40px">Quick Access Link:</h1> 
#     <br></br>
#     <a href='#1' style="font-size:20px; color:#ff3300;margin-left:100px">1. Most wicket taker bowler respect to country:</a>
#      <br></br>
#     <a href='#2' style="font-size:20px; color:#ff3300;margin-left:100px">2. Most wicket taker player:</a>
#     <br></br>
#     <a href='#3' style="font-size:20px; color:#ff3300;margin-left:100px">3. Most Matches played by player:</a>
#     <br></br>
#     <a href='#4' style="font-size:20px; color:#ff3300;margin-left:100px">4. Number of Balls throw by bowler:</a>
#     <br></br>
#     <a href='#5' style="font-size:20px; color:#ff3300;margin-left:100px">5. Most Economical bowler:</a>
#     <br></br>
#     <a href='#6' style="font-size:20px; color:#ff3300;margin-left:100px">6. Most 5 and 10 wickets in a match:</a>
#     <br></br>
#     <a href='#7' style="font-size:20px; color:#ff3300;margin-left:100px">7. Matches vs wickets:</a>
#     <br></br>
#     <a href='#8' style="font-size:20px; color:#ff3300;margin-left:100px">8. Balls vs wicket:</a>
#     <br></br>
#     <a href='#9' style="font-size:20px; color:#ff3300;margin-left:100px">9. Total wickets Country Wise:</a>
#     <br></br>
#     <a href='#10' style="font-size:20px; color:#ff3300;margin-left:100px">10. Conclusion:</a>
# </div>
# 

# 
# <div class="alert alert-block alert-danger" id='1'>
# <h1 style="color:blue; font-size:30px">1. Most wicket taker bowler respect to country:</h1> 
# </div>

# In[ ]:


f,ax=plt.subplots(1,1,figsize=(25,10))
sns.countplot(x='Country ',data=data,palette ='bright',order=data['Country '].value_counts().index,saturation=0.95)
for container in ax.containers:
    ax.bar_label(container,color='black',size=20)

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=['Sri Lanka ', 'Australia ', 'England ', 'India ', 'West Indies','South Africa', 'New Zealand ', 'Pakistan ', 'Zimbabwe','Bangladesh '], values=data['Country '].value_counts(), hole=.3)])
fig.show()    
    


# In[ ]:


data['Country '].value_counts()


# In[ ]:


Group_data=data.groupby('Country ')
Group_data.get_group('Australia ')


# 
# <div class="alert alert-block alert-danger" id ='2'>
# <h1 style="color:blue; font-size:30px;">2. Most wicket taker player:</h1> 
# </div>

# In[ ]:


Group_data['Player '].head(2).values


# In[ ]:


Group_data['Wickets '].head(2).values


# In[ ]:


df = px.data.tips()
fig = px.bar(Group_data, x=Group_data['Player '].head(2).values, y=Group_data['Wickets '].head(2).values,
             color=Group_data['Wickets '].head(2),
             labels={'y':'Wickets','x':'Player'}, height=600)
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.show()


# In[ ]:


df = px.data.tips()
fig = px.bar(Group_data, x=Group_data['Wickets '].head(1).values, y=Group_data['Player '].head(1).values, color=Group_data['Country '].head(1).values, orientation='h',
             height=600,labels={'y':'Players','x':'Wickets'})
fig.show()


# In[ ]:


fig = go.Figure(go.Funnel(
    y = data['Player '].values,
    x = data['Wickets '].values))

fig.show()


# ### `Observation:-` Most wicket taker bowler is M. Muralidaran.

# ![gettyimages-925950352-612x612.jpg](attachment:gettyimages-925950352-612x612.jpg)

# ### `Observation:- `Second wicket taker bowler is S. Warne.

# ![gettyimages-669844678-612x612.jpg](attachment:gettyimages-669844678-612x612.jpg)

# In[ ]:


Group_data['Matches'].max().sort_values(ascending=False)


# 
# <div class="alert alert-block alert-danger" id ='3'>
# <h1 style="color:blue; font-size:30px;">3. Most Matches played by player:</h1> 
# </div>

# In[ ]:


Group_data['Matches'].head(2).values


# In[ ]:


#Group_data['Matches'].max().sort_values(ascending=False)
Group_data['Player '].head(2).values


# In[ ]:


df = px.data.tips()
fig = px.bar(Group_data, x=Group_data['Player '].head(1).values, y=Group_data['Matches'].head(1).values,
             color=Group_data['Matches'].head(1),labels={'x':'Players','y':'Matches'},height=600,color_continuous_scale=px.colors.sequential.Inferno)
fig.show()


# ### `Observation:-` Most matches played player JM Anderson

# ![gettyimages-933778864-612x612.jpg](attachment:gettyimages-933778864-612x612.jpg)

# 
# <div class="alert alert-block alert-danger" id ='4'>
# <h1 style="color:blue; font-size:30px;">4. Number of Balls throw by bowler:</h1> 
# </div>

# In[ ]:


f,ax=plt.subplots(1,1,figsize=(30,16))
sns.barplot(x=Group_data['Player '].head(2).values, y=Group_data['Balls '].head(2).values,palette = 'bright',saturation=0.95)
for container in ax.containers:
    ax.bar_label(container,color='black',size=20)


# ### `Observation:-` Most Balls throw by bowler is M. Muralidaran

# <div class="alert alert-block alert-danger" id ='5'>
# <h1 style="color:blue; font-size:30px;">5. Most Economical bowler:</h1> 
# </div>

# In[ ]:


df = px.data.tips()
fig = px.bar(Group_data, x=Group_data['Player '].head(1).values, y=Group_data['Econ'].head(1).values,
             color=Group_data['Econ'].head(1),
             labels={'y':'Economical','x':'Player'}, height=600,color_continuous_scale=px.colors.sequential.Magenta)
fig.show()


# ### `Observation:-` Most Economical bowler is M. Muralidaran.

# <div class="alert alert-block alert-danger" id ='6'>
# <h1 style="color:blue; font-size:30px;">6. Most 5 and 10 wickets in a match:</h1> 
# </div>

# In[ ]:


df = px.data.tips()
fig = px.bar(Group_data, y=Group_data['Player '].head(2).values, x=Group_data['5'].head(2).values,
             color=Group_data['5'].head(2),
             labels={'x':'5 Wickets','y':'Player'}, height=600,color_continuous_scale=px.colors.sequential.Jet)
fig.show()


# In[ ]:


df = px.data.tips()
fig = px.bar(Group_data, x=Group_data['Player '].head(2).values, y=Group_data['10'].head(2).values,
             color=Group_data['10'].head(2),
             labels={'y':'10 Wickets','x':'Player'}, height=600,color_continuous_scale=px.colors.sequential.Blackbody)
fig.show()


# ### `Observation:-` Most 5 and 10 wickets taken in a match by M. Muralidaran.

# ![Muttiah-Muralitharan.jpg](attachment:Muttiah-Muralitharan.jpg)

# 
# <div class="alert alert-block alert-danger" id ='7'>
# <h1 style="color:blue; font-size:30px;">7. Matches vs wickets:</h1> 
# </div>

# In[ ]:


df = px.data.iris()
fig = px.scatter(data, x='Matches', y='Wickets ', size='Wickets ',color='Wickets ')
fig.show()


# In[ ]:


f,ax=plt.subplots(1,1,figsize=(18,10))
#sns.scatterplot(data["Matches"], data["Wickets "],hue=data['Country '])
sns.lineplot(data['Matches'],data['Wickets '], hue =data["Country "])


# In[ ]:


fig = px.scatter(data, x="Matches", y="Wickets ", trendline="ols",
                 labels={"Matches": "Matches",
                         "Wickets ": "Wickets"})
fig.update_layout(title_text='Relationship between Matches and Wickets',
                  title_x=0.5, title_font=dict(size=20))
fig.data[1].line.color = 'red'
fig.show()


# ### `Observation:-` As you can see whosoever played more matches he gets more wickets.

# 
# <div class="alert alert-block alert-danger" id ='8'>
# <h1 style="color:blue; font-size:30px;">8. Balls vs wicket:</h1> 
# </div>

# In[ ]:


df = px.data.gapminder()
fig = px.line(data, x='Balls ', y='Wickets ', color='Player ', markers=True)
fig.show()


# In[ ]:


px.defaults.color_continuous_scale = px.colors.sequential.Blackbody
df = px.data.iris()
fig = px.scatter(data, x='Balls ', y='Wickets ', size='Wickets ',color='Wickets ',color_continuous_scale=px.colors.sequential.Jet)
fig.show()


# <div class="alert alert-block alert-danger" id ='9'>
# <h1 style="color:blue; font-size:30px;">9. Total wickets Country Wise:</h1> 
# </div>

# In[ ]:


df = px.data.tips()
fig = px.ecdf(data, x="Matches", y='Wickets ',color='Country ',ecdfnorm=None)
fig.show()


# In[ ]:


df = px.data.gapminder()
fig = px.sunburst(data, path=['Country ','Player ','Matches', 'Wickets '], values='Wickets ',
                  color='Wickets ', hover_data=['Player '],height=800)
fig.show()


# 
# <div class="alert alert-block alert-danger" id ='10'>
# <h1 style="color:blue; font-size:30px;">10. Conclusion:</h1> 
# </div>

# ### Eleven years ago, Muttiah Muralitharan became the first and till now the only man to have picked up 800 wickets in Test cricket. While the saying goes that records are meant to be broken, this particular achievement of Murali may be a feat too high for anyone to achieve.

# ![gettyimages-103030214-612x612.jpg](attachment:gettyimages-103030214-612x612.jpg)

# ![thank-you-words-written-vintage-letterpress-type-34621434.jpg](attachment:thank-you-words-written-vintage-letterpress-type-34621434.jpg)
