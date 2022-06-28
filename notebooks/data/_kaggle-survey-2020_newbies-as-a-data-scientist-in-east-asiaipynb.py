#!/usr/bin/env python
# coding: utf-8

# 
# ## Newbie as a data scientist in East Asia!
# 
# Hello, Kaggers! Nice to meet you! 
# 
# We are a team in East Asia that wants to be **data scientists** 
# 
# As newbies, we want to know what and/or how Kaggler is!
# 
# so, let's have a time to learn about Kaggle as a senior with us from now.
# 
# If you want to support us*(or feel qute)*, I ask for a comment! (PLZ) ^0^
# 
# And !! Since we are **not native English speakers**, please ask questions if there is a context that you don't understand because it's not smooth.
# 
# I'll do my best to answer.
# 
# 
# # 1 Introduction
# 1. what is the Kaggle
# a subsidiary of **Google LLC**, is an online **community of data scientists and machine learning practitioners**.
# 
# If we use kaggle, we can take the following advantages.
# 
#     1) to find and publish data sets
#     2) to explore and build models in a web-based data-science environment
#     3) to work with other data scientists and machine learning engineers
#     4) to enter competitions to solve data science challenges
#     
# so, As data scientist beginners, we try to participate in the Kaggle competition.
# 
# ---
# 
# 2. **21 Kaggle** Machine Learning and Data Science Survey
# - The most comprehensive dataset available for ML and data science status
# 
# This is the theme of the competition we will participate in this time.
# 
# To become a data scientist, we compared what kind of job Kagglers has, how much experience he has, and how much money he earns by dividing into the world and East Asia.
# 
# In addition, there are detailed comparisons in East Asia, and ultimately, we will to find out what data the Kaggle competition data shows.
# 
# The 2021 survey, like 2017, 2018, 2019, and 2020, launched an industry-wide survey that comprehensively presents the current status of data science and machine learning.
# 
# The survey was conducted from 09/01/2021 to 10/04/2021, and after cleaning the data, Kaggle received 25,973 responses!
# 
# This year, Kaggle will award $30,000 in prize money to winner in this competition.
# 
# we want to receive $30,000 for winning the competition, but we just hope it will help us become a *data scientist* because it is difficult for a rookie.
# 
# 
# 
# Ref.
# 
# \[1\] [Kgg_competitions](https://www.kaggle.com/docs/competitions)
# 
# \[2\] [Kgg_definition](https://en.wikipedia.org/wiki/Kaggle)
# 
# \[3\] [kaggle-survey-2021](https://www.kaggle.com/c/kaggle-survey-2021)

# # 1.2 Contents
# ---
# 
# >     Introduction
# >     Contents
# >     Summary
# >     Data Import and Preprocessing
# >     Plots and Description
# 
# >     Kaggle's transformation. (World/East_Asia)
# >         1 user transformation
# >         2 Gender transformation
# >         3 Job transformation
# >         4 Age  transformation
# >         5 Degree transformation
# >         6 Experience transformation
# >         7 Salary transformation
# >         8 Language transformation
# 
# >     Position of Data Scientist in East Asia
# >         1 Salary
# >         2 Salary-Experience
# >         3 Degree
# >         4 Salary-Degree
# >         5 Language
# 
# >     Discussion
# >     Close

# # 1.3 Summary
# ---
# 

# <h3> used data </h3>
# 
# We used all the data for five years. (2017~2021)
# 
# <h3>  used Language and Library </h3>
# 
#    + Numpy
#    + Metplotlib
#    + seaborn
#    + Plotly
#        - plotly.express : An interface where you can draw a graph easily and quickly.
#        - plotly.graph_objects : You can customize it in the way you want because you can do more detailed work than express.
#        - plotly.figure_factory : Used before express existed and remains in the module for compatibility with previous versions
#        - plotly.subplots : A module that displays multiple graphs in one figure.
#        - plotly.offline : Save locally and create HTML that opens in a web browser and make it standalone
# 
# <h3> Grouping data sections </h3>
# 
#    - East Asia and World
#         - East Asia : ['China','Taiwan', 'South Korea', 'Japan']
#         - World : all data
#    - Gender
#         - [Male, Female, Others]
#    - Job   <br>
#     **Data_Analyst** =['Data Analyst','Data Miner,Information technology','Data Miner', 'Predictive Modeler','Information technology, networking, or system administration', 
#                         'A business discipline (accounting, economics, finance, etc.)', 'Business Analyst', Humanities', 'Statistician', 'Mathematics or statistics', 
#                         'Medical or life sciences (biology, chemistry, medicine, etc.)', Physics or astronomy', 'Social sciences (anthropology, psychology, sociology, etc.)', 
#                         'Environmental science or geology', 'Humanities (history, literature, philosophy, etc.)'] <br>
#     **Data_Scientist** =['Data Scientist',  'Research Scientist', 'Researcher','Machine Learning Engineer', 'Scientist/Researcher'] <br>
#     **Developer**=['Developer Relations/Advocacy','Data Engineer','Engineer','Engineering (non-computer focused)',
#                          'Programmer','Software Engineer', 'Computer Scientist','Computer science (software engineering, etc.)',  'Fine arts or performing arts','Product Manager', 'Software Developer/Software Engineer', 
#                          'Product/Project Manager','Program/Project Manager','DBA/Database Engineer']
#     **Not_Employed** = ['Currently not employed', 'Not employed', 'Student'] <br>
#     **Others** = ['I never declared a major', 'Other'] <br> <br>
#    - Age 
#    [18-21, 20s, 30s, 40s, 50s, 60s<] <br>
#    - Degree  
#    ['college', 'Bachelor’s degree','Master’s degree', 'Doctoral degree~', 'etc'] <br>
#    - Experience 
#    [<1, 1-3, 3-5, 5-10, 10+] <br>
#    - Salary 
#    [<999, 1,000-20,000, 20,000-59,999, 60,000-99,999, 100,000-199,999, 200,000~] <br>
#     

# ![image.png](attachment:85ab1662-6b79-4573-9fc6-fd4a06dac5d4.png)
# 
# ![image.png](attachment:a3267e76-64a2-45fa-9c8f-8136fc761106.png)
# 
# 

# # 2. data Import and pre-treatments
# ---

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
pio.templates.default = "none"

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df17= pd.read_csv("/kaggle/input/kaggle-survey-2017/multipleChoiceResponses.csv", encoding="ISO-8859-1")
df18= pd.read_csv("/kaggle/input/kaggle-survey-2018/multipleChoiceResponses.csv", )
df19= pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv", )
df20= pd.read_csv("/kaggle/input/kaggle-survey-2020/kaggle_survey_2020_responses.csv", )
df21= pd.read_csv("/kaggle/input/kaggle-survey-2021/kaggle_survey_2021_responses.csv", )


# # 3. plots and description
# ---

# In[ ]:


#질문 제거하기, replace
df17= df17.iloc[1:, :].replace("People 's Republic of China",'China')
df18= df18.iloc[1:, :].replace('Republic of Korea','South Korea')
df19= df19.iloc[1:, :].replace('Republic of Korea','South Korea')
df20= df20.iloc[1:, :].replace('Republic of Korea','South Korea')
df21= df21.iloc[1:, :]

## East Asia에는 대한민국, 일본, 중국, 타이완, 몽골, 북조선 총 6개의 국가가 속해 있다. 
## 이유는 알 수 없지만, 18년도엔 타이완이 없다. 
EastAsia17 = ['China',"People 's Republic of China", 'Taiwan', 'South Korea', 'Japan']
EastAsia18= ['China', 'South Korea', 'Japan', 'Republic of Korea'] 
EastAsia19 = ['China','Taiwan', 'South Korea', 'Japan', 'Republic of Korea']
EastAsia20 = ['China','Taiwan', 'South Korea','Republic of Korea', 'Japan']
EastAsia21 = ['China','Taiwan', 'South Korea', 'Japan']
EastAsia = ['Republic of Korea','China','Taiwan', 'South Korea', 'Japan', "People 's Republic of China" ]

df21_Ea = df21[df21['Q3'].isin(EastAsia)]
df21_Wo = df21[~df21['Q3'].isin(EastAsia)]
df21['region']=["EastAsia" if x in EastAsia else "World" for x in df21['Q3']]

df20_Ea = df20[df20['Q3'].isin(EastAsia)]
df20_Wo = df20[~df20['Q3'].isin(EastAsia)]
df20['region']=["EastAsia" if x in EastAsia else "World" for x in df20['Q3']]

df19_Ea = df19[df19['Q3'].isin(EastAsia)]
df19_Wo = df19[~df19['Q3'].isin(EastAsia)]
df19['region']=["EastAsia" if x in EastAsia else "World" for x in df19['Q3']]

df18_Ea = df18[df18['Q3'].isin(EastAsia)]
df18_Wo = df18[~df18['Q3'].isin(EastAsia)]
df18['region']=["EastAsia" if x in EastAsia else "World" for x in df18['Q3']]

df17_Ea = df17[df17['Country'].isin(EastAsia)]
df17_Wo = df17[~df17['Country'].isin(EastAsia)]
df17['region']=["EastAsia" if x in EastAsia else "World" for x in df17['Country']]

df21['year'] = '2021'
df20['year'] = '2020'
df19['year'] = '2019'
df18['year'] = '2018'
df17['year'] = '2017'

years = ['2017', '2018', '2019', '2020', '2021']

df21_Ea = df21[df21['Q3'].isin(EastAsia21)]
Ea21= (
    df21_Ea['Q3'].value_counts().to_frame()
    .reset_index().rename(columns={'index':'Country', 'Q3':'21'}))

df20_Ea=df20[df20['Q3'].isin(EastAsia)]
Ea20= (
    df20_Ea['Q3'].replace('Republic of Korea','South Korea')
    .value_counts().to_frame().reset_index()
    .rename(columns={'index':'Country', 'Q3':'20'}))

df19_Ea=df19[df19['Q3'].isin(EastAsia)]
Ea19= (df19_Ea['Q3'].replace('Republic of Korea','South Korea')
       .value_counts().to_frame().reset_index()
       .rename(columns={'index':'Country', 'Q3':'19'}))

df18_Ea=df18[df18['Q3'].isin(EastAsia)]
Ea18= (df18_Ea['Q3'].replace('Republic of Korea','South Korea')
       .value_counts().to_frame().reset_index()
       .rename(columns={'index':'Country', 'Q3':'18'}))
Ea18.value_counts()
#df18 열에 taiwan = 0을 추가 해야 합니다. 

df17_Ea = df17[df17['Country'].isin(EastAsia)]
Ea17= (df17_Ea['Country'].replace("People 's Republic of China",'China')
       .value_counts().to_frame().reset_index()
       .rename(columns={'index':'Country', 'Country':'17'}))

#data를 합쳐서 하나의 dataframe으로 만들어 줌.
df5years = pd.merge(Ea17, Ea18, on='Country', how='outer')
df5year =pd.merge(Ea19,Ea20, on='Country', how='outer')
df5year=pd.merge(df5year, Ea21, on='Country', how='outer')

df5years = pd.merge(df5years, df5year, on='Country', how='outer')

Ea21 = len(df21_Ea)
Wo21 = len(df21) - len(df21_Ea)

Ea20 = len(df20_Ea)
Wo20 = len(df20) - len(df20_Ea)

Ea19 = len(df19_Ea)
Wo19 = len(df19) - len(df19_Ea)

Ea18 = len(df18_Ea)
Wo18 = len(df18) -  len(df18_Ea)

Ea17 = len(df17_Ea)
Wo17 = len(df17) - len(df17_Ea)

years = ['2017','2018','2019','2020', '2021']

def percent (a, b):
    result =a/(a+b)*100
    result = np.round(result, 2)
    return result

def percentR (b, a):
    result =a/(a+b)*100
    result = np.round(result, 2)
    return result

percent = [percent(Ea17, Wo17), percent(Ea18, Wo18), percent(Ea19, Wo19), 
                                                 percent(Ea20, Wo20), percent(Ea21, Wo21)]


# # 3.1 Kaggle's transformation (World/East Asia)
# ---
# # 3.1.1 user transformation
# 

# <h2> Number of respondents </h2>
#     
# (bar, scatter plot : number of respondents to World and East Asia,
# Map plot : number of respondents to East Asia)
#     
# **World and East Asia: The same trend.**
#     
# East Asia: 15% of the total continent and 20.3% of the population (16/78.7: Ea/Wo)
#     
# 2018 Issue: Significant increase in respondents->Hypothesis: Due to the rapid increase in China.
#     
# 2018 Outliers Considering: 2022 Kaggle survey Respondents: Increased in both World and East Asia
#     
# I wish our team the honor of becoming a respondent to the Kaggle survey in 2022....

# In[ ]:


fig = go.Figure()
y=[len(df17_Ea),len(df18_Ea), len(df19_Ea),len(df20_Ea),len(df21_Ea)]

fig.add_trace(go.Bar(x=years, y=y,
                base=0,
                marker_color='#F2D64B',
                yaxis = "y1",
                name='East Asia',
                text= percent,
                texttemplate='%{text}  %', 
                textposition='outside',
                hovertemplate='<b>KaggleUser</b>: %{x}<br>'+ '<b>Count</b>: %{y}'))

fig.add_trace(go.Scatter(name = "World",
           x=years, 
           y=[len(df17), len(df18), len(df19), len(df20), len(df21)],
           marker_color='#979DA6',
           mode = 'lines+markers', # please check option here
           yaxis = "y2"))

fig.update_traces(hovertemplate='<b>Count</b>: %{y}<br><extra></extra>'+
                                '<b>Year</b>: %{x}<br>')

fig.update_layout(yaxis  = dict(title = "Kaggle User in East Asia",showgrid = False, range=[0, len(df21_Ea)*1.2]),
                  yaxis2 = dict(title = "Kaggle User in World", overlaying = "y1", side = "right", 
                  showgrid = False, 
                  zeroline = False, range=[0, len(df21)*1.2]))

fig.update_layout(title='<b>Kaggle Users</b>',title_font_size=20,
                  margin = dict(t=200, l=100, r=50, b=200),
                  height=700, width=700)

fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.1,
    xanchor="right",
    x=1))
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.9,
                                    y=-0.25,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


def world_map(locations,counts,title):
    data = [ dict(
            type = 'choropleth',
            locations = locations,
            z = counts,
            colorscale = 'Reds',
            locationmode = 'country names',
            autocolorscale = False,
            reversescale = False,
            marker = dict(
                line = dict(color = '#F7F7F7', width = 1.5)),
                colorbar = dict(autotick = True, legth = 3, len=0.75, title = 'respodents',
                               max = 1000, min = 0))]
    layout = dict(
        title=title,
        titlefont={'size': 28},
        width=700, 
        height=600,
        paper_bgcolor='#FFFFFF', 
        margin=dict(l=50, r=50, t=100, b=100),
        geo = dict(
            showframe = True,
            showcoastlines = True,
            fitbounds="locations"))
   
    fig = dict(data=data, layout=layout)
    iplot(fig, validate=False, filename='world-map')

z = df21_Ea['Q3'].value_counts()
 
world_map(locations=z.index, counts=z.values, title= '<b>EastAsia Countries<b>')


# <h2>18’ : </h2>
# 
# User change between United States and India.
# 
# China's markedly increase in 2018
# 
# + There is no Taiwan, but only China has increased. : East Asian political situation Issue can be suspected.

# In[ ]:


A18 = (
    df18['Q3']
    .replace({'Republic of Korea':'South Korea',
             'I do not wish to disclose my location' : 'Other'})
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={'index':'type', 'Q3':'2018'})
    .groupby('type')
    .sum()
    .reset_index()
)

A19 = (
    df19['Q3']
    .replace('Republic of Korea','South Korea')
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={'index':'type', 'Q3':'2019'})
    .groupby('type')
    .sum()
    .reset_index()
)

A17 = (
    df17['Country']
    .replace({'United States': 'United States of America',
              'Hong Kong': 'Hong Kong (S.A.R.)', 
              'United Kingdom':'United Kingdom of Great Britain and Northern Ireland',
             })
    .replace("People 's Republic of China",'China')
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={'index':'type', 'Country':'2017'})
    .groupby('type')
    .sum()
    .reset_index()
)

A18A19=pd.merge(A18,A19, how='outer',on='type').fillna(0)
A18A17=pd.merge(A18,A17, how='outer',on='type').fillna(0)
A18A19['minus']= A18A19['2018']-A18A19['2019']
A18A17['minus']= A18A17['2018']-A18A17['2017']

A18A17=A18A17.sort_values(by="minus", ascending=False)
A18A19=A18A19.sort_values(by="minus", ascending=False)


fig = go.Figure(data=[  
        go.Bar(x =A18A19['type'],
        y = A18A19['minus'],
        marker_color='#979DA6',
        name = '2018-2019', base=0),
        go.Bar(x =A18A17['type'],
        y = A18A17['minus'],
               marker_color='#F2D64B',
        name = '2018-2017', base=0)
        ])

fig.update_layout(title='<b>        Predicting outliers (2018)</b>',title_font_size=20,
                  margin = dict(t=200, l=100, r=10, b=200),
                  height=700, width=700,
                  xaxis_title=None,
                  yaxis_title=None)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.1,
    xanchor="right",
    x=1))
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.5,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# <h2>Total population: </h2>
# 
# 1.4 billion (85%) in China, 130 million in Japan, 0.5 billion in Korea, and 0.2 billion in Taiwan.
# 
# + China: The number of respondents is smaller than the population.
# + Japan: Starting in 2019, overtaking China
# + Taiwan : 2018 data 0 =? Diplomatic issues? The growth trend is weak.
# + Korea : Respondents at a similar level to Japan's population.
# + East Asia: The number of respondents will increase further.

# In[ ]:


#data preprocessing
total17 = ( 
    df17['region']
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={'index':'type', 'region':'respodents'})
    .groupby('type')
    .sum()
    .reset_index()
)
total18 = (
    df18['region']
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={'index':'type', 'region':'respodents'})
    .groupby('type')
    .sum()
    .reset_index()
)
total19 = (
    df19['region']
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={'index':'type', 'region':'respodents'})
    .groupby('type')
    .sum()
    .reset_index()
)
total20 = (
    df20['region']
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={'index':'type', 'region':'respodents'})
    .groupby('type')
    .sum()
    .reset_index()
)
total21 = (
    df21['region']
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={'index':'type', 'region':'respodents'})
    .groupby('type')
    .sum()
    .reset_index()
)

#graph
colors = ['#F2D64B','#979DA6']

fig = make_subplots(rows=1, cols=5, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}, {'type':'domain'}, {'type':'domain'}]],
                   subplot_titles=("2017", "2018", "2019", "2020", "2021"))
fig.add_trace(go.Pie(marker=dict(colors=colors),labels=total21['type'], values=total21['respodents'], name="2021", scalegroup='one'), 1, 5)
fig.add_trace(go.Pie(marker=dict(colors=colors),labels=total20['type'], values=total20['respodents'], name="2020", scalegroup='one'), 1, 4)
fig.add_trace(go.Pie(marker=dict(colors=colors),labels=total19['type'], values=total19['respodents'], name="2019", scalegroup='one'), 1, 3)
fig.add_trace(go.Pie(marker=dict(colors=colors),labels=total18['type'], values=total18['respodents'], name="2018", scalegroup='one'), 1, 2)
fig.add_trace(go.Pie(marker=dict(colors=colors),labels=total17['type'], values=total17['respodents'], name="2017", scalegroup='one'), 1, 1)

fig.update_traces(hole=.0, hoverinfo="label+percent+name", textposition='inside', textinfo='percent+label',
                  textfont_size=12)

fig.update_layout(title='<b>World vs EastAsia</b>',title_font_size=23,
                  margin = dict(t=300, l=0, r=0, b=200),
                  height=700, width=700)

fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.3,
    xanchor="right",
    x=1))
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.25,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='2017', x=df5years['Country'], y=df5years['17'], marker_color='#F2798F',text=df5years['17'].tolist(), textposition='outside'),
    go.Bar(name='2018', x=df5years['Country'], y=df5years['18'], marker_color='#88BFBA',text=df5years['18'].fillna(0).astype(int).tolist(), textposition='outside',),
    go.Bar(name='2019', x=df5years['Country'], y=df5years['19'], marker_color='#CDD9A3',text=df5years['19'].tolist(), textposition='outside'),
    go.Bar(name='2020', x=df5years['Country'], y=df5years['20'], marker_color='#F28705',text=df5years['20'].tolist(), textposition='outside',),
    go.Bar(name='2021', x=df5years['Country'], y=df5years['21'], marker_color='#D9946C',text=df5years['21'].tolist(), textposition='outside')])

fig.update_layout(barmode='group')

fig.update_layout(title='<b>Kaggle User in East Asia</b>',title_font_size=23,
                  margin = dict(t=200, l=100, r=10, b=200),
                  height=600, width=700)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_traces(hovertemplate='<b>Count</b>: %{y}')
fig.update_layout(legend=dict(
    orientation="v",
    yanchor="bottom",
    y=1.15,
    xanchor="right",
    x=1))
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.5,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# # 3.1.2 Gender transformation
# ---

# <h2>World: The proportion of female respondents increases (still below 20%) </h2>
# 
# The number of respondents is increasing in all genders.
# 
# Our team is also a team with high female members and wants to contribute as a respondent in 2022.

# In[ ]:


#data preprocessing
Gender_17 = (
    df17['GenderSelect']
    .replace(['A different identity', 'Prefer to self-describe', 'Non-binary, genderqueer, or gender non-conforming'], 'Others')
    .fillna('Others')
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={'index':'type', 'GenderSelect':'Gender'})
    .groupby('type')
    .sum()
    .reset_index())
Gender_18 = (
    df18['Q1']
    .replace(['Prefer not to say', 'Prefer to self-describe'], 'Others')
    .fillna('Others')
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={'index':'type', 'Q1':'Gender'})
    .groupby('type')
    .sum()
    .reset_index())
Gender_19 = (
    df19['Q2']
    .replace(['Prefer not to say','Prefer to self-describe'],'Others')
    .fillna('Others')
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={'index':'type', 'Q2':'Gender'})
    .groupby('type')
    .sum()
    .reset_index())
Gender_20 = (
    df20['Q2']
    .replace(['Prefer not to say', 'Prefer to self-describe', 'Nonbinary'], 'Others')
    .replace(['Man', 'Woman'], ['Male', 'Female'])
    .fillna('Others')
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={'index':'type', 'Q2':'Gender'})
    .groupby('type')
    .sum()
    .reset_index())
Gender_21 = (
    df21['Q2']
    .replace(['Prefer not to say', 'Prefer to self-describe', 'Nonbinary'], 'Others')
    .replace(['Man', 'Woman'], ['Male', 'Female'])
    .fillna('Others')
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={'index':'type', 'Q2':'Gender'})
    .groupby('type')
    .sum()
    .reset_index())

colors = ['#D9946C','#88BFBA', '#CDD9A3']

fig = make_subplots(rows=1, cols=5, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}, {'type':'domain'}, {'type':'domain'}]],)
fig.add_trace(go.Pie(marker=dict(colors=colors), labels=Gender_21['type'], values=Gender_21['Gender'], name="2021", scalegroup='one', text=np.array(Gender_21['Gender'].sum()), title="2021", titleposition='bottom center'),
              1, 5)
fig.add_trace(go.Pie(marker=dict(colors=colors), labels=Gender_20['type'], values=Gender_20['Gender'], name="2020", scalegroup='one', text=np.array(Gender_20['Gender'].sum()), title="2020", titleposition='bottom center'),
              1, 4)
fig.add_trace(go.Pie(marker=dict(colors=colors), labels=Gender_19['type'], values=Gender_19['Gender'], name="2019", scalegroup='one', text=np.array(Gender_19['Gender'].sum()), title="2019", titleposition='bottom center'),
              1, 3)
fig.add_trace(go.Pie(marker=dict(colors=colors), labels=Gender_18['type'], values=Gender_18['Gender'], name="2018", scalegroup='one', text=np.array(Gender_18['Gender'].sum()), title="2018", titleposition='bottom center'),
              1, 2)
fig.add_trace(go.Pie(marker=dict(colors=colors), labels=Gender_17['type'], values=Gender_17['Gender'], name="2017", scalegroup='one', text=np.array(Gender_17['Gender'].sum()), title="2017", titleposition='bottom center'),
              1, 1)

fig.update_traces(hole=.0, hoverinfo="label+percent+name", 
                  textinfo='label+percent+value')

fig.update_layout(title='<b>World Gender</b>',title_font_size=23,
                  margin = dict(t=300, l=100, r=0, b=200),
                  height=700, width=1000)

fig.update_layout(legend=dict(
    orientation="v",
    yanchor="bottom",
    y=1.3,
    xanchor="right",
    x=1))
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.85,
                                    y=-0.5,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# <h3>
#     
#     - Male (1004->2037 : 2017->2021) double increase
#     
#     - Female 183->327 : 2017->2021 increased 1.8 times
#     
#     - Others (8->64 : 2017->2021) 8x increase
#     
# </h3>
# [Compare the high and low points]
# 
# + It can be seen that the number of female respondents and the ratio of male respondents hardly change, which is a difference compared to World data.
# 
# + It can be seen that the degree of gender freedom in East Asia has increased relatively.
# + Compared to World data, it can be seen that in 2021 (1.87: 2.6= Wo: Ea), compared to 2017 (1.96: 0.7 = Ea), which was relatively conservative.

# In[ ]:


#data preprocessing
gender21= df21_Ea.loc[:, ['Q3', 'Q2', 'year']].rename(columns={'Q3':'Country', 'Q2':'Gender'})
gender20= df20_Ea.loc[:, ['Q3', 'Q2', 'year']].rename(columns={'Q3':'Country', 'Q2':'Gender'})
gender19= df19_Ea.loc[:, ['Q3', 'Q2', 'year']].rename(columns={'Q3':'Country', 'Q2':'Gender'})
gender18= df18_Ea.loc[:, ['Q3', 'Q1', 'year']].rename(columns={'Q3':'Country', 'Q1':'Gender'})
gender17= df17_Ea.loc[:, ['Country', 'GenderSelect', 'year']].rename(columns={'index':'type', 'GenderSelect':'Gender'})

Gender5y= pd.concat([gender17, gender18, gender19, gender20, gender21])
Gender5y= (Gender5y.replace(['Prefer not to say', 'Prefer to self-describe', 'Nonbinary', 'A different identity'], 'Others')
           .replace(['Man', 'Woman'], ['Male', 'Female'])
           .groupby(['year', 'Gender'])
           .size()
           .reset_index()
           .rename(columns = {0:"Count"}))

gen17_5y = Gender5y[Gender5y['year'] == "2017"].reset_index(drop = True)
gen18_5y = Gender5y[Gender5y['year'] == "2018"].reset_index(drop = True)
gen19_5y = Gender5y[Gender5y['year'] == "2019"].reset_index(drop = True)
gen20_5y = Gender5y[Gender5y['year'] == "2020"].reset_index(drop = True)
gen21_5y = Gender5y[Gender5y['year'] == "2021"].reset_index(drop = True)

Gen5y_ = pd.concat([gen17_5y, gen18_5y, gen19_5y, gen20_5y, gen21_5y], ignore_index = True)
Gen5y_= pd.pivot(Gen5y_, index = "year", columns = "Gender", values = "Count").reset_index()
Gen5y_

Gen5y_['year'].unique()

#graph
fig = go.Figure()

fig.add_trace(go.Bar(
    x = Gen5y_['year'],
    y = Gen5y_['Male'].tolist(),
    name = 'Male',
marker_color='#88BFBA', text=Gen5y_['Male'].tolist(), textposition='outside'))

fig.add_trace(go.Bar(
    x = Gen5y_['year'],
    y = Gen5y_['Female'].tolist(),
    name = 'Female',
marker_color='#D9946C', text=Gen5y_['Female'].tolist(), textposition='outside'))

fig.add_trace(go.Bar(
    x = Gen5y_['year'],
    y = Gen5y_['Others'].tolist(),
    name = 'Others',
marker_color='#CDD9A3', text=Gen5y_['Others'].tolist(), textposition='outside'))

fig.update_layout(barmode="group") 

fig.update_layout(title='<b>Gender by year</b>',title_font_size=22,
                  margin = dict(t=200, l=100, r=10, b=200),
                  height=700, width=700,
                  xaxis_title=None,
                  yaxis_title=None)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.5,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# # 3.1.3 Job transformation
# ---

# <h2> 21' World Vs East Asia Age Ratio: Bar plot </h2>
# 
# - Not Employed : More than 30% in both East Asia and the world, the highest.
#     
#     *Because "Students" is included.*
# - Data Scientist : High percentage in the world and East Asia.
# 
# - Relatively low proportion in East Asia.
#     <strong>**= Absolute lack of numbers**</strong>
#  
#  
# <p style="color:#FF0000";>
# We would like to move forward by selecting a **data scientist** with insufficient numbers in East Asia.
# </p>

# In[ ]:


#data preprocessing
Data_Analyst =['Data Analyst','Data Miner,Information technology','Data Miner',
                'Predictive Modeler','Information technology, networking, or system administration', 
                'A business discipline (accounting, economics, finance, etc.)', 'Business Analyst', 'Humanities',
                'Statistician', 'Mathematics or statistics', 'Medical or life sciences (biology, chemistry, medicine, etc.)', 
                'Physics or astronomy',  'Social sciences (anthropology, psychology, sociology, etc.)', 'Environmental science or geology',
                'Humanities (history, literature, philosophy, etc.)']
Data_Scientist =['Data Scientist',  'Research Scientist', 'Researcher',
                'Machine Learning Engineer', 'Scientist/Researcher']
Developer=['Developer Relations/Advocacy','Data Engineer','Engineer','Engineering (non-computer focused)',
           'Programmer','Software Engineer', 'Computer Scientist','Computer science (software engineering, etc.)', 
           'Fine arts or performing arts','Product Manager', 'Software Developer/Software Engineer',
           'Product/Project Manager','Program/Project Manager','DBA/Database Engineer']
Not_Employed =['Currently not employed', 'Not employed', 'Student']
Others = ['I never declared a major', 'Other']


df21job_Ea = df21_Ea.loc[:,['Q3','Q5']].rename(columns={'Q5':'2021'}).fillna('Other')
df20job_Ea = df20_Ea.loc[:,['Q3','Q5']].rename(columns={'Q5':'2020'}).fillna('Other')
df19job_Ea = df19_Ea.loc[:,['Q3','Q5']].rename(columns={'Q5':'2019'}).fillna('Other')
df18job_Ea = df18_Ea.loc[:,['Q3','Q5']].rename(columns={ 'Q5':'2018'}).fillna('Other')
df17job_Ea = df17_Ea.loc[:,['Country','CurrentJobTitleSelect']].rename(columns={'CurrentJobTitleSelect':'2017'}).fillna('Other')

df21job_Ea.value_counts('2021')
df21job_Ea['JOB']=["Data Analyst" if x in Data_Analyst
                   else "Data Scientist" if x in Data_Scientist # Data Scientist
                   else "Developer" if x in Developer
                    else "NotEmployed" if x in Not_Employed
                   else "Others" 
                   for x in df21job_Ea['2021']]
df21job_Ea.value_counts('JOB')

df20job_Ea.value_counts('2020')
df20job_Ea['JOB']=["Data Analyst" if x in Data_Analyst
                   else "Data Scientist" if x in Data_Scientist 
                   else "Developer" if x in Developer
                    else "NotEmployed" if x in Not_Employed
                   else "Other"
                   for x in df20job_Ea['2020']]
df20job_Ea[['2020','JOB']]

df19job_Ea.value_counts('2019')
df19job_Ea['JOB']=["Data Analyst" if x in Data_Analyst
                   else "Data Scientist" if x in Data_Scientist 
                   else "Developer" if x in Developer
                    else "NotEmployed" if x in Not_Employed
                    else "Other"
                   for x in df19job_Ea['2019']]

df19jobTest = df19job_Ea.loc[df19job_Ea.JOB == 'Other']
df19jobTest['2019'].value_counts()


df18job_Ea.value_counts('2018')
df18job_Ea['JOB']=["Data Analyst" if x in Data_Analyst
                   else "Data Scientist" if x in Data_Scientist 
                   else "Developer" if x in Developer
                    else "NotEmployed" if x in Not_Employed
                    else "Other"
                   for x in df18job_Ea['2018']]

df18jobTest = df18job_Ea.loc[df18job_Ea.JOB == 'Other']
df18jobTest['2018'].value_counts()


df17job_Ea.value_counts('2017')
df17job_Ea['JOB']=["Data Analyst" if x in Data_Analyst
                   else "Data Scientist" if x in Data_Scientist 
                   else "Developer" if x in Developer
                    else "NotEmployed" if x in Not_Employed
                    else "Other"
                   for x in df17job_Ea['2017']]

df17jobTest = df17job_Ea.loc[df17job_Ea.JOB == 'Other']
df17jobTest['2017'].value_counts()


df21jobTest = df21job_Ea.loc[df21job_Ea.JOB == 'Other']
df21jobTest['2021'].head()
df21job_Ea.value_counts('JOB')

dfjob21 =df21job_Ea.groupby(['Q3','JOB']).size().reset_index().rename(columns = {0:"Count"}).rename(columns={'Q3':'country'})
dfjob20 =df20job_Ea.groupby(['Q3','JOB']).size().reset_index().rename(columns = {0:"Count"}).rename(columns={'Q3':'country'})
dfjob19 =df19job_Ea.groupby(['Q3','JOB']).size().reset_index().rename(columns = {0:"Count"}).rename(columns={'Q3':'country'})
dfjob18 =df18job_Ea.groupby(['Q3','JOB']).size().reset_index().rename(columns = {0:"Count"}).rename(columns={'Q3':'country'})
dfjob17 =df17job_Ea.groupby(['Country','JOB']).size().reset_index().rename(columns = {0:"Count"}).rename(columns={'Country':'country'})

df21_Ea_job =df21job_Ea.groupby(['JOB']).size().reset_index().rename(columns = {0:"Count"})
df20_Ea_job =df20job_Ea.groupby(['JOB']).size().reset_index().rename(columns = {0:"Count"})
df19_Ea_job =df19job_Ea.groupby(['JOB']).size().reset_index().rename(columns = {0:"Count"})
df18_Ea_job =df18job_Ea.groupby(['JOB']).size().reset_index().rename(columns = {0:"Count"})
df17_Ea_job =df17job_Ea.groupby(['JOB']).size().reset_index().rename(columns = {0:"Count"})

df21_DA=df21[df21['Q5'].isin(Data_Analyst)]
df21_DS=df21[df21['Q5'].isin(Data_Scientist)]
df21_D=df21[df21['Q5'].isin(Developer)]
df21_N=df21[df21['Q5'].isin(Not_Employed)]
df21_O=df21[df21['Q5'].isin(Others)]

World_ = np.array([df21_DA['Q5'].count(), df21_DS['Q5'].count(), df21_D['Q5'].count(), df21_N['Q5'].count(), df21_O['Q5'].count()]) 
East_Asia_ = df21_Ea_job['Count'].to_numpy()
World =((World_/World_.sum())*100).round(1)
East_Asia =((East_Asia_/East_Asia_.sum())*100).round(1)
y = df21_Ea_job.JOB.to_numpy()

fig = go.Figure(data=[
    go.Bar(y=y, x=World, orientation='h', name="World", base=0, hovertemplate='<b>World</b>: %{x}%<br>', marker_color='#979DA6', text=World, textposition='outside'),
    go.Bar(y=y, x=-East_Asia, orientation='h', name="East Asia", base=0, hovertemplate='<b>East Asia</b>: %{x}%<br>', marker_color='#F2D64B', text=East_Asia, textposition='outside')])

fig.update_layout(barmode='stack')
fig.update_layout(title='<b>World vs EastAsia</b>',title_font_size=22,
                  margin = dict(t=200, l=100, r=50, b=200),
                  height=700, width=750,
                  xaxis_title=None,
                  yaxis_title=None)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.1,
    xanchor="right",
    x=1))
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.5,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# <h2> World Job Ratio: Heat Map </h2> 
# 
# The trend of increasing each job except Others. <br>
# Data Scientist has a high proportion, and the trend is to increase further in 2022. <br>
# 
# <h2> East Asia Job Ratio: Heat Map </h2> 
# East Asia : Increasing the ratio of data scientist. <br>
# 

# In[ ]:


#data preprocessing
df21job= df21.loc[:,['region','Q5']].rename(columns={'Q5':'2021'}).fillna('Others')
df20job= df20.loc[:,['region','Q5']].rename(columns={'Q5':'2020'}).fillna('Others')
df19job= df19.loc[:,['region','Q5']].rename(columns={'Q5':'2019'}).fillna('Others')
df18job= df18.loc[:,['region','Q6']].rename(columns={ 'Q6':'2018'}).fillna('Others')
df17job= df17.loc[:,['region','CurrentJobTitleSelect']].rename(columns={'CurrentJobTitleSelect':'2017'}).fillna('Others')

df21job['JOB']=["Data Analyst" if x in Data_Analyst
                   else "Data Scientist" if x in Data_Scientist # Data Scientist
                   else "Developer" if x in Developer
                    else "NotEmployed" if x in Not_Employed
                   else "Others" 
                   for x in df21job['2021']]


df20job['JOB']=["Data Analyst" if x in Data_Analyst
                   else "Data Scientist" if x in Data_Scientist 
                   else "Developer" if x in Developer
                    else "NotEmployed" if x in Not_Employed
                   else "Others"
                   for x in df20job['2020']]


df19job['JOB']=["Data Analyst" if x in Data_Analyst
                   else "Data Scientist" if x in Data_Scientist 
                   else "Developer" if x in Developer
                    else "NotEmployed" if x in Not_Employed
                    else "Others"
                   for x in df19job['2019']]


df18job['JOB']=["Data Analyst" if x in Data_Analyst
                   else "Data Scientist" if x in Data_Scientist 
                   else "Developer" if x in Developer
                    else "NotEmployed" if x in Not_Employed
                    else "Others"
                   for x in df18job['2018']]


df17job['JOB']=["Data Analyst" if x in Data_Analyst
                   else "Data Scientist" if x in Data_Scientist 
                   else "Developer" if x in Developer
                    else "NotEmployed" if x in Not_Employed
                    else "Others"
                   for x in df17job['2017']]

df21_job =df21job.groupby(['JOB']).size().reset_index().rename(columns = {0:"Count"})
df20_job =df20job.groupby(['JOB']).size().reset_index().rename(columns = {0:"Count"})
df19_job =df19job.groupby(['JOB']).size().reset_index().rename(columns = {0:"Count"})
df18_job =df18job.groupby(['JOB']).size().reset_index().rename(columns = {0:"Count"})
df17_job =df17job.groupby(['JOB']).size().reset_index().rename(columns = {0:"Count"})

merge11=pd.merge(df21_job,df20_job, how='outer',on='JOB')
merge21=pd.merge(df19_job,df18_job, how='outer',on='JOB')
merge31=pd.merge(merge11,merge21, how='outer',on='JOB')
merge_Wo=(pd.merge(merge31,df17_job, how='outer',on='JOB')
            .rename(columns = {'Count_x_x':'2021','Count_y_x':'2020','Count_x_y':'2019','Count_y_y':'2018','Count':'2017'}).fillna(0)
            .reindex(columns = ['JOB','2017','2018','2019','2020','2021' ]))

df21job_Ea = df21job[df21job['region'] == 'EastAsia'].loc[:,['region','JOB']].rename(columns={'region':'EastAsia'})
df20job_Ea = df20job[df20job['region'] == 'EastAsia'].loc[:,['region','JOB']].rename(columns={'region':'EastAsia'})
df19job_Ea = df19job[df19job['region'] == 'EastAsia'].loc[:,['region','JOB']].rename(columns={'region':'EastAsia'})
df18job_Ea = df18job[df18job['region'] == 'EastAsia'].loc[:,['region','JOB']].rename(columns={'region':'EastAsia'})
df17job_Ea = df17job[df17job['region'] == 'EastAsia'].loc[:,['region','JOB']].rename(columns={'region':'EastAsia'})

df21job_Ea =df21job_Ea.groupby(['JOB']).size().reset_index().rename(columns = {0:"Count"})
df20job_Ea =df20job_Ea.groupby(['JOB']).size().reset_index().rename(columns = {0:"Count"})
df19job_Ea =df19job_Ea.groupby(['JOB']).size().reset_index().rename(columns = {0:"Count"})
df18job_Ea =df18job_Ea.groupby(['JOB']).size().reset_index().rename(columns = {0:"Count"})
df17job_Ea =df17job_Ea.groupby(['JOB']).size().reset_index().rename(columns = {0:"Count"})

merge1=pd.merge(df21job_Ea,df20job_Ea, how='outer',on='JOB')
merge2=pd.merge(df19job_Ea,df18job_Ea, how='outer',on='JOB')
merge3=pd.merge(merge1,merge2, how='outer',on='JOB')
merge=(pd.merge(merge3,df17job_Ea, how='outer',on='JOB')
         .rename(columns = {'Count_x_x':'2021','Count_y_x':'2020','Count_x_y':'2019','Count_y_y':'2018','Count':'2017'}).fillna(0)
         .reindex(columns = ['JOB','2017','2018','2019','2020','2021' ]))

#graph
z1=((merge_Wo.iloc[:,[1,2,3,4,5]].to_numpy()/merge_Wo.iloc[:,[1,2,3,4,5]].to_numpy().sum())*100).round(1)
z2=((merge.iloc[:,[1,2,3,4,5]].to_numpy()/merge.iloc[:,[1,2,3,4,5]].to_numpy().sum())*100).round(1)

x=['2017-year','2018-year','2019-year','2020-year','2021-year']
y1=merge_Wo['JOB'].tolist()
y2=merge['JOB'].tolist()


fig1 = ff.create_annotated_heatmap(z1, x = x, y = y1, colorscale='sunset')
fig2 = ff.create_annotated_heatmap(z2, x = x, y = y2, colorscale='sunset')

for annot in fig2['layout']['annotations']:
    annot['xref'] = 'x2'
    
fig = make_subplots(rows=1, cols=2)
fig.add_trace(fig1.data[0], row=1, col=1)
fig.add_trace(fig2.data[0], row=1, col=2)
fig.update_layout(fig1.layout, title='<b>           World vs EastAsia</b>',title_font_size=22,
                  margin = dict(t=200, l=100, r=10, b=200),
                  height=700, width=1150, coloraxis=dict(showscale=True, colorscale='sunset'))
fig.update_traces(hovertemplate='<b>Job</b>: %{y}<br>'+
                                '<b>Year</b>: %{x}<br>'+
                                '<b>Percent</b>: %{z}%')
fig.layout.annotations += fig2.layout.annotations
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.9,
                                    y=-0.25,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))

fig.show()


# # 3.1.4 Age transformation
# ---

# <h2> > Age change in World and East Asia by year: Stacked scatter plot </h2>
# 
# 1. In the case of Age data, there is no 2017 data.
# 2. 70% of the World respondents said 20s to 30s.
# 3. 70% of East Asia respondents said 20s to 30s.
# 4. The number of respondents increases, but the ratio seems to have stabilized.

# In[ ]:


#data preprocessing
#World
Age21_W = df21.loc[:,['Q3','Q1', 'year']].reset_index().rename(columns={'Q3':'East_Asia', 'Q1':'age'}).fillna('etc')
Age20_W = df20.loc[:,['Q3','Q1','year']].reset_index().rename(columns={'Q3':'East_Asia', 'Q1':'age'}).fillna('etc')
Age19_W = df19.loc[:,['Q3','Q1','year']].reset_index().rename(columns={'Q3':'East_Asia', 'Q1':'age'}).fillna('etc')
Age18_W = df18.loc[:,['Q3','Q2','year']].reset_index().rename(columns={'Q3':'East_Asia', 'Q2':'age'}).fillna('etc')

Age5y_W= pd.concat([Age21_W, Age20_W, Age19_W, Age18_W])
Age5y_W= (Age5y_W.replace(['60-69', '70+', '70-79', '80+'], '60+')
           .replace(['22-24', '25-29'], '22-29')
           .replace(['30-34', '35-39'], '30-39')
            .replace(['40-44', '45-49'], '40-49')
        .replace(['50-54', '55-59'], '50-59')
           .groupby(['year', 'age'])
           .size()
           .reset_index()
           .rename(columns = {0:"Count"}))

Age21_percent_W = Age5y_W[Age5y_W['year'] == "2021"].reset_index(drop = True)
Age21_percent_W['percentage'] = Age21_percent_W["Count"] / Age21_percent_W["Count"].sum()
Age21_percent_W['%'] = np.round(Age21_percent_W['percentage'] * 100, 1)

Age20_percent_W = Age5y_W[Age5y_W['year'] == "2020"].reset_index(drop = True)
Age20_percent_W['percentage'] = Age20_percent_W["Count"] / Age20_percent_W["Count"].sum()
Age20_percent_W['%'] = np.round(Age20_percent_W['percentage'] * 100, 1)

Age19_percent_W = Age5y_W[Age5y_W['year'] == "2019"].reset_index(drop = True)
Age19_percent_W['percentage'] = Age19_percent_W["Count"] / Age19_percent_W["Count"].sum()
Age19_percent_W['%'] = np.round(Age19_percent_W['percentage'] * 100, 1)

Age18_percent_W = Age5y_W[Age5y_W['year'] == "2018"].reset_index(drop = True)
Age18_percent_W['percentage'] = Age18_percent_W["Count"] / Age18_percent_W["Count"].sum()
Age18_percent_W['%'] = np.round(Age18_percent_W['percentage'] * 100, 1)

Age5y_percent_W = pd.concat([Age18_percent_W, Age19_percent_W, Age20_percent_W, Age21_percent_W], ignore_index = True)
Age5y_percent_W= pd.pivot(Age5y_percent_W, index = "year", columns = 'age', values = "%").reset_index()
Age5y_percent_W

Age21 = df21_Ea.loc[:,['Q3','Q1', 'year']].reset_index().rename(columns={'Q3':'East_Asia', 'Q1':'age'}).fillna('etc')
Age20 = df20_Ea.loc[:,['Q3','Q1','year']].reset_index().rename(columns={'Q3':'East_Asia', 'Q1':'age'}).fillna('etc')
Age19 = df19_Ea.loc[:,['Q3','Q1','year']].reset_index().rename(columns={'Q3':'East_Asia', 'Q1':'age'}).fillna('etc')
Age18 = df18_Ea.loc[:,['Q3','Q2','year']].reset_index().rename(columns={'Q3':'East_Asia', 'Q2':'age'}).fillna('etc')

Age5y= pd.concat([Age21, Age20, Age19, Age18])
Age5y= (Age5y.replace(['60-69', '70+', '70-79', '80+'], '60+')
           .replace(['22-24', '25-29'], '22-29')
           .replace(['30-34', '35-39'], '30-39')
            .replace(['40-44', '45-49'], '40-49')
        .replace(['50-54', '55-59'], '50-59')
           .groupby(['year', 'age'])
           .size()
           .reset_index()
           .rename(columns = {0:"Count"}))

#EastAsia
Age21_percent = Age5y[Age5y['year'] == "2021"].reset_index(drop = True)
Age21_percent['percentage'] = Age21_percent["Count"] / Age21_percent["Count"].sum()
Age21_percent['%'] = np.round(Age21_percent['percentage'] * 100, 1)
Age21_percent

Age20_percent = Age5y[Age5y['year'] == "2020"].reset_index(drop = True)
Age20_percent['percentage'] = Age20_percent["Count"] / Age20_percent["Count"].sum()
Age20_percent['%'] = np.round(Age20_percent['percentage'] * 100, 1)
Age20_percent

Age19_percent = Age5y[Age5y['year'] == "2019"].reset_index(drop = True)
Age19_percent['percentage'] = Age19_percent["Count"] / Age19_percent["Count"].sum()
Age19_percent['%'] = np.round(Age19_percent['percentage'] * 100, 1)
Age19_percent

Age18_percent = Age5y[Age5y['year'] == "2018"].reset_index(drop = True)
Age18_percent['percentage'] = Age18_percent["Count"] / Age18_percent["Count"].sum()
Age18_percent['%'] = np.round(Age18_percent['percentage'] * 100, 1)
Age18_percent

Age5y_percent = pd.concat([Age18_percent, Age19_percent, Age20_percent, Age21_percent], ignore_index = True)
Age5y_percent= pd.pivot(Age5y_percent, index = "year", columns = 'age', values = "%").reset_index()
Age5y_percent

Age5y_percent_order = Age5y_percent_W['year'].tolist()
Age5y_order = Age5y_W['age'].unique().tolist()

#graph1
fig = go.Figure()
fig.add_trace(go.Scatter(
    x = Age5y_percent_order, 
    y = Age5y_percent_W['18-21'].tolist(), 
    mode = "lines", 
    name = '18-21',
    line = dict(width = 1),
    stackgroup = "one",
    marker_color='#F2798F'))

fig.add_trace(go.Scatter(
    x = Age5y_percent_order, 
    y = Age5y_percent_W['22-29'].tolist(), 
    mode = "lines", 
    name = "20s",
    line = dict(width = 1),
    stackgroup = "one",
    marker_color='#88BFBA'))
fig.add_trace(go.Scatter(
    x = Age5y_percent_order, 
    y = Age5y_percent_W['30-39'].tolist(), 
    mode = "lines", 
    name = "30s",
    line = dict(width = 1),
    stackgroup = "one",
    marker_color='#CDD9A3'))

fig.add_trace(go.Scatter(
    x = Age5y_percent_order, 
    y = Age5y_percent_W['40-49'].tolist(), 
    mode = "lines", 
    name = "40s",
    line = dict(width = 1),
    stackgroup = "one",
    marker_color='#F28705'))
fig.add_trace(go.Scatter(
    x = Age5y_percent_order, 
    y = Age5y_percent_W['50-59'].tolist(), 
    mode = "lines", 
    name = "50s",
    line = dict(width = 1),
    stackgroup = "one",
    marker_color='#D9946C'))
fig.add_trace(go.Scatter(
    x = Age5y_percent_order, 
    y = Age5y_percent_W['60+'].tolist(), 
    mode = "lines", 
    name = "60s<",
    line = dict(width = 1),
    stackgroup = "one",
    marker_color='#F2D64B'))

fig.update_traces(hovertemplate='<b>Percent</b>: %{y}%<br>'+
                                '<b>Year</b>: %{x}<br>')
fig.update_layout(yaxis_range = (0, 100), height=500, width=700,
                 title_text="<b>World</b>", title_font_size=20,
                 title_x=0.5)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.2,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()

#graph2
Age5y_percent_order = Age5y_percent['year'].tolist()
Age5y_order = Age5y['age'].unique().tolist()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x = Age5y_percent_order, 
    y = Age5y_percent['18-21'].tolist(), 
    mode = "lines", 
    name = '18-21',
    line = dict(width = 1),
    stackgroup = "one",
    marker_color='#F2798F'))

fig.add_trace(go.Scatter(
    x = Age5y_percent_order, 
    y = Age5y_percent['22-29'].tolist(), 
    mode = "lines", 
    name = "20s",
    line = dict(width = 1),
    stackgroup = "one",
    marker_color='#88BFBA'))
fig.add_trace(go.Scatter(
    x = Age5y_percent_order, 
    y = Age5y_percent['30-39'].tolist(), 
    mode = "lines", 
    name = "30s",
    line = dict(width = 1),
    stackgroup = "one",
    marker_color='#CDD9A3'))

fig.add_trace(go.Scatter(
    x = Age5y_percent_order, 
    y = Age5y_percent['40-49'].tolist(), 
    mode = "lines", 
    name = "40s",
    line = dict(width = 1),
    stackgroup = "one",
    marker_color='#F28705'))
fig.add_trace(go.Scatter(
    x = Age5y_percent_order, 
    y = Age5y_percent['50-59'].tolist(), 
    mode = "lines", 
    name = "50s",
    line = dict(width = 1),
    stackgroup = "one",
    marker_color='#D9946C'))
fig.add_trace(go.Scatter(
    x = Age5y_percent_order, 
    y = Age5y_percent['60+'].tolist(), 
    mode = "lines", 
    name = "60s<",
    line = dict(width = 1),
    stackgroup = "one",
    marker_color='#F2D64B'))
fig.update_traces(hovertemplate='<b>Percent</b>: %{y}%<br>'+
                                '<b>Year</b>: %{x}<br>')
fig.update_layout(yaxis_range = (0, 100), height=500, width=700,
                 title_text="<b>East Asia</b>", title_font_size=20,
                 title_x=0.5)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.2,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# <h2> 17'East Asia Age Ratio: Heat Map </h2>
# 
# + **East Asia** : 50% or more. Those in their 20s and 30s.
# + **Korea**: Those in their 20s are the highest.
#     The number of respondents in their 50s and older is also large.
# + **Taiwan** : The number of respondents in their 30s and older is relatively small.
# + **China**: 70% or more of respondents in their 30s or younger.
#     Related to life expectancy?
# + **Japan**: Like an aging country, all ages are evenly distributed.
#     Even if you're older, there are many respondents to Kaggle.

# In[ ]:


#data processing
df21Age_Ea = df21_Ea.loc[:,['Q3','Q1']].reset_index().rename(columns={'Q3':'East_Asia', 'Q1':'2021'}).fillna('etc')

df21Age_Ea=(df21Age_Ea.replace(['60-69', '70+', '70-79', '80+'], '60+')
           .replace(['22-24', '25-29'], '22-29')
           .replace(['30-34', '35-39'], '30-39')
            .replace(['40-44', '45-49'], '40-49')
        .replace(['50-54', '55-59'], '50-59'))

# 연령-지역 %
dfKo_Age21= df21Age_Ea[df21Age_Ea['East_Asia']=='South Korea']
dfKo_Age21_per=dfKo_Age21['2021'].value_counts().to_frame().reset_index()
dfKo_Age21_per['South Korea']=((dfKo_Age21_per['2021'] / len(dfKo_Age21))*100).round(2)

dfTw_Age21= df21Age_Ea[df21Age_Ea['East_Asia']=='Taiwan']
dfTw_Age21_per=dfTw_Age21['2021'].value_counts().to_frame().reset_index()
dfTw_Age21_per['Taiwan']=((dfTw_Age21_per['2021'] / len(dfTw_Age21))*100).round(2)
dfTw_Age21_per

dfCh_Age21= df21Age_Ea[df21Age_Ea['East_Asia']=='China']
dfCh_Age21_per=dfCh_Age21['2021'].value_counts().to_frame().reset_index()
dfCh_Age21_per['China']=((dfCh_Age21_per['2021'] / len(dfCh_Age21))*100).round(2)
dfCh_Age21_per

df21Age_Ea.head()
dfJp_Age21= df21Age_Ea[df21Age_Ea['East_Asia']=='Japan']
dfJp_Age21_per=dfJp_Age21['2021'].value_counts().to_frame().reset_index()
dfJp_Age21_per['Japan']=((dfJp_Age21_per['2021'] / len(dfJp_Age21))*100).round(2)
dfJp_Age21_per


merge1= pd.merge(dfKo_Age21_per,dfTw_Age21_per, on='index', how='outer')
merge2= pd.merge(dfCh_Age21_per,dfJp_Age21_per, on='index', how='outer')
merge= pd.merge(merge1,merge2, on='index', how='outer').fillna(0).sort_values(by=['index'],ascending=True)

#graph
x1=['South Korea','Taiwan','China','Japan']
y1=merge.sort_values(by=['index'], ascending=True)['index'].tolist()
z1=merge.iloc[:,[2,4,6,8]].to_numpy()

fig = go.Figure(data=go.Heatmap(
                   z=z1,
                   x=x1,
                   y=y1,
                   hoverongaps = True,
                   opacity=1.0, xgap=2.5, ygap=2.5))
fig = ff.create_annotated_heatmap(z1, x = x1, y = y1, colorscale='sunset')
fig.update_layout(height=500, width=600,
                 title_text="<b>East Asia Age (2021)</b>", title_font_size=20,
                 title_x=0.5)
fig.update_traces(hovertemplate='<b>Age</b>: %{y}<br>'+
                                '<b>Country</b>: %{x}<br>'+
                                '<b>Percent</b>: %{z}%')
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.2,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# > 17'East Asia's age ratio: Box plot
# 
# > 2017: Data is not a section but an individual number.  <br>
# > If you divide the interval, you can add it to the previous graph. <br>
# > It was data that I could draw a bar plot, so I drew it. <br>
# > You can see a 100-year-old in China, but they don't remove missing values on purpose.

# In[ ]:


# 연도별 나이 
df21Age_Ea = df21_Ea.loc[:,['Q3','Q1']].reset_index().rename(columns={'Q3':'East_Asia', 'Q1':'2021'}).fillna('etc')
df20Age_Ea = df20_Ea.loc[:,['Q3','Q1']].reset_index().rename(columns={'Q3':'East_Asia', 'Q1':'2020'}).fillna('etc')
df19Age_Ea = df19_Ea.loc[:,['Q3','Q1']].reset_index().rename(columns={'Q3':'East_Asia', 'Q1':'2019'}).fillna('etc')
df18Age_Ea = df18_Ea.loc[:,['Q3','Q2']].reset_index().rename(columns={'Q3':'East_Asia', 'Q2':'2018'}).fillna('etc')
df17Age_Ea = df17_Ea.loc[:,['Country','Age']].reset_index().rename(columns={'Country':'East_Asia', 'Age':'2017'}).fillna('etc')

#data frame 정리
dfAge21 =df21Age_Ea.groupby(['East_Asia','2021']).size().reset_index().rename(columns = {0:"Count"})
dfAge20 =df20Age_Ea.groupby(['East_Asia','2020']).size().reset_index().rename(columns = {0:"Count"})
dfAge19 =df19Age_Ea.groupby(['East_Asia','2019']).size().reset_index().rename(columns = {0:"Count"})
dfAge18 =df18Age_Ea.groupby(['East_Asia','2018']).size().reset_index().rename(columns = {0:"Count"})
dfAge17 =(df17Age_Ea.groupby(['East_Asia','2017'])
          .size().reset_index().rename(columns = {0:"Count"}))
#graph
fig = go.Figure()

x = ['China','Japan','South Korea','Taiwan']

fig.add_trace(go.Box( y=dfAge17['2017'][dfAge17['East_Asia']=="Japan"].to_numpy(),
    name='Japan',
    marker=dict(color='#CDD9A3')))
fig.add_trace(go.Box(y=dfAge17['2017'][dfAge17['East_Asia']=="China"].to_numpy(),
    name='China',
    marker=dict(color='#88BFBA')))
fig.add_trace(go.Box(y=dfAge17['2017'][dfAge17['East_Asia']=="South Korea"].to_numpy(),
    name='South Korea',
    marker=dict(color='#F2798F')))
fig.add_trace(go.Box(y=dfAge17['2017'][dfAge17['East_Asia']=="Taiwan"].to_numpy(),
    name='Taiwan',
    marker=dict(color='#F28705'
    ),))

fig.update_layout(yaxis = dict(range=[0, 120]))


fig.update_layout(yaxis_range = (0, 110), height=600, width=700,
                  title_text="<b>Age in East Asia (2017)</b>", title_font_size=20,
                  margin = dict(t=100, l=50, r=50, b=100),
                  title_x=0.5)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_layout(legend=dict(
    orientation="v",
    yanchor="bottom",
    y=0.8,
    xanchor="right",
    x=1))
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.2,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# # 3.1.5 Degree transformation
# ---

# <h2> World job ratio in each country: pie plot </h2>
# 
# - World: 90% or higher Bachelor's degree
# - East Asia: 85% bachelor's degree or higher

# In[ ]:


#data preprocessing
degree_wo = (df21['Q4']
             .replace(['No formal education past high school',
                       'Some college/university study without earning a bachelor’s degree'],'~college')
             .replace(['Doctoral degree',
                       'Professional doctorate'],'Doctoral degree~')
             .value_counts().to_frame())
degree_ea = (df21_Ea['Q4']
             .replace(['No formal education past high school',
                       'Some college/university study without earning a bachelor’s degree'],'~college')
             .replace(['Doctoral degree',
                       'Professional doctorate'],'Doctoral degree~')
             .value_counts().to_frame())

#graph
colors = ['#F2798F','#88BFBA', '#CDD9A3', '#F28705', '#D9946C']
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'pie'}]], subplot_titles=("World", "East Asia"))
fig.add_trace(go.Pie(marker=dict(colors=colors), labels=degree_wo.index, values=degree_wo['Q4'].to_numpy(), name="World"),
              1, 1)
fig.add_trace(go.Pie(marker=dict(colors=colors), labels=degree_ea.index, values=degree_ea['Q4'].to_numpy(), name="East Asia"),
              1, 2)

fig.update_traces(hole=.0, hoverinfo="label+percent+name")

fig.update_layout(title='<b>World vs East Asia</b>',title_font_size=22,
                  margin = dict(t=200, l=30, r=0, b=200),
                  height=700, width=700)
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.1,
    xanchor="right",
    x=1.0))
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.5,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# <h2> Percentage of East Asia degrees by year: sunburst plot </h2>
# 
# The highest percentage of respondents with **master's degrees** per year

# In[ ]:


#data preprocessing
df21_Ea_degree=(df21_Ea['Q4'].replace(['No formal education past high school', 'Some college/university study without earning a bachelor’s degree'],'~college')
                             .replace(['Doctoral degree','Professional doctorate'],'Doctoral degree~')
                             .value_counts().to_frame().rename(columns={'Q4':'2021'}))
df20_Ea_degree=(df20_Ea['Q4'].replace(['No formal education past high school', 'Some college/university study without earning a bachelor’s degree'],'~college')
                             .replace(['Doctoral degree', 'Professional degree'],'Doctoral degree~')
                             .value_counts().to_frame().rename(columns={'Q4':'2020'}))
df19_Ea_degree=(df19_Ea['Q4'].replace(['No formal education past high school','Some college/university study without earning a bachelor’s degree'],'~college')
                             .replace(['Doctoral degree', 'Professional degree'],'Doctoral degree~')
                             .value_counts().to_frame().rename(columns={'Q4':'2019'}))
df18_Ea_degree=(df18_Ea['Q4'].replace(['No formal education past high school', 'Some college/university study without earning a bachelor’s degree'],'~college')
                             .replace(['Doctoral degree', 'Professional degree'],'Doctoral degree~')
                             .value_counts().to_frame().rename(columns={'Q4':'2018'}))
df17_Ea_degree=(df17_Ea['FormalEducation']
                .replace(['No formal education past high school', 'Some college/university study without earning a bachelor’s degree'],'~college')
                .replace(['Doctoral degree', 'Professional degree'],'Doctoral degree~')
                .value_counts().to_frame()
                .rename(columns={'FormalEducation':'2017'} ,index = {'I did not complete any formal education past high school':'No formal education past high school','Master\'s degree':'Master’s degree','Bachelor\'s degree':'Bachelor’s degree','Some college/university study without earning a bachelor\'s degree':'Some college/university study without earning a bachelor’s degree'})  )
                
concat1 = pd.concat([df21_Ea_degree,df20_Ea_degree],axis=1, join='outer')  
concat2 = pd.concat([df19_Ea_degree,df18_Ea_degree],axis=1, join='outer')  
concat3 = pd.concat([concat1,concat2],axis=1, join='outer') 
df21_Ea_degree_yearly_=concat3.join(df17_Ea_degree).fillna(0).transpose() #.transpose() 행 열 바꾸기

df21_Ea_degree_yearly=df21_Ea_degree_yearly_.stack().to_frame().reset_index().rename(columns={'level_0':'year','level_1':'degree',0:'value'})
df21_Ea_degree_yearly

#graph
fig = px.sunburst(df21_Ea_degree_yearly, path=['year','degree'], values=df21_Ea_degree_yearly['value'].tolist())
fig.update_layout( margin = dict(t=10, l=10, r=10, b=10),colorway=("#F2798F","#88BFBA","#CDD9A3",'#F28705','#D9946C'))

fig.update_layout(title='<b>         Degree</b>',title_font_size=25,
                  margin = dict(t=100, l=100, r=50, b=100),
                  height=700, width=700)
fig.update_traces(hovertemplate='<b>Name</b>: %{id}<br>'+
                                '<b>Count</b>: %{value}<br>'+
                                '<b>Parent</b>: %{parent}') 
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.2,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# <h3> Plus we could see the advantages of Plotly in this graph. 
# </h3>
# 
# Matplotlib draws a static graph, but Plotly can dynamically click and move, and it supports zooming out, zooming in, and downloading graphs.
#     
# Because all of our graphs are made of plotly, the viewer can represent or remove items in the graph if desired. 
#    **With a click**

# <h2> East Asia Degree Ratio: Bar plot</h2>
# 
# 40% of master's degrees or higher, and respondents have a high educational background.
# 
# - China and Japan have similar trends to East Asia and the World. <br>
#     The number of people itself is large, so a representative trend seems to appear here. <br>
#     However, it is noteworthy that the two countries have the same tendency. <br>
# 
# 
# - Korea: It is the only country among the four countries with a high degree of education below Ph.D., bachelor's degree, and junior college. Only masters are low. 
#     (Polarization of education?)
# 
# - Taiwan: 1st place in master's ratio (55%), 2nd place in Ph.D. or higher (13.8%).
#     = The highest level of education.

# In[ ]:


#data preprocessing
df21Edu_Ea = df21_Ea.loc[:,['Q3','Q4']].reset_index().rename(columns={'Q3':'East_Asia', 'Q4':'Dgree'}).fillna('etc')
df21Edu_Ea =(df21Edu_Ea.replace({'I prefer not to answer':'etc'}).replace(['No formal education past high school',
                       'Some college/university study without earning a bachelor’s degree'],'~college')
             .replace(['Doctoral degree',
                       'Professional doctorate'],'Doctoral degree~'))

df21Edu_Ea= (df21Edu_Ea
           .groupby(['East_Asia', 'Dgree'])
           .size()
           .reset_index()
           .rename(columns = {0:"Count"}))

# 연령-지역 %
dfKo_Edu21= df21Edu_Ea[df21Edu_Ea['East_Asia']=='South Korea']
dfKo_Edu21['%']=((dfKo_Edu21['Count'] / dfKo_Edu21['Count'].sum()*100)).round(2)
dfKo_Edu21=dfKo_Edu21.sort_values(by='%', ascending=False)
dfTw_Edu21= df21Edu_Ea[df21Edu_Ea['East_Asia']=='Taiwan']
dfTw_Edu21['%']=((dfTw_Edu21['Count'] / dfTw_Edu21['Count'].sum())*100).round(2)
dfTw_Edu21=dfTw_Edu21.sort_values(by='%', ascending=False)
dfCh_Edu21= df21Edu_Ea[df21Edu_Ea['East_Asia']=='China']
dfCh_Edu21['%']=((dfCh_Edu21['Count'] / dfCh_Edu21['Count'].sum())*100).round(2)
dfCh_Edu21=dfCh_Edu21.sort_values(by='%', ascending=False)
dfJp_Edu21= df21Edu_Ea[df21Edu_Ea['East_Asia']=='Japan']
dfJp_Edu21['%']=((dfJp_Edu21['Count'] / dfJp_Edu21['Count'].sum())*100).round(2)
dfJp_Edu21=dfJp_Edu21.sort_values(by='%', ascending=False)

# #data 완성
# dfEdu_21_per = pd.concat([dfKo_Edu21, dfTw_Edu21, dfCh_Edu21, dfJp_Edu21], ignore_index = True)
# dfEdu_21_per= pd.pivot(dfEdu_21_per, index = "Dgree", columns = 'East_Asia', values = "%").reset_index()
# dfEdu_21_per

#graph
fig = make_subplots(rows = 1, cols = 4, 
                    shared_yaxes=True, 
                    vertical_spacing = 0.05)

fig.add_trace(go.Bar(x = dfCh_Edu21['Dgree'], 
                     y = dfCh_Edu21['%'], 
                     text = dfCh_Edu21['%'].astype(str) + "%", 
                     textposition='outside',
                     name='China',
                     marker_color='#88BFBA'), 
                     row = 1, col = 1)

fig.add_trace(go.Bar(x = dfJp_Edu21['Dgree'], 
                     y = dfJp_Edu21['%'], 
                     text = dfJp_Edu21['%'].astype(str) + "%", 
                     textposition='outside',
                     name='Japan',
                     marker_color='#CDD9A3'), 
                      row = 1, col = 2)

fig.add_trace(go.Bar(x = dfKo_Edu21['Dgree'], 
                     y = dfKo_Edu21['%'], 
                     text = dfKo_Edu21['%'].astype(str) + "%", 
                     textposition='outside',
                     name='South Korea',
                     marker_color='#F28705'), 
                      row = 1, col = 3)

fig.add_trace(go.Bar(x = dfTw_Edu21['Dgree'], 
                     y = dfTw_Edu21['%'], 
                     text = dfTw_Edu21['%'].astype(str) + "%", 
                     textposition='outside',
                     name='Taiwan',
                     marker_color='#D9946C'), 
                     row = 1, col = 4)

fig.update_layout(showlegend=True,title='<b>Degree in East Asia</b>',title_font_size=22,
                  margin = dict(t=200, l=100, r=50, b=200),
                  height=700, width=700)
fig.update_traces(hovertemplate='<b>Percent</b>: %{y}%<br>'+
                                '<b>Degree</b>: %{x}<br>')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.1,
    xanchor="right",
    x=1))
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.5,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# # 3.1.6 Experience transformation
# ---

# <h2> Trends in World & East Asia Career: Stacked Scatter plot </h2>
# - < 2 years: 50% of the total. <br>
# - 3-5 years: Decrease in the world, maintain East Asia ratio <br>
# - 2021 'etc data' disappeared. <br>

# In[ ]:


#Exp data 전처리
# Exp 뽑아오기
Exp21_Wo = df21.loc[:,['Q3','Q6', 'year']].reset_index().rename(columns={'Q3':'Country', 'Q6':'Exp'}).fillna('etc')
Exp20_Wo = df20.loc[:,['Q3','Q6','year']].reset_index().rename(columns={'Q3':'Country', 'Q6':'Exp'}).fillna('etc')
Exp19_Wo = df19.loc[:,['Q3','Q15','year']].reset_index().rename(columns={'Q3':'Country', 'Q15':'Exp'}).fillna('etc')
Exp18_Wo = df18.loc[:,['Q3','Q8','year']].reset_index().rename(columns={'Q3':'Country', 'Q8':'Exp'}).fillna('etc')
Exp17_Wo = df17.loc[:,['Country','Tenure', 'year']].reset_index().rename(columns={'Country':'Country', 'Tenure':'Exp'}).fillna('etc')

Exp21_Wo= Exp21_Wo.replace({'I have never written code': '< 1 years',  '1-3 years': '1-2 years'}).replace(['10-20 years', '20+ years'], '10+ years' )
Exp20_Wo= Exp20_Wo.replace({'I have never written code': '< 1 years'}).replace(['10-20 years', '20+ years'], '10+ years' )
Exp19_Wo= Exp19_Wo.replace({'I have never written code': '< 1 years'}).replace(['10-20 years', '20+ years'], '10+ years' )
Exp18_Wo= (Exp18_Wo.replace({'0-1': '< 1 years', '1-2': '1-2 years', '5-10':'5-10 years'})
        .replace(['2-3', '3-4',  '4-5'],'3-5 years')
       .replace(['10-15', '15-20','20-25', '30 +','25-30'],'10+ years'))
Exp17_Wo=(Exp17_Wo.replace({'More than 10 years':'10+ years', '1 to 2 years':'1-2 years',  'Less than a year':'< 1 years',
       '3 to 5 years':'3-5 years', "I don't write code to analyze data":'< 1 years',
       '6 to 10 years':'5-10 years'}))
                                                                                                                   
#data 정제(한꺼번에 이름바꾸기)
Exp5y_Wo= pd.concat([Exp17_Wo, Exp18_Wo, Exp19_Wo, Exp20_Wo, Exp21_Wo]).reset_index()
Exp5y_Wo=(Exp5y_Wo.groupby(['year', 'Exp'])
           .size()
           .reset_index()
           .rename(columns = {0:"Count"}))

#percent data 넣기
Exp21_per_W= Exp5y_Wo[Exp5y_Wo['year'] == "2021"].reset_index(drop = True)
Exp21_per_W['percentage'] = Exp21_per_W["Count"] / Exp21_per_W["Count"].sum()
Exp21_per_W['%'] = np.round(Exp21_per_W['percentage'] * 100, 1)

Exp20_per_W = Exp5y_Wo[Exp5y_Wo['year'] == "2020"].reset_index(drop = True)
Exp20_per_W['percentage'] = Exp20_per_W["Count"] / Exp20_per_W["Count"].sum()
Exp20_per_W['%'] = np.round(Exp20_per_W['percentage'] * 100, 1)

Exp19_per_W = Exp5y_Wo[Exp5y_Wo['year'] == "2019"].reset_index(drop = True)
Exp19_per_W['percentage'] = Exp19_per_W["Count"] / Exp19_per_W["Count"].sum()
Exp19_per_W['%'] = np.round(Exp19_per_W['percentage'] * 100, 1)

Exp18_per_W = Exp5y_Wo[Exp5y_Wo['year'] == "2018"].reset_index(drop = True)
Exp18_per_W['percentage'] = Exp18_per_W["Count"] / Exp18_per_W["Count"].sum()
Exp18_per_W['%'] = np.round(Exp18_per_W['percentage'] * 100, 1)

Exp17_per_W = Exp5y_Wo[Exp5y_Wo['year'] == "2017"].reset_index(drop = True)
Exp17_per_W['percentage'] = Exp17_per_W["Count"] / Exp17_per_W["Count"].sum()
Exp17_per_W['%'] = np.round(Exp17_per_W['percentage'] * 100, 1)

#data 완성
Exp5y_per_W = pd.concat([Exp17_per_W, Exp18_per_W, Exp19_per_W, Exp20_per_W, Exp21_per_W], ignore_index = True)
Exp5y_per_W= pd.pivot(Exp5y_per_W, index = "year", columns = 'Exp', values = "%").reset_index()
Exp5y_per_W.fillna('0')
Exp5y_percent_order = Exp5y_per_W['year'].tolist()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x = Exp5y_percent_order, 
    y = Exp5y_per_W['< 1 years'].tolist(), 
    mode = "lines", 
    name = '< 1 years',
    line = dict(width = 0.5),
    stackgroup = "one",
    marker_color='#F2798F'))
fig.add_trace(go.Scatter(
    x = Exp5y_percent_order, 
    y = Exp5y_per_W['1-2 years'].tolist(), 
    mode = "lines", 
    name = '1-2 years',
    line = dict(width = 0.5),
    stackgroup = "one",
    marker_color='#88BFBA'))
fig.add_trace(go.Scatter(
    x = Exp5y_percent_order, 
    y = Exp5y_per_W['3-5 years'].tolist(), 
    mode = "lines", 
    name = '3-5 years',
    line = dict(width = 0.5),
    stackgroup = "one",
    marker_color='#CDD9A3'))
fig.add_trace(go.Scatter(
    x = Exp5y_percent_order, 
    y = Exp5y_per_W['5-10 years'].tolist(), 
    mode = "lines", 
    name = '5-10 years',
    line = dict(width = 0.5),
    stackgroup = "one",
    marker_color='#F28705'))
fig.add_trace(go.Scatter(
    x = Exp5y_percent_order, 
    y = Exp5y_per_W['10+ years'].tolist(), 
    mode = "lines", 
    name = '10+ years',
    line = dict(width = 0.5),
    stackgroup = "one",
    marker_color='#D9946C'))

fig.add_trace(go.Scatter(
    x = Exp5y_percent_order, 
    y = Exp5y_per_W['etc'].tolist(), 
    mode = "lines", 
    name = 'etc',
    line = dict(width = 1),
    stackgroup = "one",
    marker_color='#F2D64B'))

fig.update_traces(hovertemplate='<b>Percent</b>: %{y}%<br>')
fig.update_layout(yaxis_range = (0, 100), title_font_size=20,
                  title_text="<b>Experience in world</b>",
                  height=500, width=700,
                  title_x=0.5)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.2,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# In[ ]:


#data preprocessing
Exp21 = df21_Ea.loc[:,['Q3','Q6', 'year']].reset_index().rename(columns={'Q3':'East_Asia', 'Q6':'Exp'}).fillna('etc')
Exp20 = df20_Ea.loc[:,['Q3','Q6','year']].reset_index().rename(columns={'Q3':'East_Asia', 'Q6':'Exp'}).fillna('etc')
Exp19 = df19_Ea.loc[:,['Q3','Q15','year']].reset_index().rename(columns={'Q3':'East_Asia', 'Q15':'Exp'}).fillna('etc')
Exp18 = df18_Ea.loc[:,['Q3','Q8','year']].reset_index().rename(columns={'Q3':'East_Asia', 'Q8':'Exp'}).fillna('etc')
Exp17 = df17_Ea.loc[:,['Country','Tenure', 'year']].reset_index().rename(columns={'Country':'East_Asia', 'Tenure':'Exp'}).fillna('etc')

Exp21Uni=['3-5 years', '< 1 years', '1-3 years', '10-20 years',
       'I have never written code', '5-10 years', '20+ years']
Exp20Uni= ['3-5 years', '< 1 years', '5-10 years', '1-2 years', 'etc',
       '20+ years', '10-20 years', 'I have never written code']
Exp19Uni=['1-2 years', '5-10 years', '< 1 years',
       'I have never written code', '3-5 years', '10-20 years',
       '20+ years', 'etc']
Exp18Uni=['0-1', '2-3', '1-2', '5-10', '3-4', '10-15', '15-20', '4-5',
       '20-25', '30 +', 'etc', '25-30']
Exp17Uni=['More than 10 years', '1 to 2 years', 'etc', 'Less than a year',
       '3 to 5 years', "I don't write code to analyze data",
       '6 to 10 years']

Exp21= Exp21.replace({'I have never written code': '< 1 years',  '1-3 years': '1-2 years'}).replace(['10-20 years', '20+ years'], '10+ years' )
Exp20= Exp20.replace({'I have never written code': '< 1 years'}).replace(['10-20 years', '20+ years'], '10+ years' )
Exp19= Exp19.replace({'I have never written code': '< 1 years'}).replace(['10-20 years', '20+ years'], '10+ years' )
Exp18= (Exp18.replace({'0-1': '< 1 years', '1-2': '1-2 years', '5-10':'5-10 years'})
        .replace(['2-3', '3-4',  '4-5'],'3-5 years')
       .replace(['10-15', '15-20','20-25', '30 +','25-30'],'10+ years'))
Exp17=(Exp17.replace({'More than 10 years':'10+ years', '1 to 2 years':'1-2 years',  'Less than a year':'< 1 years',
       '3 to 5 years':'3-5 years', "I don't write code to analyze data":'< 1 years',
       '6 to 10 years':'5-10 years'}))
                                                                                                                
Exp5y= pd.concat([Exp17, Exp18, Exp19, Exp20, Exp21]).reset_index()
Exp5y=(Exp5y.groupby(['year', 'Exp'])
           .size()
           .reset_index()
           .rename(columns = {0:"Count"}))

Exp21_percent = Exp5y[Exp5y['year'] == "2021"].reset_index(drop = True)
Exp21_percent['percentage'] = Exp21_percent["Count"] / Exp21_percent["Count"].sum()
Exp21_percent['%'] = np.round(Exp21_percent['percentage'] * 100, 1)
Exp21_percent

Exp20_percent = Exp5y[Exp5y['year'] == "2020"].reset_index(drop = True)
Exp20_percent['percentage'] = Exp20_percent["Count"] / Exp20_percent["Count"].sum()
Exp20_percent['%'] = np.round(Exp20_percent['percentage'] * 100, 1)
Exp20_percent

Exp19_percent = Exp5y[Exp5y['year'] == "2019"].reset_index(drop = True)
Exp19_percent['percentage'] = Exp19_percent["Count"] / Exp19_percent["Count"].sum()
Exp19_percent['%'] = np.round(Exp19_percent['percentage'] * 100, 1)
Exp19_percent

Exp18_percent = Exp5y[Exp5y['year'] == "2018"].reset_index(drop = True)
Exp18_percent['percentage'] = Exp18_percent["Count"] / Exp18_percent["Count"].sum()
Exp18_percent['%'] = np.round(Exp18_percent['percentage'] * 100, 1)
Exp18_percent

Exp17_percent = Exp5y[Exp5y['year'] == "2017"].reset_index(drop = True)
Exp17_percent['percentage'] = Exp17_percent["Count"] / Exp17_percent["Count"].sum()
Exp17_percent['%'] = np.round(Exp17_percent['percentage'] * 100, 1)
Exp17_percent


#graph
Exp5y_percent = pd.concat([Exp17_percent, Exp18_percent, Exp19_percent, Exp20_percent, Exp21_percent], ignore_index = True)
Exp5y_percent= pd.pivot(Exp5y_percent, index = "year", columns = 'Exp', values = "%").reset_index()
Exp5y_percent.fillna('0')

Exp5y_percent_order = Exp5y_percent['year'].tolist()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x = Exp5y_percent_order, 
    y = Exp5y_percent['< 1 years'].tolist(), 
    mode = "lines", 
    name = '< 1 years',
    line = dict(width = 0.5),
    stackgroup = "one",
    marker_color='#F2798F'))
fig.add_trace(go.Scatter(
    x = Exp5y_percent_order, 
    y = Exp5y_percent['1-2 years'].tolist(), 
    mode = "lines", 
    name = '1-2 years',
    line = dict(width = 0.5),
    stackgroup = "one",
    marker_color='#88BFBA'))
fig.add_trace(go.Scatter(
    x = Exp5y_percent_order, 
    y = Exp5y_percent['3-5 years'].tolist(), 
    mode = "lines", 
    name = '3-5 years',
    line = dict(width = 0.5),
    stackgroup = "one",
    marker_color='#CDD9A3'))
fig.add_trace(go.Scatter(
    x = Exp5y_percent_order, 
    y = Exp5y_percent['5-10 years'].tolist(), 
    mode = "lines", 
    name = '5-10 years',
    line = dict(width = 0.5),
    stackgroup = "one",
    marker_color='#F28705'))
fig.add_trace(go.Scatter(
    x = Exp5y_percent_order, 
    y = Exp5y_percent['10+ years'].tolist(), 
    mode = "lines", 
    name = '10+ years',
    line = dict(width = 0.5),
    stackgroup = "one",
    marker_color='#D9946C'))
fig.add_trace(go.Scatter(
    x = Exp5y_percent_order, 
    y = Exp5y_percent['etc'].tolist(), 
    mode = "lines", 
    name = 'etc',
    line = dict(width = 0.5),
    stackgroup = "one",
    marker_color='#F2D64B'))
fig.update_traces(hovertemplate='<b>Percent</b>: %{y}%<br>')
fig.update_layout(yaxis_range = (0, 100),
                  title_text="<b>Experience in East Asia</b>",
                  height=500, width=700, title_font_size=20,
                  title_x=0.5)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.2,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# # 3.1.7 Salary transformation
# ---

# <h2> World & East Asia Annual salary: Bar-H plot
# </h2>
# 
# - \$ 200,000 ~ : World (2.9%) is more than 50% compared to East Asia (1.3%)
# - \$ ~250,000 : World (59.2%) is less than East Asia (50.3%) <br>
#     = East Asia's annual salary gap between rich and poor is less.
# - \$ 25,000~60,000: The highest section in East Asia at 24%. <br>
# = The annual salary section that we aim for.

# In[ ]:


#data preprocessing
df21_salary_=df21['Q25'].value_counts().to_frame().rename(index={'$0-999':'<999','>$1,000,000':'1,000,000~','$500,000-999,999':'500,000-999,999'}).fillna(0)
df21_Ea_salary_=df21_Ea['Q25'].value_counts().to_frame().rename(index={'$0-999':'<999','>$1,000,000':'1,000,000~','$500,000-999,999':'500,000-999,999'}).fillna(0)

#퍼센트
df21_salary__=(df21_salary_['Q25']/(df21_salary_['Q25'].sum())*100).round(1).to_frame().rename(columns={'Q25':'World'})
df21_Ea_salary__=(df21_Ea_salary_['Q25']/(df21_Ea_salary_['Q25'].sum())*100).round(1).to_frame().rename(columns={'Q25':'EA'})

#그룹화
df21_salary=(df21_salary__.rename(index=
                               {'1,000-1,999':'1,000-7,499',
                                '2,000-2,999':'1,000-7,499',
                                '3,000-3,999':'1,000-7,499',
                                '4,000-4,999':'1,000-7,499',
                                '5,000-7,499':'1,000-7,499'})
                                .rename(index={'7,500-9,999':'7,500-24,999',
                                '10,000-14,999':'7,500-24,999',
                                '15,000-19,999':'7,500-24,999',
                                '20,000-24,999':'7,500-24,999' })
                                .rename(index={'25,000-29,999':'25,000-59,999',
                                 '30,000-39,999':'25,000-59,999',
                                 '40,000-49,999':'25,000-59,999',
                                 '50,000-59,999':'25,000-59,999'})
                                .rename(index={'60,000-69,999':'60,000-99,999',
                                 '70,000-79,999':'60,000-99,999',
                                 '80,000-89,999':'60,000-99,999',
                                 '90,000-99,999':'60,000-99,999'})
                                .rename(index={'100,000-124,999':'100,000-199,999',
                                 '125,000-149,999':'100,000-199,999',
                                 '150,000-199,999':'100,000-199,999'})
                                .rename(index={'200,000-249,999':'200,000-1,000,000~',
                                 '250,000-299,999':'200,000-1,000,000~',
                                 '300,000-499,999':'200,000-1,000,000~',
                                 '500,000-999,999':'200,000-1,000,000~',
                                 '1,000,000~':'200,000-1,000,000~'})
                                .reset_index().groupby('index').sum()
                                 .reindex(index = ['<999', 
                                                  '1,000-7,499',
                                                  '7,500-24,999', 
                                                  '25,000-59,999', 
                                                  '60,000-99,999', 
                                                 '100,000-199,999', 
                                                 '200,000-1,000,000~']))

df21_Ea_salary=(df21_Ea_salary__.rename(index=
                               {'1,000-1,999':'1,000-7,499',
                               '2,000-2,999':'1,000-7,499',
                               '3,000-3,999':'1,000-7,499',
                               '4,000-4,999':'1,000-7,499',
                               '5,000-7,499':'1,000-7,499'})
                                .rename(index={'7,500-9,999':'7,500-24,999',
                               '10,000-14,999':'7,500-24,999',
                               '15,000-19,999':'7,500-24,999',
                               '20,000-24,999':'7,500-24,999'})
                                .rename(index={'25,000-29,999':'25,000-59,999',
                               '30,000-39,999':'25,000-59,999',
                               '40,000-49,999':'25,000-59,999',
                               '50,000-59,999':'25,000-59,999'})
                                .rename(index={'60,000-69,999':'60,000-99,999',
                               '70,000-79,999':'60,000-99,999',
                               '80,000-89,999':'60,000-99,999',
                               '90,000-99,999':'60,000-99,999'})
                                .rename(index={'100,000-124,999':'100,000-199,999',
                               '125,000-149,999':'100,000-199,999',
                               '150,000-199,999':'100,000-199,999'})
                                .rename(index={'200,000-249,999':'200,000-1,000,000~',
                               '250,000-299,999':'200,000-1,000,000~',
                               '300,000-499,999	':'200,000-1,000,000~',
                               '500,000-999,999':'200,000-1,000,000~',
                               '1,000,000~':'200,000-1,000,000~'})
                                .reset_index().groupby('index').sum()
                                .reindex(index = ['<999', 
                                                  '1,000-7,499',
                                                  '7,500-24,999', 
                                                  '25,000-59,999', 
                                                  '60,000-99,999', 
                                                 '100,000-199,999', 
                                                 '200,000-1,000,000~']))

#graph
World = df21_salary['World'].values
East_Asia = df21_Ea_salary['EA'].values
y = df21_salary.index

fig = go.Figure(data=[
    go.Bar(y=y, x=World, orientation='h', name="World", base=0, hovertemplate='<b>World</b>: %{x}%<br>', marker_color='#979DA6'),
    go.Bar(y=y, x=-East_Asia, orientation='h', name="East Asia", base=0, hovertemplate='<b>East Asia</b>: %{x}%<br>', marker_color='#F2D64B')
    ])

fig.update_layout(barmode='stack')
fig.update_layout(
    margin=dict(l=200, r=0, t=200, b=100),
    autosize=False,
    title_text="<b>                       Salary in East Asia vs World</b>", height=600, width=700, title_font_size=20, title_x=0.5)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.1,
    xanchor="right",
    x=1))
fig.show()


# <h2> World experience and annual salary: Heat Map </h2>
# 
# <h4>Relatively **positive correlation.** </h4>
# 
# - Even with 5-10 years of experience, more than 45% has an annual salary of less than $20,000
# 
# - With more than 10 years of experience, more than 30% receive an annual salary of $100,000.

# In[ ]:


#data preprocessing
SalExp21= df21.loc[:, ['region', 'Q25', 'Q6']].rename(columns={'Q6':'Exp', 'Q25':'Salary'})


SalExp21=(SalExp21
          .replace(['0-999','$0-999','0'], '< 999')
          .replace({'>$1,000,000':'200,000~'})
          .replace(['1,000-1,999','2,000-2,999','3,000-3,999', 
        '4,000-4,999','5,000-7,499','7,500-9,999','10,000-14,999', '15,000-19,999'],'1,000-20,000')
          .replace(['20,000-24,999''25,000-29,999','30,000-39,999', '40,000-49,999', 
        '50,000-59,999'],'20,000-59,999') 
           .replace(['60,000-69,999', '70,000-79,999', '80,000-89,999', 
        '90,000-99,999'], '60,000-99,999')
          .replace(['100,000-124,999', '300,000-499,999',
        '125,000-149,999', '125,000-149,999',
        '150,000-199,999'],'100,000-199,999')
          .replace(['200,000-249,999', '250,000-299,999', 
        '1,000,000','$500,000-999,999'], '200,000~')
        .replace({'I have never written code': '< 1 years'})
          .replace(['10-20 years', '20+ years'], '10+ years' )
         )

sal_order=['< 999', '1,000-20,000', '20,000-59,999', '60,000-99,999','100,000-199,999', '200,000~']
Exp21_order=['< 1 years', '1-3 years','3-5 years', '5-10 years', '10+ years' ]



SalExp21_Ea = SalExp21[SalExp21['region'] == "EastAsia"].reset_index(drop = True)
SalExp21_Ea=(SalExp21_Ea.groupby(['Exp', 'Salary'])
           .size()
          .unstack().fillna(0).astype('int64'))

SalExp21_Wo = SalExp21[SalExp21['region'] == "World"].reset_index(drop = True)
SalExp21_Wo=(SalExp21_Wo.groupby(['Exp', 'Salary'])
           .size()
          .unstack().fillna(0).astype('int64'))
SalExp21_Wo



#graph
#World
z = SalExp21_Wo
z = z[sal_order]
z = z.reindex(Exp21_order)

z_data = z.apply(lambda x:np.round(x/x.sum()*100, 2), axis = 1).to_numpy() # convert to correlation matrix
x = sal_order
y = Exp21_order

fig = ff.create_annotated_heatmap(z_data, x = x, y = y, colorscale = "sunset")
fig.update_layout( title_text="<b>Experience and salary in World</b>",
                  height=700, width=700, title_font_size=20,
                  title_x=0.5,
                  margin=dict(l=100, r=100, t=200, b=100))

fig.add_annotation(dict(font=dict(size=14),
                                    x=0.85,
                                    y=-0.1,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


#East Asia
z = SalExp21_Ea
z = z[sal_order]
z = z.reindex(Exp21_order)
z_data = z.apply(lambda x:np.round(x/x.sum(), 2), axis = 1).to_numpy() # convert to correlation matrix
x = sal_order
y = Exp21_order

fig = ff.create_annotated_heatmap(z_data, x = x, y = y, colorscale = "sunset")
fig.update_layout(title_text="<b>Experience and salary in East Asia</b>",
                  height=700, width=700, title_font_size=20,
                  title_x=0.5,
                  margin=dict(l=100, r=100, t=200, b=100))

fig.add_annotation(dict(font=dict(size=14),
                                    x=0.85,
                                    y=-0.1,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# <h2> World & East Asia Degree/Annual salary: Heat Map </h2>
# 
# - \\$ ~20,000 : Regardless of degree, about 40% of the annual salary is \$ 20,000 or less. <br>
#     Guess it's the ratio that comes from a student. <br>
# - \$ 25,000-100,000 : Earned more than 40% with a bachelor's degree alone in East Asia  <br>
#     (World: less than 20%) <br>
# - \$ 200,000~ : Even with a doctorate or higher, it is difficult to obtain it from East Asia.

# In[ ]:


#data preprocessing
Salary21= df21.loc[:, ['region', 'Q25', 'year']].rename(columns={'Q3':'Country', 'Q25':'Salary'})
salary21_Index=['< 999', '1,000-20,000', '20,000-59,999', '60,000-99,999','100,000-199,999', '200,000~']

Salary21=(Salary21
          .replace(['0-999','$0-999','0'], '< 999')
          .replace({'>$1,000,000':'200,000~'})
          .replace(['1,000-1,999','2,000-2,999','3,000-3,999', '4,000-4,999','5,000-7,499','7,500-9,999','10,000-14,999', '15,000-19,999'],'1,000-20,000')
          .replace(['20,000-24,999''25,000-29,999','30,000-39,999', '40,000-49,999',  '50,000-59,999'],'20,000-59,999') 
           .replace(['60,000-69,999', '70,000-79,999', '80,000-89,999','90,000-99,999'], '60,000-99,999')
          .replace(['100,000-124,999', '300,000-499,999', '125,000-149,999', '125,000-149,999', '150,000-199,999'],'100,000-199,999')
          .replace(['200,000-249,999', '250,000-299,999','1,000,000','$500,000-999,999'], '200,000~')).fillna('0')
sal_order=['< 999', '1,000-20,000', '20,000-59,999', '60,000-99,999','100,000-199,999', '200,000~']

Salary21=(Salary21.groupby(['region', 'Salary'])
           .size()
           .reset_index()
           .rename(columns = {0:"Count"}))

Salary21_Ea = Salary21[Salary21['region'] == "EastAsia"].reset_index(drop = True)
Salary21_Ea['%']=((Salary21_Ea['Count'] / Salary21_Ea['Count'].sum())*100).round(2)
Salary21_Wo = Salary21[Salary21['region'] == "World"].reset_index(drop = True)
Salary21_Wo['%']=((Salary21_Wo['Count'] / Salary21_Wo['Count'].sum())*100).round(2)

Dgr_Sal_21= df21.loc[:, ['region', 'Q25', 'Q4']].rename(columns={'Q4':'Dgree', 'Q25':'Salary'})
Dgr_Sal_21 = (Dgr_Sal_21.replace(['0-999','$0-999','0'], '< 999')
          .replace({'>$1,000,000':'200,000~'})
          .replace(['1,000-1,999','2,000-2,999','3,000-3,999', '4,000-4,999','5,000-7,499','7,500-9,999','10,000-14,999', '15,000-19,999'],'1,000-20,000')
          .replace(['20,000-24,999''25,000-29,999','30,000-39,999', '40,000-49,999', '50,000-59,999'],'20,000-59,999') 
          .replace(['60,000-69,999', '70,000-79,999', '80,000-89,999', '90,000-99,999'], '60,000-99,999')
          .replace(['100,000-124,999', '300,000-499,999', '125,000-149,999', '125,000-149,999','150,000-199,999'],'100,000-199,999')
          .replace(['200,000-249,999', '250,000-299,999','1,000,000','$500,000-999,999'], '200,000~')
          .replace({'I prefer not to answer':'etc'})
          .replace(['No formal education past high school', 'Some college/university study without earning a bachelor’s degree'],'~college')
          .replace(['Doctoral degree', 'Professional doctorate'],'Doctoral degree~'))


#EastAsia 뽑기
Dgr_Sal_21_Ea= Dgr_Sal_21[Dgr_Sal_21['region'] == "EastAsia"].reset_index(drop = True)
Dgr_Sal_21_Ea = Dgr_Sal_21_Ea.groupby(['Dgree', 'Salary']).size().unstack().fillna(0).astype('int64')

dgree_order=[ '~college','Bachelor’s degree', 'Master’s degree', 'Doctoral degree~', 'etc']


#graph
#World
z = Dgr_Sal_21.groupby(['Dgree', 'Salary']).size().unstack().fillna(0).astype('int64')
z = z[sal_order]
z = z.reindex(dgree_order)

z_data = z.apply(lambda x:np.round(x/x.sum()*100, 2), axis = 1).to_numpy() # convert to correlation matrix
x = sal_order
y = dgree_order

fig = ff.create_annotated_heatmap(z_data, x = x, y = y, colorscale = "sunset")
fig.update_layout( title_text="<b>    Degree-Salary in World</b>",
                  height=700, width=700, title_font_size=20,
                  title_x=0.5,
                  margin=dict(l=150, r=100, t=200, b=50))
fig.update_traces(hovertemplate='<b>Degree</b>: %{y}<br>'+
                                '<b>Salary</b>: %{x}<br>'+
                                '<b>Percent</b>: %{z}%')
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.1,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


#East Asia
z = Dgr_Sal_21_Ea
z = z[sal_order]
z = z.reindex(dgree_order)
z_data = z.apply(lambda x:np.round(x/x.sum()*100, 2), axis = 1).to_numpy() # convert to correlation matrix
x = sal_order
y = dgree_order

fig = ff.create_annotated_heatmap(z_data, x = x, y = y, colorscale = "sunset")
fig.update_layout(title_text="<b>    Degree-Salary in East Asia</b>",
                  height=700, width=700, title_font_size=20,
                  title_x=0.5,
                  margin=dict(l=150, r=100, t=200, b=50))
fig.update_traces(hovertemplate='<b>Degree</b>: %{y}<br>'+
                                '<b>Salary</b>: %{x}<br>'+
                                '<b>Percent</b>: %{z}%')
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.1,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# # 3.1.8 Language transformation
# ---

# <h2>World & East Asia Programming Language: Bar plot </h2>
# - Python: 80% of the world and 85% of East Asia use it.
# 
# <h4>
# <p style="color:#FF0000";>
# We've been working on the project as python, so I hope we can continue to learn python and become experienced Data Scientists!
#     </p></h4>

# In[ ]:


#data preprocessing
#world
programming_list = ["Python", "R", "SQL", "Java", "C", "Bash", "Javascript", "C++"]
programming_df = pd.Series(programming_list)

df_2019 = df19[df19['Q19'].isin(programming_df)]
df_2020 = df20[df20['Q8'].isin(programming_df)]
df_2021 = df21[df21['Q8'].isin(programming_df)]

df19Lag = df_2019.loc[:, ['region', 'Q5', 'Q19', 'year']]
df19Lag = df19Lag.rename(columns = {'Q19': 'Language'}, inplace = False) # To match with other datasets
df20Lag = df_2020.loc[:, ['region', 'Q5', 'Q8', 'year']].rename(columns = {'Q8': 'Language'}, inplace = False)
df21Lag = df_2021.loc[:, ['region', 'Q5', 'Q8', 'year']].rename(columns = {'Q8': 'Language'}, inplace = False)

df3y_Lag = pd.concat([df19Lag, df20Lag, df21Lag])
df3y_Lag = df3y_Lag.groupby(['year', 'Language']).size().reset_index().rename(columns = {0:"Count"})
df3y_Lag

# 2019
dfLang_19 = df3y_Lag[df3y_Lag['year'] == "2019"].reset_index(drop = True)
dfLang_19['percentage'] = dfLang_19["Count"] / dfLang_19["Count"].sum()
dfLang_19['%'] = np.round(dfLang_19['percentage'] * 100, 1)

# 2020
dfLang_20 = df3y_Lag[df3y_Lag['year'] == "2020"].reset_index(drop = True)
dfLang_20['percentage'] = dfLang_20["Count"] / dfLang_20["Count"].sum()
dfLang_20['%'] = np.round(dfLang_20['percentage'] * 100, 1)

# 2021
dfLang_21 = df3y_Lag[df3y_Lag['year'] == "2021"].reset_index(drop = True)
dfLang_21['percentage'] = dfLang_21["Count"] / dfLang_21["Count"].sum()
dfLang_21['%'] = np.round(dfLang_21['percentage'] * 100, 1)

dfLang_19=dfLang_19.sort_values(by='%', ascending=False)
dfLang_20=dfLang_20.sort_values(by='%', ascending=False)
dfLang_21=dfLang_21.sort_values(by='%', ascending=False)

#graph
fig = go.Figure()

fig.add_trace(go.Bar(x = dfLang_19['Language'], 
                     y = dfLang_19['%'], 
                     name = "2019", 
                     text = dfLang_19['%'].astype(str) + "%", 
                     textposition='auto', marker_color='#CDD9A3'))

fig.add_trace(go.Bar(x = dfLang_20['Language'], 
                     y = dfLang_20['%'], 
                     name = "2020", 
                     text = dfLang_20['%'].astype(str) + "%", 
                     textposition='auto', marker_color='#F28705'))

fig.add_trace(go.Bar(x = dfLang_21['Language'], 
                     y = dfLang_21['%'], 
                     name = "2021", 
                     text = dfLang_21['%'].astype(str) + "%", 
                     textposition='auto', marker_color='#88BFBA'))
fig.update_layout(title='<b>Language in World</b>',title_font_size=20,
                  margin = dict(t=100, l=100, r=50, b=100),
                  height=600, width=700,
                  xaxis_title=None,
                  yaxis_title=None)
fig.update_traces(hovertemplate='<b>Percent</b>: %{y}%<br>'+
                                '<b>Language</b>: %{x}<br>')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_layout(legend=dict(
    orientation="v",
    yanchor="bottom",
    y=0.8,
    xanchor="right",
    x=1))
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.2,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# In[ ]:


#data prprocessing
#Ea

df_2019 = df19_Ea[df19_Ea['Q19'].isin(programming_df)]
df_2020 = df20_Ea[df20_Ea['Q8'].isin(programming_df)]
df_2021 = df21_Ea[df21_Ea['Q8'].isin(programming_df)]

df19Lag = df_2019.loc[:, ['region', 'Q5', 'Q19', 'year']]
df19Lag = df19Lag.rename(columns = {'Q19': 'Language'}, inplace = False) # To match with other datasets
df20Lag = df_2020.loc[:, ['region', 'Q5', 'Q8', 'year']].rename(columns = {'Q8': 'Language'}, inplace = False)
df21Lag = df_2021.loc[:, ['region', 'Q5', 'Q8', 'year']].rename(columns = {'Q8': 'Language'}, inplace = False)


df3y_Lag = pd.concat([df19Lag, df20Lag, df21Lag])
df3y_Lag = df3y_Lag.groupby(['year', 'Language']).size().reset_index().rename(columns = {0:"Count"})
df3y_Lag


# 2019
dfLang_19 = df3y_Lag[df3y_Lag['year'] == "2019"].reset_index(drop = True)
dfLang_19['percentage'] = dfLang_19["Count"] / dfLang_19["Count"].sum()
dfLang_19['%'] = np.round(dfLang_19['percentage'] * 100, 1)

# 2020
dfLang_20 = df3y_Lag[df3y_Lag['year'] == "2020"].reset_index(drop = True)
dfLang_20['percentage'] = dfLang_20["Count"] / dfLang_20["Count"].sum()
dfLang_20['%'] = np.round(dfLang_20['percentage'] * 100, 1)

# 2021
dfLang_21 = df3y_Lag[df3y_Lag['year'] == "2021"].reset_index(drop = True)
dfLang_21['percentage'] = dfLang_21["Count"] / dfLang_21["Count"].sum()
dfLang_21['%'] = np.round(dfLang_21['percentage'] * 100, 1)

dfLang_19=dfLang_19.sort_values(by='%', ascending=False)
dfLang_20=dfLang_20.sort_values(by='%', ascending=False)
dfLang_21=dfLang_21.sort_values(by='%', ascending=False)

#graph
fig = go.Figure()

fig.add_trace(go.Bar(x = dfLang_19['Language'], 
                     y = dfLang_19['%'], 
                     name = "2019", 
                     text = dfLang_19['%'].astype(str) + "%", 
                     textposition='auto', marker_color='#CDD9A3'))

fig.add_trace(go.Bar(x = dfLang_20['Language'], 
                     y = dfLang_20['%'], 
                     name = "2020", 
                     text = dfLang_20['%'].astype(str) + "%", 
                     textposition='auto', marker_color='#F28705'))

fig.add_trace(go.Bar(x = dfLang_21['Language'], 
                     y = dfLang_21['%'], 
                     name = "2021", 
                     text = dfLang_21['%'].astype(str) + "%", 
                     textposition='auto', marker_color='#88BFBA'))
fig.update_layout(title='<b>Language in EastAsia</b>',title_font_size=20,
                  margin = dict(t=100, l=100, r=50, b=100),
                  height=600, width=700,
                  xaxis_title=None,
                  yaxis_title=None)
fig.update_traces(hovertemplate='<b>Percent</b>: %{text}')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.2,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


#  # 3.2 Position of Data Scientist in East Asia
#  ---

# In[ ]:


# data preprocessing
df21_Ea_DS = df21_Ea[df21_Ea['Q5'].isin(Data_Scientist)].fillna(0)

salary_order= ['<999', '1,000-19,999', '20,000-59,999', '60,000-99,999','100,000-199,999', '200,000~']
dgree_order=[ '~college','Bachelor’s degree', 'Master’s degree', 'Doctoral degree~', 'etc']

df21_Ea_DS=(df21_Ea_DS
  #salary      
          .replace({'$0-999':'<999','>$1,000,000':'1,000,000~','$500,000-999,999':'500,000-999,999'})
         
          .replace(['1,000-1,999','2,000-2,999','3,000-3,999', '4,000-4,999','5,000-7,499','7,500-9,999','10,000-14,999', '15,000-19,999'],'1,000-19,999')
          .replace(['20,000-24,999','25,000-29,999','30,000-39,999', '40,000-49,999', '50,000-59,999'],'20,000-59,999') 
          .replace(['60,000-69,999', '70,000-79,999', '80,000-89,999', '90,000-99,999'], '60,000-99,999')
          .replace(['100,000-124,999','125,000-149,999','150,000-199,999'],'100,000-199,999')
          .replace(['200,000-249,999', '250,000-299,999', '300,000-499,999','500,000-999,999', '1,000,000~'], '200,000~')
  #degree          
          .replace({'I prefer not to answer':'etc'})
          .replace(['No formal education past high school','Some college/university study without earning a bachelor’s degree'],'~college')
          .replace(['Doctoral degree', 'Professional doctorate'],'Doctoral degree~')
          )
sal_order= ['<999', '1,000-19,999', '20,000-59,999', '60,000-99,999','100,000-199,999', '200,000~']
dgree_order=[ '~college','Bachelor’s degree', 'Master’s degree', 'Doctoral degree~', 'etc']



# # 3.2.1 Salary
# ---

# - Annual salary of Research Scientist.
# : The highest percentage of $2.6 million is 29.81%.
# 
# - The annual salary of Machine Learning Engineer.
# : The highest rate of $999 is 31.89%.
# 
# - The annual salary of Data Scientist is..
# : The ratio of $1,000 to $20,000 is the highest at 29.19%.
# 
# 
# <h2>⇒ The higher the annual salary, the lower the overall job rate.</h2>

# In[ ]:


df21_Ea_DS_= df21_Ea_DS.loc[:,['Q5','Q25']].reset_index().rename(columns={'Q5':'Data_Scientist', 'Q25':'Salary'}).fillna('etc')
df21_Ea_DS_= (df21_Ea_DS_.groupby(['Data_Scientist', 'Salary']).size()
                         .reset_index()
                         .rename(columns = {0:"Count"}))

#Data Scientist
df21_Ea_DS_Ds = df21_Ea_DS_[df21_Ea_DS_['Data_Scientist'] == "Data Scientist"].reset_index(drop = True)
df21_Ea_DS_Ds['%']=((df21_Ea_DS_Ds['Count'] / df21_Ea_DS_Ds['Count'].sum())*100).round(2)

#Machine Learning Engineer
df21_Ea_DS_Mle = df21_Ea_DS_[df21_Ea_DS_['Data_Scientist'] == "Machine Learning Engineer"].reset_index(drop = True)
df21_Ea_DS_Mle['%']=((df21_Ea_DS_Mle['Count'] / df21_Ea_DS_Mle['Count'].sum())*100).round(2)

#Research Scientist
df21_Ea_DS_Rs = df21_Ea_DS_[df21_Ea_DS_['Data_Scientist'] == "Research Scientist"].reset_index(drop = True)
df21_Ea_DS_Rs['%']=((df21_Ea_DS_Rs['Count'] / df21_Ea_DS_Rs['Count'].sum())*100).round(2)
df21_Ea_DS_Rs


df21_Ea_DS_salary = pd.concat([df21_Ea_DS_Ds, df21_Ea_DS_Mle, df21_Ea_DS_Rs], ignore_index = True)
df21_Ea_DS_salary= pd.pivot(df21_Ea_DS_salary, index = "Salary", columns = 'Data_Scientist', values = "%").reset_index().fillna('0')
df21_Ea_DS_salary= df21_Ea_DS_salary.set_index("Salary").reindex(sal_order)

#graph
fig = go.Figure()
fig.add_trace(go.Bar(x = df21_Ea_DS_salary.index, 
                     y = df21_Ea_DS_salary['Data Scientist'], 
                     name = "Data Scientist", 
                     text = df21_Ea_DS_salary['Data Scientist'].astype(str) + "%", 
                     textposition='auto', marker_color='#F2798F'))

fig.add_trace(go.Bar(x = df21_Ea_DS_salary.index, 
                     y = df21_Ea_DS_salary['Machine Learning Engineer'], 
                     name = "Machine Learning Engineer", 
                     text = df21_Ea_DS_salary['Machine Learning Engineer'].astype(str) + "%", 
                     textposition='auto', marker_color='#CDD9A3'))

fig.add_trace(go.Bar(x = df21_Ea_DS_salary.index, 
                     y = df21_Ea_DS_salary['Research Scientist'], 
                     name = "Research Scientist", 
                     text = df21_Ea_DS_salary['Research Scientist'].astype(str) + "%", 
                     textposition='auto', marker_color='#88BFBA'))

fig.update_layout(barmode='stack',
                 showlegend=True,
                 height=600, width=700,
                 title_text="<b>Data Scientist's Salary in East Asia</b>",
                 title_x=0.5,
                 title_font_size=20, 
                  margin=dict(l=100, r=100, t=100, b=100))
fig.update_traces(hovertemplate='<b>Percent</b>: %{y}%<br>'+
                                '<b>Salary</b>: %{x}$<br>')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_layout(legend=dict(
    orientation="v",
    yanchor="bottom",
    y=0.8,
    xanchor="right",
    x=1.2))

fig.show()


# # 3.2.2 Salary-Experience
# ---

# <h2>The correlation between the career of a Data Scientist and the annual salary.</h2>
# 
# If you don't have experience, you have the highest rate of $999.
# 
# Less than 1 year, 1-3 years have the highest percentage of $999.
# 
# The highest percentage of $20,000 to $60,000 in 3-10 years.
# 
# 10-20 years have the highest percentage of $60,000 to $100,000.

# In[ ]:


df21Ea_DS_ExSal = df21_Ea_DS.loc[:,['Q6','Q25']].reset_index().rename(columns={'Q25':'Salary', 'Q6':'Exp'}).fillna('etc')
df21Ea_DS_ExSal= (df21Ea_DS_ExSal.groupby(['Exp', 'Salary']).size().unstack().fillna(0).astype('int64'))

Exp_order=['< 1 years','1-3 years','3-5 years', '5-10 years', '10-20 years', '20+ years', 'I have never written code']

df21Ea_DS_ExSal

z = df21Ea_DS_ExSal
z = z[sal_order]
z = z.reindex(Exp_order)

z_data = z.apply(lambda x:np.round(x/x.sum()*100, 2), axis = 1).to_numpy() # convert to correlation matrix
x = sal_order
y = Exp_order

fig = ff.create_annotated_heatmap(z_data, x = x, y = y, colorscale = "sunset")
fig.update_layout(title_text="<b>    Data Scientist's Experience & Salary </b>",title_font_size=20,
                  height=700, width=700,
                  title_x=0.5,
                  margin=dict(l=100, r=100, t=200, b=100))
fig.update_traces(hovertemplate='<b>Salary</b>: %{y}<br>'+
                                '<b>Experience</b>: %{x}<br>'+
                                '<b>Percent</b>: %{z}%')

fig.show()


# # 3.2.3 Degree
# ---

# <h2>Comparison of educational background of Data Scientists. <br></h2>
#  <br>
# - It has the highest level of Master's Degrees. <br>
#  <br>
# - Next, Doctoral Degree, <br>
#  <br>
# - The figure was high in the order of Bachelor's Degree. <br>
# 

# In[ ]:


df21_Ea_degree = df21_Ea_DS['Q4'].value_counts().to_frame()
degree = df21_Ea_degree.index
values = df21_Ea_degree['Q4'].tolist()

colors = ['#F2798F','#88BFBA', '#CDD9A3', '#F28705', '#D9946C']
fig = go.Figure(data=[go.Bar(name='Degree', x=degree, y=values ,orientation='v', marker_color=colors, text=values, textposition='outside')])
fig.update_layout(title_text="<b>Data Scientist's Degree (2021)</b>", title_font_size=20,
                  height=600, width=700,
                  title_x=0.5,
                  margin=dict(l=100, r=100, t=200, b=100))
fig.update_traces(hovertemplate='<b>Count</b>: %{y}<br>'+
                                '<b>Degree</b>: %{x}<br>')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.2,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# # 3.2.4 Salary-Degree
# ---

# <h2>Relationship between Data Scientist's academic background and annual salary. <br></h2>
#  <br>
#  
# - If your educational background is below college, <br>
# : Less than 999 dollars. <br>
# <br>
# 
# - The lowest annual salary accounts for the highest percentage.<br>
#  <br>
# - Bachelor's degree, Master's Degree, Doctoral degree <br>
# :$2~60,000 dollars accounts for a large proportion <br>
# 
#  <br>
# <h2>⇒ The higher the education level, the higher the annual salary. <br></h2>

# In[ ]:


df21Ea_DS_EduSal= df21_Ea_DS.loc[:, ['Q4', 'Q25']].rename(columns={'Q4':'Edu', 'Q25':'Salary'})
df21Ea_DS_EduSal['Edu'].unique()
Edu_order=['~college', 'Bachelor’s degree','Master’s degree', 'Doctoral degree~', 'etc']

df21Ea_DS_EduSal= (df21Ea_DS_EduSal.groupby(['Edu', 'Salary']).size().unstack().fillna(0).astype('int64'))
df21Ea_DS_EduSal

z = df21Ea_DS_EduSal
z = z[sal_order]
z = z.reindex(Edu_order)

z_data = z.apply(lambda x:np.round(x/x.sum()*100, 2), axis = 1).to_numpy() # convert to correlation matrix
x = sal_order
y = Edu_order

fig = ff.create_annotated_heatmap(z_data, x = x, y = y, colorscale = "sunset")
fig.update_layout(title_text="<b>       Data Scientist's Degree & Salary </b>", title_font_size=20,
                  height=700, width=700,
                  title_x=0.5,
                  margin=dict(l=150, r=100, t=200, b=50))
fig.update_traces(hovertemplate='<b>Degree</b>: %{y}<br>'+
                                '<b>Salary</b>: %{x}<br>'+
                                '<b>Percent</b>: %{z}%')
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.1,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# # 3.2.5 Language
# ---

# <h2>The language that Data Scientist uses a lot. <br></h2>
#  <br>
# - Python accounts for the highest percentage of 80% or more. <br>
#  <br>
# - Second, I use R the most. <br>
# R is used less frequently in the order of 2019, 20, and 21. <br>
#  <br>
# - From 19 to 21, the percentage of use rate of use 10% -> 4%, a total of 6% decrease. <br>
#  <br>
# - The third most frequently used language is SQL. <br>
# SQL increased 0.6 percent in 2020 from 2021. <br>
#  <br>
# - The fourth most frequently used languages are C language and C++. <br>
#  <br>
# <h2>⇒ To become a Data Scientist, Let's study Python first! <br></h2>

# In[ ]:


#data preprocessing
df20_Ea_DS = df20_Ea[df20_Ea['Q5'].isin(Data_Scientist)]
df19_Ea_DS =df19_Ea[df19_Ea['Q5'].isin(Data_Scientist)]
df19Ea_DSLag = df19_Ea_DS.loc[:, [ 'Q5', 'Q19', 'year']]
df19Ea_DSLag = df19Ea_DSLag.rename(columns = {'Q19': 'Language'}, inplace = False) # To match with other datasets
df20Ea_DSLag = df20_Ea_DS.loc[:, [ 'Q5', 'Q8', 'year']].rename(columns = {'Q8': 'Language'}, inplace = False)
df21Ea_DSLag = df21_Ea_DS.loc[:, [ 'Q5', 'Q8', 'year']].rename(columns = {'Q8': 'Language'}, inplace = False)

df3y_Ds_Lag = pd.concat([df19Ea_DSLag, df20Ea_DSLag, df21Ea_DSLag])
df3y_Ds_Lag = df3y_Ds_Lag.groupby(['year', 'Language']).size().reset_index().rename(columns = {0:"Count"})
df3y_Ds_Lag

# 2019
dfLang_Ds_19 = df3y_Ds_Lag[df3y_Ds_Lag['year'] == "2019"].reset_index(drop = True)
dfLang_Ds_19['percentage'] = dfLang_Ds_19["Count"] / dfLang_Ds_19["Count"].sum()
dfLang_Ds_19['%'] = np.round(dfLang_Ds_19['percentage'] * 100, 1)

# 2020
dfLang_Ds_20 = df3y_Ds_Lag[df3y_Ds_Lag['year'] == "2020"].reset_index(drop = True)
dfLang_Ds_20['percentage'] = dfLang_Ds_20["Count"] / dfLang_Ds_20["Count"].sum()
dfLang_Ds_20['%'] = np.round(dfLang_Ds_20['percentage'] * 100, 1)

# 2021
dfLang_Ds_21 = df3y_Ds_Lag[df3y_Ds_Lag['year'] == "2021"].reset_index(drop = True)
dfLang_Ds_21['percentage'] = dfLang_Ds_21["Count"] / dfLang_Ds_21["Count"].sum()
dfLang_Ds_21['%'] = np.round(dfLang_Ds_21['percentage'] * 100, 1)

dfLang_Ds_19=dfLang_Ds_19.sort_values(by='%', ascending=False)
dfLang_Ds_20=dfLang_Ds_20.sort_values(by='%', ascending=False)
dfLang_Ds_21=dfLang_Ds_21.sort_values(by='%', ascending=False)

#graph
fig = go.Figure()

fig.add_trace(go.Bar(x = dfLang_Ds_19['Language'], 
                     y = dfLang_Ds_19['%'], 
                     name = "2019", 
                     text = dfLang_Ds_19['%'].astype(str) + "%", 
                     textposition='auto', 
                     marker_color='#CDD9A3'))

fig.add_trace(go.Bar(x = dfLang_Ds_20['Language'], 
                     y = dfLang_Ds_20['%'], 
                     name = "2020", 
                     text = dfLang_Ds_20['%'].astype(str) + "%", 
                     textposition='auto', 
                     marker_color='#F28705'))

fig.add_trace(go.Bar(x = dfLang_Ds_21['Language'], 
                     y = dfLang_Ds_21['%'], 
                     name = "2021", 
                     text = dfLang_Ds_21['%'].astype(str) + "%", 
                     textposition='auto', 
                     marker_color='#88BFBA'))

fig.update_layout(title='<b>        The language used by the data scientist</b>',title_font_size=22,
                  margin = dict(t=120, l=100, r=10, b=150),
                  height=600, width=700)
fig.update_traces(hovertemplate='<b>Percent</b>: %{y}%<br>'+
                                '<b>Language</b>: %{x}<br>')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=0.8,
    xanchor="right",
    x=1))
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.2,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# <h3> Parallel Categories Diagram <br></h3>
# : Visualization of multidimensional categorical datasets <br>
#  <br>
# About 555 Data Scientist Jobs, Visualize it. <br>
# The higher the height of the category, the more data is generated. <br>
# It indicates that the frequency increases. <br>

# In[ ]:


ds_pc=(df21_Ea_DS.loc[:, ['Q5','Q25','Q6','Q4','Q8']]
                 .replace({'I have never written code': '< 1 years',  '1-3 years': '1-2 years'})
                 .replace(['10-20 years', '20+ years'], '10+ years' )
                 .replace([0,'<999'])
                 )
fig = px.parallel_categories(ds_pc, labels={'Q5':'Job', 'Q25':'Salary', 'Q6':'Experience', 'Q4':'Degree', 'Q8':'Language'})

fig.update_layout(hovermode = 'x')
fig.update_layout(title='<b>        Data Scientist</b>',title_font_size=20,
                  margin = dict(t=120, l=100, r=10, b=150),
                  height=600, width=700)
fig.add_annotation(dict(font=dict(size=14),
                                    x=0.8,
                                    y=-0.2,
                                    showarrow=False,
                                    text="@green_yhjw",
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
fig.show()


# # 4. Ref.
# ---

# <h2> Ref. </h2>
# 
# - 동아시아 지역 https://ko.wikipedia.org/wiki/%EB%8F%99%EC%95%84%EC%8B%9C%EC%95%84
# 
# - 동아시아 인구 https://ko.wikipedia.org/wiki/%EC%95%84%EC%8B%9C%EC%95%84%EC%9D%98_%EC%9D%B8%EA%B5%AC
# 
# - 세계 인구 https://ko.wikipedia.org/wiki/%EC%84%B8%EA%B3%84_%EC%9D%B8%EA%B5%AC <br> https://ko.wikipedia.org/wiki/%EC%9D%B8%EA%B0%84_%EA%B0%9C%EB%B0%9C_%EC%A7%80%EC%88%98#2020%EB%85%84
# 
# - 동아시아 인간개발지수  https://namu.wiki/w/%EB%8F%99%EC%95%84%EC%8B%9C%EC%95%84
# 
# -  Data Scientist란  https://dataprofessional.tistory.com/126 <br> https://terms.naver.com/entry.naver?docId=1691563&cid=42171&categoryId=42183
# 
# - Kaggle이란 https://ko.wikipedia.org/wiki/%EC%BA%90%EA%B8%80
# - Python이란 https://ko.wikipedia.org/wiki/%ED%8C%8C%EC%9D%B4%EC%8D%AC
# 
# - Kaggle competition Ref. https://www.kaggle.com/miguelfzzz/the-typical-kaggle-data-scientist-in-2021 <br> https://www.kaggle.com/desalegngeb/how-popular-is-kaggle-in-africa
# 
# 
# - flaricon: <div>Icons made by <a href="https://www.freepik.com" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>
# 
# 
# 

# # 5. close
# ---

# 안녕하세요 한국에 사는 YH입니다. <br>
# python을 배운지 한달이 채 안되서 명이 한 팀이 되어  이번 대회에 참가 하게 되었습니다. <br>
# 많이 부족하지만 여기까지 읽어 주셔서 감사합니다. <br>
# 아직은 너무너무 부족한 제출물 이지만, 앞으로 열심히 해서 케글 대회에서 1등하는 그 날까지 지켜봐 주세요 ^^! <br>
#  혹시 코멘트로 다 전하지 못하셨던 말이 있으시다면, 저의 [github blog](https://yoonhwa-p.github.io/)에 방문하여 도움을 주세요!  <br>
# 별거 없지만 놀러오세요  ;-)
# 
# Hello, I'm YH and I live in Korea.<br>
# Less than a month after learning python, people became a team and participated in this competition.  <br>
# It's not enough, but thank you for reading it up to here.  <br>
# It's still not enough, but please watch until the day we win first place at the Kaggle competition ^^! <br>
#  <br>
# If there's anything you haven't said in the comments, please visit my [github blog](https://yoonhwa-p.github.io/) and help me! <br>
#  It's nothing special, but come and play. ;-) <br>
#  
#  <br>
#  <br>
#  
# 안녕하세요 저는 YH님과 같이 Kaggle 대회를 준비 한JW 입니다.  <br>
# python을 제대로 배우지도 못한채로 나오게 된 대회라 코드 부분에서 미숙한 점도 많고  <br>
# 오류도 많습니다!   <br>
# 하지만 대회를 출전하면서, python에 대해서 많은 공부도 되었고, 재미도 있어서 좋은 기회가 되었던것 같습니다.  <br>
#    <br>
# 아래는 저의 깃허브 주소 입니다  <br>
# 데이터 관련 분야에서 일하시는 분들은 저에게 팔로우를 걸어주세요!  <br>
# [github](https://github.com/wldnjd2)   <br>
#  
#  Hello, I'm JW who prepared for the Kaggle competition with YH.  <br>
# It's a competition where I didn't learn python properly, so I'm not good at codes.  <br>
# There are a lot of errors, too!  <br>
# However, as I participated in the competition, I studied a lot about Python and it was a good opportunity because it was fun.  <br>
#   <br>
# Below is my Git Hub address.  <br>
# For those who work in data-related fields, please follow me!  <br>
# [github](https://github.com/wldnjd2)   <br>
#  
