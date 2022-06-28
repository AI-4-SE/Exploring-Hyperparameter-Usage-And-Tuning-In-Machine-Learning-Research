#!/usr/bin/env python
# coding: utf-8

# ![spotify logo](https://www.scdn.co/i/_global/open-graph-default.png)
# 
# # **✨️ 🎵️ Spotify Top Hits | 📊️ Exploratory Data Analysis ✨️**
# 
# ## Dataset: 
# 
# This dataset contains audio statistics of the top 2000 tracks on Spotify from 2000-2019. The data contains about 18 columns each describing the track and it's qualities.
# 
# #### Dataset Link: [Top Hits Spotify from 2000-2019 by user @paradisejoy](https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019)
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot, plot

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


path = '/kaggle/input/top-hits-spotify-from-20002019/songs_normalize.csv'

ds = pd.read_csv(path)


# first few examples of the data

# In[ ]:


ds.head()


# we observe 2000 songs each with 18 features including the name

# In[ ]:


ds.shape, ds.columns


# we don't have any null values

# In[ ]:


ds.info()


# In[ ]:


ds.describe().T


# ## Let's look into what each feature means:
# 
# - **artist:** Name of the Artist.
# - **song:** Name of the Track.
# - **duration_ms:** Duration of the track in milliseconds.
# - **explicit:** The lyrics or content of a song or a music video contain one or more of the criteria which could be considered offensive or unsuitable for children.
# - **year:** Release Year of the track.
# - **popularity:** The higher the value the more popular the song is.
# - **danceability:** Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
# - **energy:** Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.
# - **key:** The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
# - **loudness:** The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.
# - **mode:** Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
# - **speechiness:** Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
# - **acousticness:** A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
# - **instrumentalness:** Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
# - **liveness:** Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
# - **valence:** A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
# - **tempo:** The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
# - **genre:** Genre of the track.
# 

# ### How many songs we have from each year

# In[ ]:


songs_per_years = ds['year'].value_counts().sort_index()

iplot(px.bar(songs_per_years, 
       title='Number of songs per year', 
       text_auto='.2s',
       labels=dict(index='year',value='number of songs'),
       color_discrete_sequence=['#1DB954']
      ).update_xaxes(type='category'))


# ### **All the artists in this list:**
# 
# We've 2000 songs, let's see how many artists are there in total

# In[ ]:


artists = ds['artist'].value_counts()
artists


# ### **We've a total of 835 artists. Let's explore the top.**
# 

# In[ ]:


iplot(px.pie(values=[artists[:50].sum(),2000-artists[:50].sum()], 
       names=['top 50 artists',f'remaining {835-50} artists'], 
       title="How many songs do the top 50 artists have",
       color_discrete_sequence = ['mediumpurple', 'moccasin']
      ).update_traces(textinfo='value+percent'))


# ### **No. of artists with only 1 top hit song VS artists with more than 1**

# In[ ]:


iplot(px.pie(names=['1 song', '>1 songs'], 
       values=[len(artists.loc[lambda x:x==1]), 
          len(artists)-len(artists.loc[lambda x:x==1])
         ],
       title="Artists with 1 top hit VS Artists with >1 top hit",
       color_discrete_sequence=['salmon','lightpink']
      ).update_traces(textinfo='label+percent'))


# ### **Top 10 Artists vs the average popularity of their songs**

# In[ ]:


artist_df = ds[['artist', 'popularity']].groupby('artist').mean().sort_values(by='artist')
artists = artists.sort_index()
artist_df['total songs'] = artists.values
artist_df.sort_values(by='total songs',ascending=False, inplace=True)
artist_df.reset_index(inplace=True)
artist_df[:10]

iplot(px.scatter(artist_df[:10], 
           x='artist', 
           y='popularity', 
           size='total songs',
           size_max=40,
           color='popularity',
           title='Top 10 artists vs average popularity of their top hits',
           hover_name='total songs'
          ))


# ### **How has the average duration of songs changed through the years**
# in this, we'll skip the years 1998, 1999 and 2020 since they have few examples

# In[ ]:


def ms_to_minsec(ms):
    sec = ms/1000
    return f"{int(sec//60)}:{int(sec%60)}"

durations = ds[['duration_ms','year']].groupby('year').mean().reset_index().iloc[2:22]
durations['duration_s'] = durations['duration_ms'] / 1000
durations['min:sec'] = durations['duration_ms'].apply(ms_to_minsec)


iplot(px.line(durations, 
        x='year', 
        y='duration_s',
        title='average song duration over the years',
        text='min:sec'
       ).update_xaxes(type='category').update_traces(textposition='top right'))


# In[ ]:


asdy = ds[['artist','song','duration_ms','year']]
print(asdy[asdy.duration_ms == asdy.duration_ms.max()])
print(asdy[asdy.duration_ms == asdy.duration_ms.min()])


# #### **stats regarding song duration**
# 
# The longest song is ***Mirrors*** by **Justin Timberlake**: 8:04s
# 
# The shortest song is ***Old Town Road*** by **Lil Nas X**: 1:53s

# ### **Explicit vs Clean over the years**

# In[ ]:


"""
ds['explicit'].value_counts().reset_index()

   index  explicit
0  False      1449
1   True       551
"""

iplot(px.pie(ds['explicit'].value_counts().reset_index(), 
       values='explicit', 
       names=['Clean', 'Explicit'],
       title='Explicit or not?',
       color_discrete_sequence = ['cornflowerblue', 'crimson']
      ).update_traces(textinfo='label+percent'))


# In[ ]:


year_explicit = ds.groupby(['year','explicit']).size().unstack(fill_value=0).reset_index()
year_explicit.rename(columns={False:'Clean', True: 'Explicit'}, inplace=True)

iplot(px.bar(year_explicit, 
       y=['Clean', 'Explicit'], 
       x='year',
       title='Explicit vs Clean distribution each year',
       color_discrete_sequence=['cornflowerblue', 'crimson']
      ).update_xaxes(type='category'))


# ### **Key and Mode**

# In[ ]:


iplot(px.pie(ds['key'].value_counts().reset_index(), 
       names=r'C C♯/D♭ D E♭/D♯ E F F♯/G♭ G A♭/G♯ A B♭/A♯ B'.split(), 
       values='key',
       color_discrete_sequence = px.colors.qualitative.Set3,
       title='Key Distribution'
      ).update_traces(textinfo='label+percent'))


# In[ ]:


"""
    index   mode
0       1   1107
1       0   893
"""
iplot(px.pie(ds['mode'].value_counts().reset_index(), 
       names=['major','minor'], 
       values='mode',
       color_discrete_sequence = px.colors.qualitative.Pastel2,
       title='Major/Minor Distribution'
      ).update_traces(textinfo='label+percent'))


# In[ ]:


key_mode = ds.groupby(['key','mode']).size().unstack(fill_value=0).reset_index()
key_names = r'C C♯/D♭ D E♭/D♯ E F F♯/G♭ G A♭/G♯ A B♭/A♯ B'.split()
key_mode.rename(columns={0:'minor',1:'major'}, inplace=True)
key_mode['key name'] = key_names

iplot(px.bar(key_mode, 
       x='key name', 
       y=['major', 'minor'],
       color_discrete_sequence=px.colors.qualitative.Pastel2,
       title='Major vs Minor per key'
      ))


# ## **Histograms**

# In[ ]:


from plotly.subplots import make_subplots

histogram_labels = ['popularity',
                    'danceability', 
                    'energy', 
                    'speechiness', 
                    'loudness', 
                    'acousticness', 
                    'liveness', 
                    'instrumentalness',
                    'valence',
                    'tempo'
                   ]

colors = px.colors.qualitative.Vivid
for i in range(len(histogram_labels)):
    fig = px.histogram(ds, 
                       histogram_labels[i], 
                       title=f'{histogram_labels[i]} distribution in top hits', 
                       height=400, 
                       width=500,
                       color_discrete_sequence=[colors[i]]
                      )
    iplot(fig)


# ## **Genres**

# In[ ]:


def split_genres(genre):
    g = genre.replace(' ','').split(',')
    g = [t for t in g if t!='set()']
    return g

def flatten(t):
    return [item for sublist in t for item in sublist]

def remove_duplicates(l):
    res = []
    l = [res.append(x) for x in l if x not in res]
    return res

all_genres = remove_duplicates(flatten(list(ds['genre'].apply(split_genres))))

genre_count = {genre: 0 for genre in all_genres}

for song in list(ds['genre']):
    for genre in split_genres(song):
        genre_count[genre]+=1


# In[ ]:


iplot(px.pie(names=genre_count.keys(), 
       values=[genre_count[key] for key in genre_count.keys()],
       title="Genre Distribution"
      ).update_traces(textinfo='label+percent'))


# ## Thank you for going through my notebook! ✨️
# 
# Please leave your suggestions in the comments! Appreciate it :)
