#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

import plotly
plotly.offline.init_notebook_mode(connected=True) 

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().system('pip install git+https://github.com/LIAAD/yake -q')
get_ipython().system('pip install wordcloud -q')


# In[ ]:


import yake 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[ ]:


df = pd.read_csv('../input/hindimediatweets/JagranNews.csv',delimiter=',', encoding='utf-8')
df.head()


# In[ ]:


#Code by KOUSTUBHK https://www.kaggle.com/code/kkhandekar/bhagavad-gita-keyword-extraction-wordcloud

# Custom function to extract keyword using YAKE
def extract_keyword(txt):
    max_ngram_size = 1  # single word
    deduplication_thresold = 0.1   #avoid the repetition of words in keywords
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = 1  # top word with lowest score
    
    custom_kw_extractor = yake.KeywordExtractor(n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(txt)
    return keywords[0][0]


# In[ ]:


#Code by KOUSTUBHK https://www.kaggle.com/code/kkhandekar/bhagavad-gita-keyword-extraction-wordcloud

# Applying the function
df['Keywrd_Hin'] = df['text'].apply(lambda x: extract_keyword(x))


# In[ ]:


#Code by KOUSTUBHK https://www.kaggle.com/code/kkhandekar/bhagavad-gita-keyword-extraction-wordcloud

# Generate Word Cloud
cloud_txt = df['Keywrd_Hin'].tolist() 
cloud_txt = ' '.join(cloud_txt)
font_path = '../input/hindi-font-stopwords/gargi.ttf'
stopword_path = '../input/hindi-font-stopwords/stopwords_hin.txt'

wordcloud = WordCloud(font_path=font_path
                      ,width = 1000
                      ,height = 800
                      ,background_color ='black'
                      ,colormap='Set3'
                      ,stopwords = stopword_path
                      ,min_font_size = 8
                      ,collocations=False).generate(cloud_txt)

#plot the wordcloud object
plt.figure(figsize = (15, 15), facecolor = None)
plt.imshow(wordcloud, interpolation='bilInear')
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# #Unfortunately, it didn't work with YAKE. Let's go to plan B.

# In[ ]:


#https://colab.research.google.com/github/rahul1990gupta/indic-nlp-datasets/blob/master/examples/Getting_started_with_processing_hindi_text.ipynb#scrollTo=ghnpMGzngcOn

# That doesn't look right. We need to provide a custom font file to render it correctly. 
# the issue is highlighted here: https://github.com/amueller/word_cloud/issues/70
import requests
url = "https://hindityping.info/download/assets/Hindi-Fonts-Unicode/gargi.ttf"

r = requests.get(url, allow_redirects=True)
font_path="gargi.ttf"

with open(font_path, "wb") as fw:
  fw.write(r.content)


# In[ ]:


from wordcloud import WordCloud
from spacy.lang.hi import STOP_WORDS as STOP_WORS_HI

from spacy.lang.hi import STOP_WORDS as STOP_WORDS_HI


# In[ ]:


#https://colab.research.google.com/github/rahul1990gupta/indic-nlp-datasets/blob/master/examples/Getting_started_with_processing_hindi_text.ipynb#scrollTo=ghnpMGzngcOn

wordcloud = WordCloud(
    width=400,
    height=300,
    max_font_size=50, 
    max_words=1000,
    background_color="black",
    colormap='Set3',
    stopwords=STOP_WORDS_HI,
    font_path=font_path
).generate(str(df["text"]))
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

