#!/usr/bin/env python
# coding: utf-8

# #### The data obtained in Japan contains not only English but also Japanese letters: Kanji, Hiragana and Katakana.

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv("../input/foursquare-location-matching/train.csv")
train_JP = train[train["country"] == "JP"].copy()
train_JP.head()


# #### There are many mixed notations such as "東京" vs. "Tokyo", which should confuse model-training. 
# #### *Unidecode* can convert them into alphabets, but it reflects Chinese pronunciation.
# Here is an example:

# In[ ]:


from unidecode import unidecode
train_JP_uni=train_JP.copy()
for column in ["name", "address", "city", "state"]:
    train_JP_uni[column]=train_JP[column].astype(str).apply(unidecode)
print(train_JP[["name", "address", "city", "state"]][:1])
print(train_JP_uni[["name", "address", "city", "state"]][:1])


# The state "東京都" reads "Toukyouto" or "Tokyo-to" in Japan, but unidecode results in "Dong Jing Du", which is Chinese pronunciation.
# 
# #### Another module *pykakasi* is designed to convert Japanese letters into Latin/Roman letters more precisely.
# https://github.com/miurahr/pykakasi

# In[ ]:


get_ipython().system('pip install pykakasi -U')


# In[ ]:


import pykakasi
kakasi = pykakasi.kakasi()
kakasi.setMode('H', 'a') # Convert Hiragana into alphabet
kakasi.setMode('K', 'a') # Convert Katakana into alphabet
kakasi.setMode('J', 'a') # Convert Kanji into alphabet
conversion = kakasi.getConverter()
def convert(row):
    for column in ["name","address", "city", "state"]:
        try:
            row[column] = conversion.do(row[column])
        except:
            pass
    return row
print(train_JP[["name", "address", "city", "state"]][:1])
print(train_JP[:1].apply(convert, axis = 1)[["name", "address", "city", "state"]])


# #### All texts follow Japanese pronunciation.

# In[ ]:


train[train["country"] == "JP"] = train[train["country"] == "JP"].apply(convert, axis = 1) 
train[train["country"] == "JP"][["name", "address", "city", "state"]]


# #### I suppose that the kakasi preprocess should be followed by the unidecode preprocess to convert Chinese (and other languages) texts.
# #### Hope it helps your improvement.
