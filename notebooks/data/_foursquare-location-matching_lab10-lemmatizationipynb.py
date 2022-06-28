#!/usr/bin/env python
# coding: utf-8

# <h1><center>Lab No. 10 </center></h1>
# 
# **Objective(s)**: Performing white-space eviction, digit removal, case normalization, and lemmatization: Text preprocessing Techniques
# 
# **Tools**: Python language, IDLE, NLTK
# 
# **Theory**:
# - Text data is prone to noise. Text data usually contains irrelevant or inflected terms that increase feature vector space. Therefore, it is necessary to romove or manage them. Digists, upper-lower case, and inflected form of verbs and nouns are among these entities.

# **NLTK**: It is a python module for natural language processing.
# 
# **Digits**
# - Text processing requires text data. However, free text may also contain digits that needs to be removed in most of text mining applications. Such as sentiment analysis
# 
# **Case Nomalization**
# - Upper case and lower case characters have no semantic value. They just increase vector space. Such variations need to be rectified. For example, "Apple", "aPple", "APPLE", and so on. This simple 5-character word can have 32 variation. In case case normalization is not performed on a dirty dataset, there can be huge number of features which may be beyond the limitations of hardware resoruces.
# 
# **Lemmatization**
# - It is text preprocessing procedure used in Text Analytics. Poeple use different forms of a word, such as work, works, worked, working, and worker. Additionally, there are families of derivationally related words with similar meanings, such as efficiency and efficient. In all situations, it is useful to substitute all variants with a single representation in the dataset.
# 
# - The goal of lemmatization aim at reducing the feature space and convert the inflected terms into grammatically correct form.
# 
# **Lemmatization vs stemming**
# - The difference between stemming and lemmatization is, lemmatization considers the context and converts the word to its meaningful base form, whereas stemming just removes the last few characters, often leading to incorrect meanings and spelling errors.
# 
# - For example, lemmatization would correctly identify the base form of ‘caring’ to ‘care’, whereas, stemming would cutoff the ‘ing’ part and convert it to car. ‘Caring’ -> Lemmatization -> ‘Care’ ‘Caring’ -> Stemming -> ‘Car’
# 
# - Also, sometimes, the same word can have multiple different ‘lemma’s. So, based on the context it’s used, you should identify the ‘part-of-speech’ (POS) tag for the word in that specific context and extract the appropriate lemma.

# In[ ]:


import nltk
import pandas as pd


# In[ ]:


text = [" I   like Apples1", "do You like mangoes22?  ", \
        "  is Fruites available in        bahawalpur?", \
           "she likes berries 444.", \
        "But eating56 fruit now is coslty   .    "]


# In[ ]:


df = pd.DataFrame(text, columns=["text"])
df


# In[ ]:


df["text"] = df.text.str.strip()
df


# ### Digits removal

# In[ ]:


df["text"] = df.text.str.replace('\d+', '')
df


# ### Case Normalization - case conversion

# In[ ]:


df["text"] = df.text.str.lower()
df


# In[ ]:


df["text"] = df.text.str.upper()
df


# ### Performing Lemmatization

# In[ ]:


#demo
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()

print(lem.lemmatize("cats"))
print(lem.lemmatize("cacti"))
print(lem.lemmatize("geese"))
print(lem.lemmatize("rocks"))
print(lem.lemmatize("python"))
print(lem.lemmatize("better", pos="a"))
print(lem.lemmatize("best", pos="a"))
print(lem.lemmatize("run"))
print(lem.lemmatize("run",'v'))


# - By default, pos=noun

# In[ ]:


#demo
text = ["is", "am", "are", "was", "were", "shall", "will"]
for w in text:
    print(lem.lemmatize(w))


# In[ ]:


tokenizer = nltk.tokenize.WhitespaceTokenizer()

def lemmatize_text(text):
    return [lem.lemmatize(w) for w in tokenizer.tokenize(text)]

df['lemmatized'] = df.text.apply(lemmatize_text)
df


# <h1><center> Lab Task(s) </center></h1>
# 
# 1. Choose any three text preprocessing methods and apply it (Other than use in Lab09 or Lab10)
