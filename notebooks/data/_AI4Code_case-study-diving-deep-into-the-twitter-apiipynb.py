#!/usr/bin/env python
# coding: utf-8

# ## Twitter API and authentication
# 
# This project is undertaken to work on the twitter API using pre-supplied authentication keys. The aim of this project is to extract certain keywords from live twitter data and plot a chart to measure popularity of the keywords **"clinton", "trump", "sanders", and "cruz"**.

# ### Streaming tweets
# 
# It's time to stream some tweets! Task here is to create the *Stream* object and to filter tweets according to particular keywords using the *tweepy* package.

# * Create the *Stream* object with the given credentials.

# In[ ]:


get_ipython().system(' pip install tweepy')


# In[ ]:


# importing the relevant package
import tweepy, json


# In[ ]:


# Store credentials in relevant variables
consumer_key = "nZ6EA0FxZ293SxGNg8g8aP0HM"
consumer_secret = "fJGEodwe3KiKUnsYJC3VRndj7jevVvXbK2D5EiJ2nehafRgA6i"
access_token = "1092294848-aHN7DcRP9B4VMTQIhwqOYiB14YkW92fFO8k8EPy"
access_token_secret = "X4dHmhPfaksHcQ7SCbmZa2oYBBVSD2g8uIHXsp5CTaksx"

# Create Stream object with credentials
stream = tweepy.Stream(consumer_key, consumer_secret, access_token, access_token_secret)


# * Filter Stream variable for the keywords "clinton", "trump", "sanders", and "cruz".

# In[ ]:


# Filter Stream variable
#stream.filter(track=['clinton', 'trump', 'sanders', 'cruz'])

# This process was not run since data was found on the Kaggle database


# ### Load and explore Twitter data
# 
# Now, we'll read the Twitter data into a list: tweets_data.

# In[ ]:


# Import package
import json

# String of path to file: tweets_data_path
tweets_data_path = '../input/importingdatainpython/tweets3.txt'

# Initialize empty list to store tweets: tweets_data
tweets_data = []

# Open & close connection to file
with open(tweets_data_path, "r") as tweets_file:

# Read in tweets and store in list: tweets_data
    for line in tweets_file:
        tweet = json.loads(line)
        tweets_data.append(tweet)

# Print the keys of the first tweet dict
print(tweets_data[0].keys())


# ### Twitter data to DataFrame
# 
# Now we have the Twitter data in a list of dictionaries, tweets_data, where each dictionary corresponds to a single tweet. Next, we're going to extract the text and language of each tweet. The text in a tweet, t1, is stored as the value t1['text']; similarly, the language is stored in t1['lang']. Now, let's build a DataFrame in which each row is a tweet and the columns are 'text' and 'lang'.

# In[ ]:


# Import package
import pandas as pd

# Build DataFrame of tweet texts and languages. to do so, the first argument should be tweets_data,
# a list of dictionaries. The second argument to pd.DataFrame() is a list of the keys as columns.
df = pd.DataFrame(tweets_data, columns=['text', 'lang'])

# Print head of DataFrame
print(df.head(8))


# ### A little bit of Twitter text analysis
# 
# Now that we have our DataFrame of tweets set up, we're going to do a bit of text analysis to count how many tweets contain the words **'clinton', 'trump', 'sanders' and 'cruz'**.

# In[ ]:


# Let's define a function word_in_text(), which will tell us whether the first argument (a word)
# occurs within the 2nd argument (a tweet).

import re

def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)

    if match:
        return True
    return False


# We're going to iterate over the rows of the DataFrame and calculate how many tweets contain each of our keywords! The list of objects for each candidate has been initialized to 0.

# In[ ]:


# Initialize list to store tweet counts
[clinton, trump, sanders, cruz] = [0, 0, 0, 0]

# Iterate through df, counting the number of tweets in which
# each candidate is mentioned
for index, row in df.iterrows():
    clinton += word_in_text('clinton', row['text'])
    trump += word_in_text('trump', row['text'])
    sanders += word_in_text('sanders', row['text'])
    cruz += word_in_text('cruz', row['text'])
    
[clinton, trump, sanders, cruz]


# ### Plotting the Twitter data
# 
# Now that we have the number of tweets that each candidate was mentioned in, we can plot a bar chart of this data. We'll use the seaborn package to do so. This quick exercise will help use visualize our data to the audience in an easy to understand manner

# In[ ]:


# Import packages
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(color_codes=True, rc={'figure.figsize':(12,6)})

# Complete the arguments of sns.barplot:
# The first argument should be the list of labels to appear on the x-axis.
# The second argument should be a list of the variables you wish to plot,
# as produced in the previous exercise (i.e. a list containing clinton, trump, etc).

ax = sns.barplot(x=['clinton', 'trump', 'sanders', 'cruz'], y=[clinton, trump, sanders, cruz])
ax.set(ylabel="count")
plt.show()


# Here, we see that **trump** clearly is the most talked about keyword out of **'clinton', 'trump', 'sanders' and 'cruz'**.
# 
# For additional analysis we can play around with the keywords and, following the above method, further drill down and figure out whether tweets for our presidential candidates are in a positive light or negative light.

# 
# #### Disclaimer
# 
# This exercise is part of a guided datacamp exercise. While the markdown text is pre-filled by the exercise provider, the code is original and follows the exercise provider guidelines.
# 
