#!/usr/bin/env python
# coding: utf-8

# # Assignment 1 - Python Basics Practice
# 
# *This assignment is a part of the course ["Data Analysis with Python: Zero to Pandas"](https://jovian.ml/learn/data-analysis-with-python-zero-to-pandas)*
# 
# In this assignment, you'll get to practice some of the concepts and skills covered in the following notebooks:
# 
# 1. [First Steps with Python and Jupyter](https://jovian.ml/aakashns/first-steps-with-python)
# 2. [A Quick Tour of Variables and Data Types](https://jovian.ml/aakashns/python-variables-and-data-types)
# 3. [Branching using Conditional Statements and Loops](https://jovian.ml/aakashns/python-branching-and-loops)
# 
# 
# As you go through this notebook, you will find the symbol **???** in certain places. To complete this assignment, you must replace all the **???** with appropriate values, expressions or statements to ensure that the notebook runs properly end-to-end. 
# 
# **Guidelines**
# 
# 1. Make sure to run all the code cells, otherwise you may get errors like `NameError` for undefined variables.
# 2. Do not change variable names, delete cells or disturb other existing code. It may cause problems during evaluation.
# 3. In some cases, you may need to add some code cells or new statements before or after the line of code containing the **???**. 
# 4. Since you'll be using a free online service for code execution, save your work by running `jovian.commit` at regular intervals.
# 5. Questions marked **(Optional)** will not be considered for evaluation, and can be skipped. They are for your learning.
# 6. If you are stuck, you can ask for help on the course forum. Post errors, ask for hints and help others, but **please don't share the full solution answer code** to give others a chance to write the code themselves.
# 7. After submission your code will be tested with some hidden test cases. Make sure to test your code exhaustively to cover all edge cases.
# 
# 
# Important Links:
# 
# * Make submissions here: https://jovian.ml/learn/data-analysis-with-python-zero-to-pandas/assignment/assignment-1-python-basics-practice
# 
# * Get help on the community forum: https://jovian.ml/forum/t/assignment-1-python-practice/7761 . You can get help with errors or ask for hints, but **please don't ask for or share the full working answer code** on the forum.
# 

# ### How to Run the Code and Save Your Work
# 
# **Option 1: Running using free online resources (1-click, recommended)**: Click the **Run** button at the top of this page and select **Run on Binder**. You can also select "Run on Colab" or "Run on Kaggle", but you'll need to create an account on [Google Colab](https://colab.research.google.com) or [Kaggle](https://kaggle.com) to use these platforms.
# 
# 
# **Option 2: Running on your computer locally**: To run the code on your computer locally, you'll need to set up [Python](https://www.python.org) & [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/), download the notebook and install the required libraries. Click the **Run** button at the top of this page, select the **Run Locally** option, and follow the instructions.
# 
# **Saving your work**: You can save a snapshot of the assignment to your [Jovian](https://jovian.ai) profile, so that you can access it later and continue your work. Keep saving your work by running `jovian.commit` from time to time.

# In[ ]:


# Install the library
get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


# Import it
import jovian


# In[ ]:


project_name='python-practice-assignment'


# In[ ]:


# Capture and upload a snapshot
jovian.commit(project=project_name, privacy='secret', evironment=None)


# You'll be asked to provide an API Key, to securely upload the notebook to your Jovian.ml account. You can get the API key from your Jovian.ml profile page after logging in / signing up. See the docs for details: https://jovian.ml/docs/user-guide/upload.html . The privacy of your assignment notebook is set to *Secret*, so that you can the evlauators can access it, but it will not shown on your public profile to other users.

# ## Problem 1 - Variables and Data Types
# 
# **Q1: Assign your name to the variable `name`.**

# In[ ]:


name = 'Deepak'


# **Q2: Assign your age (real or fake) to the variable `age`.**

# In[ ]:


age = 22


# **Q3: Assign a boolean value to the variable `has_android_phone`.**

# In[ ]:


has_android_phone = True


# You can check the values of these variables by running the next cell.

# In[ ]:


name, age, has_android_phone


# **Q4: Create a dictionary `person` with keys `"Name"`, `"Age"`, `"HasAndroidPhone"` and values using the variables defined above.**

# In[ ]:


person = {'Name':'Deepak','Age':22,'HasAndroidPhone':True}


# Let's use the `person` dictionary to print a nice message.

# In[ ]:


print("{} is aged {}, and owns an {}.".format(
    person["Name"], 
    person["Age"], 
    "Android phone" if person["HasAndroidPhone"] else "iPhone"
))


# **Q4b (Optional): Use a `for` loop to display the `type` of each value stored against each key in `person`.**
# 
# Here's the expected output for the key `"Name"`: 
# 
# ```
# The key "Name" has the value "Derek" of the type "<class 'str'>"
# ```

# In[ ]:


# this is optional
for key, value in person.items():
    print("The key {} has the value {} of the type {}".format(key, value, type(value)))


# Now that you've solved one problem, it would be a good idea to record a snapshot of your notebook.

# In[ ]:


jovian.commit(project=project_name,environment=None)


# ## Problem 2 - Working with Lists
# 
# **Q5: Create a list containing the following 3 elements:**
# 
# * your favorite color
# * the number of pets you have
# * a boolean value describing whether you have previous programming experience
# 

# In[ ]:


my_list = ['Red',0,True]


# Let's see what the list looks like:

# In[ ]:


my_list


# **Q6: Complete the following `print` and `if` statements by accessing the appropriate elements from `my_list`.**
# 
# *Hint*: Use the list indexing notation `[]`.

# In[ ]:


print('My favorite color is', my_list[0])


# In[ ]:


print('I have {} pet(s).'.format(my_list[1]))


# In[ ]:


if my_list[2]:
    print("I have previous programming experience")
else:
    print("I do not have previous programming experience")


# **Q7: Add your favorite single digit number to the end of the list using the appropriate list method.**

# In[ ]:


my_list.append(1)


# Let's see if the number shows up in the list.

# In[ ]:


my_list


# **Q8: Remove the first element of the list, using the appropriate list method.**
# 
# *Hint*: Check out methods of list here: https://www.w3schools.com/python/python_ref_list.asp

# In[ ]:


my_list.pop(0)


# In[ ]:


my_list


# **Q9: Complete the `print` statement below to display the number of elements in `my_list`.**

# In[ ]:


print("The list has {} elements.".format(len(my_list)))


# Well done, you're making good progress! Save your work before continuing

# In[ ]:


jovian.commit(project=project_name,environment=None)


# ## Problem 3 - Conditions and loops
# 
# **Q10: Calculate and display the sum of all the numbers divisible by 7 between 18 and 534 i.e. `21+28+35+...+525+532`**.
# 
# *Hint*: One way to do this is to loop over a `range` using `for` and use an `if` statement inside it.

# In[ ]:


# store the final answer in this variable
sum_of_numbers = 0

# perform the calculation here
for i in range (18,535):
    if i%7==0:
        sum_of_numbers += i


# In[ ]:


print('The sum of all the numbers divisible by 7 between 18 and 534 is', sum_of_numbers)


# If you are not able to figure out the solution to this problem, you can ask for hints on the community forum: https://jovian.ml/forum/t/assignment-1-python-practice/7761 . Remember to save your work before moving forward.

# In[ ]:


jovian.commit(project=project_name,environment=None)


# ## Problem 4 - Flying to the Bahamas
# 
# **Q11: A travel company wants to fly a plane to the Bahamas. Flying the plane costs 5000 dollars. So far, 29 people have signed up for the trip. If the company charges 200 dollars per ticket, what is the profit made by the company?**
# 
# Fill in values or arithmetic expressions for the variables below.

# In[ ]:


cost_of_flying_plane = 5000


# In[ ]:


number_of_passengers = 29


# In[ ]:


price_of_ticket = 200


# In[ ]:


profit = price_of_ticket*number_of_passengers - cost_of_flying_plane


# In[ ]:


print('The company makes of a profit of {} dollars'.format(profit))


# **Q11b (Optional): Out of the 29 people who took the flight, only 12 buy tickets to return from the Bahamas on the same plane. If the flying the plane back also costs 5000 dollars, and does the company make an overall profit or loss? The company charges the same fee of 200 dollars per ticket for the return flight.**
# 
# Use an `if` statement to display the result.

# In[ ]:


# this is optional
return_passengers = 12
overall_profit = price_of_ticket*number_of_passengers + price_of_ticket*return_passengers - cost_of_flying_plane*2


# In[ ]:


# this is optional
if overall_profit>0:
    print("The company makes an overall profit of {} dollars".format(overall_profit))
else:
    print("The company makes an overall loss of {} dollars".format(-overall_profit))


# Great work so far! Want to take a break? Remember to save and upload your notebook to record your progress.

# In[ ]:


jovian.commit(project=project_name,environment=None)


# ## Problem 5 - Twitter Sentiment Analysis
# 
# Are your ready to perform some *Data Analysis with Python*? In this problem, we'll analyze some fictional tweets and find out whether the overall sentiment of Twitter users is happy or sad. This is a simplified version of an important real world problem called *sentiment analysis*.
# 
# Before we begin, we need a list of tweets to analyze. We're picking a small number of tweets here, but the exact same analysis can also be done for thousands, or even millions of tweets. The collection of data that we perform analysis on is often called a *dataset*.

# In[ ]:


tweets = [
    "Wow, what a great day today!! #sunshine",
    "I feel sad about the things going on around us. #covid19",
    "I'm really excited to learn Python with @JovianML #zerotopandas",
    "This is a really nice song. #linkinpark",
    "The python programming language is useful for data science",
    "Why do bad things happen to me?",
    "Apple announces the release of the new iPhone 12. Fans are excited.",
    "Spent my day with family!! #happy",
    "Check out my blog post on common string operations in Python. #zerotopandas",
    "Freecodecamp has great coding tutorials. #skillup"
]


# Let's begin by answering a very simple but important question about our dataset.
# 
# **Q12: How many tweets does the dataset contain?**

# In[ ]:


number_of_tweets = len(tweets)


# Let's create two lists of words: `happy_words` and `sad_words`. We will use these to check if a tweet is happy or sad.

# In[ ]:


happy_words = ['great', 'excited', 'happy', 'nice', 'wonderful', 'amazing', 'good', 'best']


# In[ ]:


sad_words = ['sad', 'bad', 'tragic', 'unhappy', 'worst']


# To identify whether a tweet is happy, we can simply check if contains any of the words from `happy_words`. Here's an example:

# In[ ]:


sample_tweet = tweets[0]


# In[ ]:


sample_tweet


# In[ ]:


is_tweet_happy = False

# Get a word from happy_words
for word in happy_words:
    # Check if the tweet contains the word
    if word in sample_tweet:
        # Word found! Mark the tweet as happy
        is_tweet_happy = True


# Do you understand what we're doing above? 
# 
# > For each word in the list of happy words, we check if is a part of the selected tweet. If the word is indded a part of the tweet, we set the variable `is_tweet_happy` to `True`. 

# In[ ]:


is_tweet_happy


# **Q13: Determine the number of tweets in the dataset that can be classified as happy.**
# 
# *Hint*: You'll need to use a loop inside another loop to do this. Use the code from the example shown above.

# In[ ]:


# store the final answer in this variable
number_of_happy_tweets = 0

# perform the calculations here
for word in happy_words:
    for tweet in tweets:
        if word in tweet:
            is_tweet_happy = True
            number_of_happy_tweets += 1


# In[ ]:


print("Number of happy tweets:", number_of_happy_tweets)


# If you are not able to figure out the solution to this problem, you can ask for hints on the community forum: https://jovian.ml/forum/t/assignment-1-python-practice/7761 . Also try adding `print` statements inside your loops to inspect variables and make sure your logic is correct.

# **Q14: What fraction of the total number of tweets are happy?**
# 
# For example, if 2 out of 10 tweets are happy, then the answer is `2/10` i.e. `0.2`.

# In[ ]:


happy_fraction = number_of_happy_tweets/number_of_tweets


# In[ ]:


print("The fraction of happy tweets is:", happy_fraction)


# To identify whether a tweet is sad, we can simply check if contains any of the words from `sad_words`.
# 
# **Q15: Determine the number of tweets in the dataset that can be classified as sad.**

# In[ ]:


# store the final answer in this variable
number_of_sad_tweets = 0

# perform the calculations here
for tweet in tweets:
    for word in tweet.split(' '):
        # Check whether the word is found in happy_words
        if word in sad_words:
            number_of_sad_tweets +=1


# In[ ]:


print("Number of sad tweets:", number_of_sad_tweets)


# **Q16: What fraction of the total number of tweets are sad?**

# In[ ]:


sad_fraction = number_of_sad_tweets/number_of_tweets


# In[ ]:


print("The fraction of sad tweets is:", sad_fraction)


# The rest of this problem is optional. Let's save your work before continuing.

# In[ ]:


jovian.commit(project=project_name,environment=None)


# Great work, even with some basic analysis, we already know a lot about the sentiment of the tweets given to us. Let us now define a metric called "sentiment score", to summarize the overall sentiment of the tweets.
# 
# **Q16b (Optional): Calculate the sentiment score, which is defined as the difference betweek the fraction of happy tweets and the fraction of sad tweets.**

# In[ ]:


sentiment_score = happy_fraction - sad_fraction


# In[ ]:


print("The sentiment score for the given tweets is", sentiment_score)


# In a real world scenario, we could calculate & record the sentiment score for all the tweets sent out every day. This information can be used to plot a graph and study the trends in the changing sentiment of the world. The following graph was creating using the Python data visualization library `matplotlib`, which we'll cover later in the course.
# 
# <img src="https://i.imgur.com/6CCIwCb.png" style="width:400px">
# 
# What does the sentiment score represent? Based on the value of the sentiment score, can you identify if the overall sentiment of the dataset is happy or sad?
# 
# **Q16c (Optional): Display whether the overall sentiment of the given dataset of tweets is happy or sad, using the sentiment score.**

# In[ ]:


if sentiment_score>0:
    print("The overall sentiment is happy")
else:
    print("The overall sentiment is sad")


# Finally, it's also important to track how many tweets are neutral i.e. neither happy nor sad. If a large fraction of tweets are marked neutral, maybe we need to improve our lists of happy and sad words. 
# 
# **Q16d (Optional): What is the fraction of tweets that are neutral i.e. neither happy nor sad.**

# In[ ]:


# store the final answer in this variable
number_of_neutral_tweets = 0

# perform the calculation here
number_of_neutral_tweets = number_of_tweets - number_of_happy_tweets - number_of_sad_tweets


# In[ ]:


neutral_fraction = number_of_neutral_tweets/number_of_tweets


# In[ ]:


print('The fraction of neutral tweets is', neutral_fraction)


# Ponder upon these questions and try some experiments to hone your skills further:
# 
# * What are the limitations of our approach? When will it go wrong or give incorrect results?
# * How can we improve our approach to address the limitations?
# * What are some other questions you would like to ask, given a list of tweets?
# * Try collecting some real tweets from your Twitter timeline and repeat this analysis. Do the results make sense?
# 
# **IMPORTANT NOTE**: If you want to try out these experiments, please create a new notebook using the "New Notebook" button on your Jovian.ml profile, to avoid making unintended changes to your assignment submission notebook.

# ## Submission 
# 
# Congratulations on making it this far! You've reached the end of this assignment, and you just completed your first data analysis problem. It's time to record one final version of your notebook for submission.
# 
# Make a submission here by filling the submission form: https://jovian.ml/learn/data-analysis-with-python-zero-to-pandas/assignment/assignment-1-python-basics-practice

# In[ ]:


jovian.commit(project=project_name,environment=None)


# In[ ]:




