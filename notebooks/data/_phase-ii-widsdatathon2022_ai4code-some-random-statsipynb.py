#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import ujson
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

random.seed(98)


# In[ ]:


train_folder = '../input/AI4Code/train/'
test_folder = '../input/AI4Code/test/'

train_data = pd.read_csv('../input/AI4Code/train_orders.csv')
test_data = pd.read_csv('../input/AI4Code/train_ancestors.csv')


# In[ ]:


train_data.head()


# In[ ]:


def get_cell_order_and_json_data(file_id,display=True):
    if isinstance(file_id,int):
        file_id = train_data['id'].iloc[file_id]
        
    cell_order = train_data[train_data['id']==file_id]['cell_order'].iloc[0].split(' ')
    file = open(train_folder+file_id+'.json')
    json_data = ujson.load(file)
    
    source = json_data['source']
    
    # for displaying it in order
    if display:
        for cell_id in cell_order:
            print(source[cell_id])
    
    return cell_order,json_data


# In[ ]:


get_cell_order_and_json_data(2);


# ### Question has to be asked<br>
# Do most notebook have please upvote line at the beginning or at the top ?<br>
# I am asking this question because this line has nothing to do with the content of the code cell.<br>
# I will consider any sentence that contains the word "upvote" as a cell asking for upvote.<br>
# If upvote line is in top 20% of the cell it is top else if it is int bottom 20% it is in bottom else middle.<br>
# 
# While we are at it we could also get some other stats using same loop<br>
# What is the gap between markdown cells ?<br>
# Distribution of len of markdown cells ?<br>
# What is the probability that markdown is at x position ?<br>
# 
# And one hypothesis I want to test is that is longer markdown are the starting of the notebook.<br>
# I am considering markdown with more than 50 words as long<br>
# 
# I will run this on random 10000 samples 

# In[ ]:


position_of_upvote = [0,0,0] # beginning, end and middle
len_of_markdowns = list()
markdown_gap = dict()
markdown_position_counter = {}
markdowns_in_notebook = list()
longer_markdown_position = list()
short_markdown_position = list()

random_list = random.sample(range(train_data.shape[0]),10_000) # random 10k samples

for i in tqdm(random_list):
    cell_order, json_data = get_cell_order_and_json_data(i,display=False)
    cell_type = json_data['cell_type']
    source = json_data['source']
    total_cell_orders = len(cell_order)
    markdown_gap_counter = 0
    markdown_counter = 0
    
    for j,cell_id in enumerate(cell_order):
        if cell_type[cell_id] == 'markdown':
            markdown_gap_counter += 1
            markdown_counter += 1
            markdown_split = source[cell_id].lower().split()
            
            position =j/total_cell_orders
            markdown_len = len(markdown_split)
            len_of_markdowns.append(markdown_len)
            if markdown_len >= 50:
                longer_markdown_position.append(position)
            else:
                short_markdown_position.append(position)
            
            #oth position is for counting all makrkdown cell in position x
            if markdown_position_counter.get(j):
                markdown_position_counter[j][0] += 1
            else:
                markdown_position_counter[j] = [1,0]
            
            if 'upvote' in source[cell_id].lower():
                markdown_len = len(markdown_split)
                if position <= 0.2:                 #in beginning
                    position_of_upvote[0] +=1
                elif position >= 0.8:               #in the end
                    position_of_upvote[1] += 1
                else:
                    position_of_upvote[2] += 1          #in the middle
        else:
            if markdown_gap.get(markdown_gap_counter):
                markdown_gap[markdown_gap_counter] += 1
            else:
                markdown_gap[markdown_gap_counter] = 1
            markdown_gap_counter = 0
            
        #1st position is for counting all cells in the position
        if markdown_position_counter.get(j):
            markdown_position_counter[j][1] += 1
        else:
            markdown_position_counter[j] = [0,1]
            
    markdowns_in_notebook.append(markdown_counter)


# ## Count of "Upvote ask" at beginning end and middle of the notebook

# In[ ]:


plt.figure(figsize=(15,7))
sns.barplot(x=['beginning','end','middle'],
            y=position_of_upvote)
plt.title("Count of upvote at beginning, end and middle")
plt.xlabel("Position")
plt.ylabel("Count");


# Yeah so I was expecting more upvote ask at the end, there are some middle ones too.

# ## Counter of Gap Betwwen Markdown cells

# In[ ]:


plt.figure(figsize=(15,7))
sns.barplot(x=list(markdown_gap.keys()),
            y=list(markdown_gap.values()))
plt.title("Count of gap between markdown cells")
plt.xlabel("Gap")
plt.ylabel("Count");


# I was expecting 1 to be much higher but it seems like there are much more continous markdown cells.

# ## Distribution of length of markdown

# In[ ]:


plt.figure(figsize=(15,7))
len_of_markdowns = [x for x in len_of_markdowns if x < 500]
sns.histplot(len_of_markdowns,color='pink')
plt.title("Distribution of length of markdown");


# ## Distribution of Count of Markdowns in a Notebook

# In[ ]:


plt.figure(figsize=(15,7))
sns.histplot(markdowns_in_notebook,color='yellow')
plt.title("Distribution of count of markdown in a Notebook");


# ## Position of longer markdown

# In[ ]:


plt.figure(figsize=(15,7))
sns.histplot(longer_markdown_position,color='red')
plt.title("Longer markdown position in a Notebook");


# Longer markdown are much more closer to the starting as I expected.<br>
# (This could also mean that first markdown is default kaggle one, this needs to be checked)<br>
# But one more thing is markdown are equally distributed among the other half of the notebook<br>

# ## Short Markdown positions

# In[ ]:


plt.figure(figsize=(15,7))
sns.histplot(short_markdown_position,color='green')
plt.title("Short markdown position in a Notebook");


# Interestingly short markdowns also follow similar patern as the longer ones

# ## Probability of markdown in a position

# In[ ]:


plt.figure(figsize=(20,10))
probability_of_markdown = [x/y for x,y in list(markdown_position_counter.values())]
sns.barplot(x=list(markdown_position_counter.keys()),y=probability_of_markdown)
plt.title("Probability of markdown in a position")
plt.xticks(rotation=90)
plt.show()


# One interesting thing here is if you see in between the pattern you will see there is a dip in the markdown cells.<br>
# which is sort of expected as there should be more code in the middle of the notebook.<br>

# ## I think this competition is about two question 
# 1. Whether there should be a code or markdown in this position?
# 2. If a markdown then which markdown?
#  
# I think both the questions could be answered by measuring relationship between code and markdown(just saying obvious things).<br>
# If markdown is related to the code place it at the top of the code,or maybe below who knows I have seen notebooks explaining code after the code cell.<br>
# 
# Hard ones are non-code related markdown and markdown which explains concepts, maybe a good model might pick up on this as well.<br>

# ## obviously in the end, Please Upvote If you find this usefull 

# In[ ]:




