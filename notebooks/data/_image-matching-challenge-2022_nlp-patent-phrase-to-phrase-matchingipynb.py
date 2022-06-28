#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # On Kaggle

# In[ ]:


import os
from pathlib import Path
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')


# # Import and EDA

# In[ ]:


if iskaggle:
    path = Path('../input/us-patent-phrase-to-phrase-matching')
    get_ipython().system(' pip install -q datasets')


# Documents in NLP datasets are generally in one of two main forms:
# 
# Larger documents: One text file per document, often organised into one folder per category
# Smaller documents: One document (or document pair, optionally with metadata) per row in a CSV file.
# Let's look at our data and see what we've got. In Jupyter you can use any bash/shell command by starting a line with a !, and use {} to include python variables, like so:

# In[ ]:


get_ipython().system('ls  {path}')


# It looks like this competition uses CSV files. For opening, manipulating, and viewing CSV files, it's generally best to use the Pandas library, which is explained brilliantly in this book by the lead developer (it's also an excellent introduction to matplotlib and numpy, both of which I use in this notebook). Generally it's imported as the abbreviation pd.

# In[ ]:


import pandas as pd


# Let's set a path to our data:

# In[ ]:


df=pd.read_csv(path/'train.csv')
df


# In[ ]:


df.describe(include='object')


# We can see that in the 36473 rows, there are 733 unique anchors, 106 contexts, and nearly 30000 targets. Some anchors are very common, with "component composite coating" for instance appearing 152 times.
# 
# Earlier, I suggested we could represent the input to the model as something like "TEXT1: abatement; TEXT2: eliminating process". We'll need to add the context to this too. In Pandas, we just use + to concatenate, like so:

# In[ ]:


df['input']='Text1:' + df.context +' ;Text2:'+ df.target +' ;text3:'+ df.anchor


# In[ ]:


df.input.head()


# # Tokenization

# Transformers uses a Dataset object for storing a... well a dataset, of course! We can create one like so:

# In[ ]:


from datasets import Dataset,DatasetDict
ds=Dataset.from_pandas(df)


# In[ ]:


ds


# But we can't pass the texts directly into a model. A deep learning model expects numbers as inputs, not English sentences! So we need to do two things:
# 
# Tokenization: Split each text up into words (or actually, as we'll see, into tokens)
# Numericalization: Convert each word (or token) into a number.
# The details about how this is done actually depend on the particular model we use. So first we'll need to pick a model. There are thousands of models available, but a reasonable starting point for nearly any NLP problem is to use this (replace "small" with "large" for a slower but more accurate model, once you've finished exploring):

# In[ ]:


model_nm='microsoft/deberta-v3-small'


# AutoTokenizer will create a tokenizer appropriate for a given model:

# In[ ]:


from transformers import AutoModelForSequenceClassification,AutoTokenizer
tokz=AutoTokenizer.from_pretrained(model_nm)


# Here's an example of how the tokenizer splits a text into "tokens" (which are like words, but can be sub-word pieces, as you see below):

# In[ ]:


tokz.tokenize('hi,im zeeba. Iam learning NLP using deep learning ')


# Uncommon words will be split into pieces. The start of a new word is represented by ▁:

# Here's a simple function which tokenizes our inputs:

# 

# In[ ]:


def tok_func(x):
    return tokz(x['input'])


# To run this quickly in parallel on every row in our dataset, use map:

# 

# In[ ]:


tokz_ds=ds.map(tok_func,batched=True)


# In[ ]:


tokz_ds


# This adds a new item to our dataset called input_ids. For instance, here is the input and IDs for the first row of our data:

# In[ ]:


row=tokz_ds[0]
row['input'],row['input_ids']


# So, what are those IDs and where do they come from? The secret is that there's a list called vocab in the tokenizer which contains a unique integer for every possible token string. We can look them up like this, for instance to find the token for the word "of":

# In[ ]:


tokz.vocab['▁of']


# In[ ]:


tokz.vocab['▁pollution']


# Looking above at our input IDs, we do indeed see that 265 appears as expected.
# 
# Finally, we need to prepare our labels. Transformers always assumes that your labels has the column name labels, but in our dataset it's currently score. Therefore, we need to rename it:

# In[ ]:


tokz_ds


# In[ ]:


tokz_ds=tokz_ds.rename_columns({'score':'labels'})
tokz_ds


# Now that we've prepared our tokens and labels, we need to create our validation se

# 

# # Test and validation sets

# You may have noticed that our directory contained another file:

# In[ ]:


eval_df=pd.read_csv(path/'test.csv')
eval_df


# This is the test set. Possibly the most important idea in machine learning is that of having separate training, validation, and test data sets.

# # Validation set

# In[ ]:


dds=tokz_ds.train_test_split(0.25,seed=42)


# As you see above, the validation set here is called test and not validate, so be careful!

# # Test set

# So that's the validation set explained, and created. What about the "test set" then -- what's that for?
# 
# The test set is yet another dataset that's held out from training. But it's held out from reporting metrics too! The accuracy of your model on the test set is only ever checked after you've completed your entire training process, including trying different models, training methods, data processing, etc.
# 
# You see, as you try all these different things, to see their impact on the metrics on the validation set, you might just accidentally find a few things that entirely coincidentally improve your validation set metrics, but aren't really better in practice. Given enough time and experiments, you'll find lots of these coincidental improvements. That means you're actually over-fitting to your validation set!
# 
# That's why we keep a test set held back. Kaggle's public leaderboard is like a test set that you can check from time to time. But don't check too often, or you'll be even over-fitting to the test set!
# 
# Kaggle has a second test set, which is yet another held-out dataset that's only used at the end of the competition to assess your predictions. That's called the "private leaderboard". Here's a great post about what can happen if you overfit to the public leaderboard.
# 
# We'll use eval as our name for the test set, to avoid confusion with the test dataset that was created above.

# In[ ]:


eval_df['input']='Text 1:'+df.context +'; Text2:' +df.target + '; Text3:'+df.anchor
eval_ds=Dataset.from_pandas(eval_df).map(tok_func,batched=True)


# In[ ]:


eval_ds


# # # # Metrics and correlation

# 

# When we're training a model, there will be one or more metrics that we're interested in maximising or minimising. These are the measurements that should, hopefully, represent how well our model will works for us.
# 
# In real life, outside of Kaggle, things not easy... As my partner Dr Rachel Thomas notes in The problem with metrics is a big problem for AI:
# 
# At their heart, what most current AI approaches do is to optimize metrics. The practice of optimizing metrics is not new nor unique to AI, yet AI can be particularly efficient (even too efficient!) at doing so. This is important to understand, because any risks of optimizing metrics are heightened by AI. While metrics can be useful in their proper place, there are harms when they are unthinkingly applied. Some of the scariest instances of algorithms run amok all result from over-emphasizing metrics. We have to understand this dynamic in order to understand the urgent risks we are facing due to misuse of AI.
# 
# In Kaggle, however, it's very straightforward to know what metric to use: Kaggle will tell you! According to this competition's evaluation page, "submissions are evaluated on the Pearson correlation coefficient between the predicted and actual similarity scores." This coefficient is usually abbreviated using the single letter r. It is the most widely used measure of the degree of relationship between two variables.
# 
# r can vary between -1, which means perfect inverse correlation, and +1, which means perfect positive correlation. The mathematical formula for it is much less important than getting a good intuition for what the different values look like. To start to get that intuition, let's look at some examples using the California Housing dataset, which shows "is the median house value for California districts, expressed in hundreds of thousands of dollars". This dataset is provided by the excellent scikit-learn library, which is the most widely used library for machine learning outside of deep learning.

# In[ ]:


def show_corr(df, a, b):
    x,y = df[a],df[b]
    plt.scatter(x,y, alpha=0.5, s=4)
    plt.title(f'{a} vs {b}; r: {corr(x, y):.2f}')


# In[ ]:


def corr(x,y): return np.corrcoef(x,y)[0][1]


# 

# 

# In[ ]:


def corr_d(eval_pred): 
    return {'pearson': corr(*eval_pred)}


# # # Training our model

# To train a model in Transformers we'll need this:

# In[ ]:


from transformers import TrainingArguments,Trainer


# We pick a batch size that fits our GPU, and small number of epochs so we can run experiments quickly:

# In[ ]:


bs=128
epoch=4


# Transformers uses the TrainingArguments class to set up arguments. Don't worry too much about the values we're using here -- they should generally work fine in most cases. It's just the 3 parameters above that you may need to change for different models.

# In[ ]:


lr=8e-5


# In[ ]:


args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=False,
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epoch, weight_decay=0.01, report_to='none')



# We can now create our model, and Trainer, which is a class which combines the data and model together (just like Learner in fastai):

# In[ ]:


model=AutoModelForSequenceClassification.from_pretrained(model_nm,num_labels=1)
trainer=Trainer(model,args,train_dataset=dds['train'],eval_dataset=dds['test'],tokenizer=tokz,
                compute_metrics=corr_d)


# As you see, Transformers spits out lots of warnings. You can safely ignore them.
# 
# Let's train our model!

# In[ ]:


trainer.train();


# Lots more warning from Transformers again -- you can ignore these as before.
# 
# The key thing to look at is the "Pearson" value in table above. As you see, it's increasing, and is already above 0.8. That's great news! We can now submit our predictions to Kaggle if we want them to be scored on the official leaderboard. Let's get some predictions on the test set:

# In[ ]:


preds=trainer.predict(eval_ds).predictions.astype(float)

preds


# Look out - some of our predictions are <0, or >1! This once again shows the value of remember to actually look at your data. Let's fix those out-of-bounds predictions

# In[ ]:


preds=np.clip(preds,0,1)


# In[ ]:


preds


# OK, now we're ready to create our submission file. If you save a CSV in your notebook, you will get the option to submit it later.

# In[ ]:


import datasets

submission=datasets.Dataset.from_dict({
    'id':eval_ds['id'],
    'score' : preds
})
submission.to_csv('submission.csv',index=False)


# Unfortunately this is a code competition and internet access is disabled. That means the pip install datasets command we used above won't work if you want to submit to Kaggle. To fix this, you'll need to download the pip installers to Kaggle first, as described here. Once you've done that, disable internet in your notebook, go to the Kaggle leaderboards page, and click the Submission button.
