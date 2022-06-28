#!/usr/bin/env python
# coding: utf-8

# ## Welcome to this notebook!
# 
# # <b style="color:lightgreen; font-weight:1200"> | </b> What is this about?
# 
# This notebook is about Titanic Survival Prediction using **LightPredict** library. Well, never heard of it right?!
# 
# Because it's made by me!😁
# 
# I was practising Python. So, I thought to take some project up 🚀. As we have to select our training models, we need to check various models for that purpose. It can be time consuming⌛. So, I tried to create a library called **LightPredict** which automatically fits ⚡ most commonly used Sklearn models to the data and shows their score in tabular form.
# 
# Yes, this library has been inspired by **LazyPredict**. But I've took it a little step further as you'll see. I'll keep adding different functionalities to it and keep it as my hobby project. 
# 
# So, let me give you an introduction to it.
# 
# Also, the name's different right?! Well, whatever names I came up with were already taken! 🥲
# 
# So, let's start!

# # 1 <b style="color:lightgreen; font-weight:1200"> | </b> Usual Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train.head(5)


# In[ ]:


train.isnull().sum()


# In[ ]:


train.isna().mean()


# # 2 <b style="color:lightgreen; font-weight:1200"> | </b> Preprocessing and Feature Engineering

# In[ ]:


train.drop('Cabin', axis=1, inplace=True)


# In[ ]:


train[train["Embarked"] == "S"].shape[0], train[train["Embarked"] == "C"].shape[0], train[train["Embarked"] == "Q"].shape[0]


# In[ ]:


train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Embarked'] = train['Embarked'].fillna('S')


# In[ ]:


train.drop(['Name','PassengerId','Ticket', 'SibSp'], axis=1, inplace=True)


# In[ ]:


mask = np.zeros_like(train.corr())
tri_indices = np.triu_indices_from(mask)
mask[tri_indices] = True
mask


# In[ ]:


plt.figure(figsize=[10, 8])
sns.heatmap(data=train.corr(), annot=True, mask=mask)


# In[ ]:


train['Sex'].replace('male', 0, inplace=True)
train['Sex'].replace('female', 1, inplace=True)

train['Embarked'].replace('S', 0, inplace=True)
train['Embarked'].replace('C', 1, inplace=True)
train['Embarked'].replace('Q', 2, inplace=True)


# In[ ]:


train.head()


# In[ ]:


ids = test.PassengerId


# In[ ]:


test.isnull().sum()


# In[ ]:


test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Embarked'] = test['Embarked'].fillna('S')


# In[ ]:


test['Sex'].replace('male', 0, inplace=True)
test['Sex'].replace('female', 1, inplace=True)

test['Embarked'].replace('S', 0, inplace=True)
test['Embarked'].replace('C', 1, inplace=True)
test['Embarked'].replace('Q', 2, inplace=True)


# In[ ]:


test.drop(['PassengerId', 'Name', 'SibSp', 'Ticket', 'Cabin'], axis=1, inplace=True)


# In[ ]:


test.head(5)


# In[ ]:


X = train.drop('Survived', axis=1)
y = train['Survived']


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# # 3 <b style="color:lightgreen; font-weight:1200"> | </b> Training using LightPredict

# First, we need to partition our data into training and testing sets. There are many methods. Here, **train_test_split** is used.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# Then, install the **lightpredict** library using **pip install**.

# In[ ]:


pip install lightpredict


# **LightPredict** has two modules:
# - **LightClassifier**: for classification and hyperparam. tuning;
# - **LightRegressor**: for regression.
# 
# As **Titanic** is a case of classification, we can use **LightClassifier** for that purpose.

# In[ ]:


from lightpredict import LightClassifier

import warnings
warnings.filterwarnings('ignore')


# # 4 <b style="color:lightgreen; font-weight:1200"> | </b> LightClassifier

# LightClassifier takes following arguments:
# - **x_train**
# - **x_test**
# - **y_train**
# - **y_test**
# 
# Above ones are the usual ones that we use. In addition to them, it has 2 more arguments:
# - **rounds**: the no. of digits to round off the result; and
# - **plot**: plot=True plots the accuracy score graph of various models for visual comparison.

# In[ ]:


lcf = LightClassifier()
lcf.fit(X_train, X_test, y_train, y_test, rounds=3)


# In[ ]:


lcf = LightClassifier()
lcf.fit(X_train, X_test, y_train, y_test, rounds=3, plot=True)


# # 5 <b style="color:lightgreen; font-weight:1200"> | </b> LightClassifier: ROC-AUC curve
# 
# **LightClassifier** also has a function that automatically plots **roc_auc** curves of models. Just call the following code:

# In[ ]:


lcf.roc_auc_curves(X_train, X_test, y_train, y_test)


# # 6 <b style="color:lightgreen; font-weight:1200"> | </b> LightClassifier: Optimizing Models
# 
# **LightClassifier** can also optimize some models using **Optuna** and return the new **optimized scores** along with the **best params** that led to those scores.
# 
# The **optimize** function takes:
# - **x_train**
# - **x_test**
# - **y_train**
# - **y_test**
# - **trials**: the no. of iterations to undergo
# - **plot**: plot=True plots the optimization and params importance plots

# In[ ]:


lcf.optimize(X_train, X_test, y_train, y_test,trials=10) # Here, trials means no. of iterations 


# In[ ]:


lcf.optimize(X_train,X_test,y_train,y_test,trials=10, plot=True)


# # <b style="color:lightgreen; font-weight:1200"> | </b> The End
# 
# This is all it is in the library right now. Some things could be wrong or off-track.
# 
# But to think that I've built something like that gives me more motivation 💪 for Machine Learning. 
# 
# Thanks for reading and I hope this library may be of some use to you!😄
# 
# Also, if you like it check its <a href='https://github.com/arnavrneo/LightPredict'><em>Github repo</em></a>.
