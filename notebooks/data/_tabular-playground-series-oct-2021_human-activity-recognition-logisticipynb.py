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


# # **IMPORTING LIBRARIES**

# In[ ]:


#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # **LOADING THE DATASET**

# **READING THE DATASET**

# In[ ]:


#reading dataset

df = pd.read_csv('../input/human-activity-recognition-dataset/Human_activity_Dataset.csv')


# **DISPLAYING FIRST FIVE RECORDS**

# In[ ]:


#displaying the first 5 records

df.head()


# > **Our aim is to predict the Activity using the features provided. The activities are of 6 types - hence this is a classification problem**

# **DISPLAYING THE DATASET**

# In[ ]:


#number of rows and columns

df.shape


# > There are 10299 rows and 563 columns in the dataset

# In[ ]:


#displaying all the columns in the dataset

df.columns


# In[ ]:


#displaying the data types of the columns

df.dtypes


# # **STATISTICS OF THE DATASET**

# >**Statistics of the dataset**

# In[ ]:


#statistical details of the dataset

df.describe()


# > There are no missing values in the dataset

# **NULL VALUES**

# In[ ]:


#checking for null values

df.isnull().sum().sort_values(ascending=False)


# In[ ]:


#checking for null values - displaying the first five records

df.isnull().sum().sort_values(ascending=False).head(5)


# In[ ]:


#checking for null values - - displaying the first last records

df.isnull().sum().sort_values(ascending=False).tail(5)


# **MISSING VALUES**

# In[ ]:


#checking for missing values

df.isna().sum().sort_values(ascending=False)


# In[ ]:


#checking for missing values - displaying the first last records

df.isna().sum().sort_values(ascending=False).head(5)


# In[ ]:


#checking for missing values - displaying the last last records

df.isna().sum().sort_values(ascending=False).tail(5)


# **DUPLICATES**

# In[ ]:


#checking for duplicates

print('Number of duplicate entries in the dataset {}'.format(sum(df.duplicated())))


# **GROUPING**

# > Grouping to find the number of records for each type of label

# In[ ]:


# grouping the target variable - to check the count for each type

df['Activity'].groupby(df['Activity']).count()


# # **DATA VISUALIZATION**

# **COUNTPLOT**

# In[ ]:


# Plotting data with respect to subject

sns.set_style('whitegrid')
plt.figure(figsize=(20,10))
plt.title('Observations per User', fontsize=20)
sns.countplot(x='subject/Participant', hue='Activity', data=df)
plt.plot()


# In[ ]:


# Let's check number of observations per label

plt.title('Number of Observation per Activity', fontsize=20)
sns.countplot(df.Activity)
plt.xticks(rotation=90)
plt.show() 


# **PIE CHART**

# In[ ]:


# pie chart to show the distribution of the data

import plotly.express as px

fig = px.pie(df, names='Activity',height=400,width=600,title='Percentage of Distribution of Activity')
fig.show()


# > This shows the distribution of the target variable 'Activity' ******

# In[ ]:


#subject/Participant count

df['subject/Participant'].sort_values().unique()


# > there are 30 number of subject/Participants

# **HISTOGRAM**

# > Depicting the datapoints with respect to subject/Participant

# In[ ]:


px.histogram(data_frame=df,x='subject/Participant',color='Activity',barmode='group')


# **BAR PLOT**

# In[ ]:


fig = px.box(data_frame=df,x='Activity',y='subject/Participant',width=800)
fig.show()


# # **TRAIN TEST SPLIT**

# In[ ]:


# Dropping the subject column because it will not affect the dataset


data_set = df.drop('subject/Participant',axis=1)


# In[ ]:


#separating the features and labels

x = df.drop(columns='Activity')
y = df['Activity']


# In[ ]:


#features which are the independent variables

x


# In[ ]:


#label which is the dependent variable

y


# # **STANDARDIZATION**

# **MIN MAX SCALING**

# In[ ]:


#Min Max Scaling - importing libraries

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


#model building

scaler = MinMaxScaler()


# In[ ]:


#model fitting

scaled = scaler.fit_transform(x)


# In[ ]:


#displaying the scaled values

x_scaled = pd.DataFrame(scaled)
x_scaled


# In[ ]:


#statistical details of the scaled dataset

x_scaled.describe()


# In[ ]:


#finding the correlations in the scaled dataset

x_corr = x_scaled.corr()
x_corr.head()


# **FEATURE SELECTION**

# > Let us use correlation to select the features(choosing those with correlation with 0.95)

# In[ ]:


#selecting the feature with only Correlation 0.95

corr_columns = set()
for i in range(len(x_corr.columns)):
    for j in range(i):
        if (x_corr.iloc[i, j]) > 0.95:
            corr_columns.add(x.columns[i])
len(corr_columns)
print('Number of features:', len(corr_columns))


# In[ ]:


#choosing the 292 features for model building

final_df = x.loc[:,x.columns.isin(list(corr_columns))]


# In[ ]:


#number of rows and columns of our final dataset

final_df.shape


# In[ ]:


# heat map for the selected features

f, ax = plt.subplots(figsize =(20, 20))
sns.heatmap(final_df.corr(), ax = ax, cmap ="YlGnBu", linewidths = 0.1)


# # **TRAIN TEST SPLIT**

# **IMPORTING LIBRARIES**

# In[ ]:


#import library

from sklearn.model_selection import train_test_split


# In[ ]:


#splitting the dataset into train and test set

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=101)


# # **LABEL ENCODING**

# > Let us convert out target variable (label column) to numberic values hence, changing the 6 types from categorical to numerical values using Label Encoding technique

# **IMPORTING LIBRARIES**

# In[ ]:


#importing library from sklearn library

from sklearn.preprocessing import LabelEncoder


# **MODEL BUILDING**

# In[ ]:


#train set

le = LabelEncoder()
y_train = le.fit_transform(y_train)


# In[ ]:


#test set

le = LabelEncoder()
y_test = le.fit_transform(y_test)


# In[ ]:


#mapping to numerical values

le_mapping = dict(zip(le.classes_,le.transform(le.classes_)))
le_mapping


# # **LOGISTIC REGRESSION**

# **IMPORTING LIBRARY**

# In[ ]:


#importing library

rom sklearn.linear_model import LogisticRegression


# **MODEL BUILDING**

# In[ ]:


#Logistic Regression model building

log = LogisticRegression(penalty='l2',C=0.01, multi_class='ovr',solver='lbfgs',class_weight='balanced',max_iter=10000,random_state=101)


# **MODEL FITTING**

# In[ ]:


log.fit(x_train,y_train)


# **MODEL PREDICTION**

# In[ ]:


y_pred = log.predict(x_test)
y_pred


# # **PERFORMANCE METRICS**

# **IMPORTING LIBRARIES**

# In[ ]:


#importing from sklearn library

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# **CONFUSION MATRIX**

# In[ ]:


confusion_matrix(y_test,y_pred)


# **CLASSIFICATION REPORT**

# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


accuracy_score(y_test,y_pred)


# > **Our model has an accuracy score of 94%**

# **PLEASE UPVOTE IF YOU FOUND THE CONTENT HELPFUL :)**
