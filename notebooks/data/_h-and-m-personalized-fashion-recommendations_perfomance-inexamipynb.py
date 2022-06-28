#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool


# In[ ]:


#create dataframe
df = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
df.info()


# In[ ]:


#show first 10 rows
df.head(10)


# In[ ]:


# check columns of dataframe
df.columns


# In[ ]:


#update columns name

df.rename(columns={"race/ethnicity":"race_ethnicity"}, inplace=True)
df.rename(columns={"parental level of education":"parental_level_of_education"}, inplace=True)
df.rename(columns={"test preparation course":"test_preparation_course"}, inplace=True)
df.rename(columns={"math score":"math_score"}, inplace=True)
df.rename(columns={"reading score":"reading_score"}, inplace=True)
df.rename(columns={"writing score":"writing_score"}, inplace=True)
df.columns
df.head()


# In[ ]:


#check correlation
df.corr()


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


# Line Plot
df.reading_score.plot(kind = 'line', color = 'g',label = 'reading score',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
df.writing_score.plot(color = 'r',label = 'writing score',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')    
plt.xlabel('x axis')            
plt.ylabel('y axis')
plt.title('Line Plot')           
plt.show()


# In[ ]:


# Scatter Plot 
# x = reading score, y = writing score
df.plot(kind='scatter', x='reading_score', y='writing_score',alpha = 0.5,color = 'blue')
plt.legend(loc='upper right') 
plt.xlabel('reading_score')              
plt.ylabel('writing_score')
plt.title('reading_score and writing_score Scatter Plot')  
plt.show()


# In[ ]:


# Histogram math score
df.math_score.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# In[ ]:


df.head()


# In[ ]:





# In[ ]:


#show females who are math_score>50 
df[np.logical_and(df['math_score']>50, df['gender']=='female' )]


# In[ ]:


#while and for
for index,value in df[['race_ethnicity']][0:5].iterrows():
    print(index," : ",value)


# In[ ]:


#check null values
df.isnull().sum()


# In[ ]:


# for example count of passed status in  exam
passing_score=50
df['Math_Pass'] = np.where(df['math_score']>=passing_score, 'PASSED', 'FAILED')
df.Math_Pass.value_counts()




# In[ ]:


#or
pass_c=0
fail_c=0
for score in df['math_score']:
    if (score>=passing_score):
        pass_c+=1
    else:
        fail_c+=1
print('PASSED: ',pass_c)
print('FAILED: ',fail_c)
   


# In[ ]:


p = sns.countplot(x="math_score", data = df, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=90) 

