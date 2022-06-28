#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import io


import plotly.express as px


# In[ ]:


df = pd.read_csv('/kaggle/input/student-mental-health/Student Mental health.csv')


# In[ ]:





# In[ ]:


df


# In[ ]:


#Copying the data of df into new dataframe so that the real data remain unchanged
student_mental_health_df = df.copy() 


# In[ ]:


student_mental_health_df.info() #cecking the data type of all columns



# In[ ]:





# From above we can say that Timestamp datatype is object and one age is missing, So we have to change the data type of Timestamp and find out the row whose age is nan and replace it with the average of the values of above and bottom.

# In[ ]:


#changing the data type of Timestamp
student_mental_health_df['Timestamp'] = pd.to_datetime(student_mental_health_df['Timestamp'],errors = 'coerce')


# In[ ]:


student_mental_health_df.info()


# In[ ]:


student_mental_health_df.describe()


# In[ ]:


#finding out the column number whose age is unknown to us
student_mental_health_df['Age'][student_mental_health_df['Age'].isna()==True]


# In[ ]:


student_mental_health_df.loc[40:45]


# In[ ]:


#replacing nan with average
stu_age_42 = student_mental_health_df['Age'][42]
stu_age_44 = student_mental_health_df['Age'][44]
stu_age_43 = student_mental_health_df['Age'][43]
stu_age_43=(stu_age_42+stu_age_44)/2


# In[ ]:


#rounding of it nearest integer
stu_age_43=stu_age_43.round(0)


# In[ ]:


stu_age_43


# In[ ]:


#Counting Courses which are opted by most of the student
course_count=student_mental_health_df['What is your course?'].value_counts().reset_index().head()
course_count


# In[ ]:


#Counting number of Female and Male
count_gender=student_mental_health_df['Choose your gender'].value_counts().reset_index()
count_gender


# In[ ]:


#Finding Mean Age of Female and Male Respectively
student_mental_health_df['Age'].groupby(student_mental_health_df['Choose your gender']).mean().reset_index()


# In[ ]:


#Finding Number of Students who are married and not married respectively
stu_marital_status = student_mental_health_df['Marital status'].value_counts().reset_index()
stu_marital_status


# In[ ]:


#percentage of students facing depression
facing_deppression = student_mental_health_df['Do you have Depression?'][student_mental_health_df['Do you have Depression?']=='Yes'].count()

dep_percent= (facing_deppression/student_mental_health_df.shape[0])*100
dep_percent


# In[ ]:


#percentage of students facing anxiety
facing_anxiety = student_mental_health_df['Do you have Anxiety?'][student_mental_health_df['Do you have Anxiety?']=='Yes'].count()
anx_percent = (facing_anxiety/student_mental_health_df.shape[0])*100
anx_percent


# In[ ]:


#percentage of student having panic attacks
facing_panic_attack = student_mental_health_df['Do you have Panic attack?'][student_mental_health_df['Do you have Panic attack?']=='Yes'].count()
panic_attack_percent = (facing_panic_attack/student_mental_health_df.shape[0])*100
panic_attack_percent


# In[ ]:


#creating a dataframe of students facing all three problems
students_facingall_df= student_mental_health_df[student_mental_health_df[['Do you have Depression?','Do you have Anxiety?','Do you have Panic attack?']].nunique(axis=1)==1]
students_facingall_df[students_facingall_df['Do you have Depression?']=='Yes']
                          


# In[ ]:


#Counting students who have consulted any specialist for trreatment or not
student_mental_health_df['Did you seek any specialist for a treatment?'].value_counts().reset_index()


# In[ ]:


#Gender Distribution
plt.figure(figsize=(11,10))
plt.hist(student_mental_health_df['Choose your gender'],color = 'g')
plt.title('Gender Distribution')


# In[ ]:


#countplot facing Depression and panic attack
plt.figure(figsize=(10,10))
sns.countplot(student_mental_health_df['Do you have Depression?'],hue=student_mental_health_df['Do you have Panic attack?']);
plt.title("Students facing Depression and Panic attack")


# In[ ]:


#Course Distrbution
stu_course = student_mental_health_df['What is your course?'].value_counts().reset_index()
stu_course.columns = ['What is your course?', 'value_counts']
px.pie(stu_course,names='What is your course?',values='value_counts', width = 600, height = 1500)


# In[ ]:


#CGPA distibution
stu_cgpa = student_mental_health_df['What is your CGPA?'].value_counts().reset_index()
stu_cgpa.columns = ['What is your CGPA?', 'value_counts']
px.pie(stu_cgpa,names='What is your CGPA?',values='value_counts', width = 600, height = 1500)


# 

# In[ ]:




