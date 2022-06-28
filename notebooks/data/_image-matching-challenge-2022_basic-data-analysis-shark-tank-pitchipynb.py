#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[ ]:


#importin csv file
shark_tank_df = pd.read_csv('../input/shark-tank-data-121pitches/Shark Tank Data complete 121pitches.csv')


# In[ ]:





# In[ ]:


shark_tank_df.info() #finding  out the information about the dataframe


# 

# Above Informatin tells us that there are two columns which are **Unamed :23** and **Unamed:24** whic don't have any values,So even if drop them from the  table they won't produce any impact on the studying of the data.
# And we also noticed that the data type of Column **Ashneer Grover** is int which we are going to convert it into float

# In[ ]:


shark_tank_df


# In[ ]:


shark_tank_df['Deal_% of Equity ']


# In[ ]:


df = shark_tank_df.copy() #producing a copy of dataframe so that it won't impact the actual data


# In[ ]:


#changing the data type of Ashneer Grover
df['Ashneer Grover'] = df['Ashneer Grover'].astype(float)


# In[ ]:


df.describe()


# In[ ]:


df=df.drop(columns=['Unnamed: 23','Unnamed: 24']) #droping columns


# In[ ]:


df.shape


# In[ ]:


list(df.columns) #listing down the columnsafter removal of two columns


# In[ ]:


df['Ask_value'][df['Ask_value'].isna()==True]


# Here we found out that the Ask Value is Nan for row number - 46 and 81 respectively.
# But Here we cannot change the value of the column Ask_value with the overall mean of the column,because every individual start-up will have different Ask value and that depends upon various other factors of the Start-up.
# 
# So, You can either replace those nan with zero or Nan is also fine

# In[ ]:


max_Revenue=df[df['Revenue YoY']==df['Revenue YoY'].max()]


# In[ ]:


max_Revenue


# In[ ]:


min_revenue=df[df['Revenue YoY']==df['Revenue YoY'].min()]
min_revenue


# In[ ]:


Max_ask_value = df[df['Ask_value']==df['Ask_value'].max()]
Max_ask_value


# In[ ]:


Min_ask_value=df[df['Ask_value']==df['Ask_value'].min()]
Min_ask_value


# In[ ]:


df['Sex'].value_counts()


# In[ ]:


df[df['Deal']=='N'].head(3)


# In[ ]:


#Here we are trying to check if there was any startup where every investor has invested money 
valued_all=df[df[['Ashneer Grover','Vineeta Singh','Peyush Bansal','Namita Thapar','Anupam Mittal','Ghazal Alagh']].nunique(axis=1)==1]
valued_all[valued_all['Anupam Mittal']!=0.0]


# In[ ]:


plt.figure(figsize=(11,10))
plt.hist(df['Sex'])
plt.title('Gender Distribution')


# In[ ]:


plt.figure(figsize=(12,12))
sector_count = df['Sector'].value_counts().reset_index()
sector_count.columns=['Sector','no_of_sector']
sns.barplot(data=sector_count,x='no_of_sector',y='Sector')


# In[ ]:





# In[ ]:


plt.figure(figsize=(20,20))
sns.barplot(x='Revenue YoY',y='Start-Up Name',data=df)


# In[ ]:





# In[ ]:


total_investment_aman_gupta = df['Aman Gupta '].sum()
total_investment_Ashneer_Grover = df['Ashneer Grover'].sum()
total_investment_Vineeta_Singh = df['Vineeta Singh'].sum()
total_investment_Peyush_bansal  = df['Peyush Bansal'].sum()
total_investment_Namita_Thapar = df['Namita Thapar'].sum()
total_investment_anupam_mittal = df['Anupam Mittal'].sum()
total_investment_Ghazal_Alagh = df['Ghazal Alagh'].sum()
li=[]
li.append(total_investment_aman_gupta)
li.append(total_investment_Ashneer_Grover)
li.append(total_investment_Vineeta_Singh)
li.append(total_investment_Peyush_bansal)
li.append(total_investment_Namita_Thapar)
li.append(total_investment_anupam_mittal)
li.append(total_investment_Ghazal_Alagh)
investor_names=['Aman Gupta','Ashneer Grover','Vineeta Singh','Peyush Bansal','Namita Thapar','Anupam Mittal','Ghazal Alagh']
plt.figure(figsize=(12,10))
sns.barplot(x=investor_names,y=li)


# In[ ]:




