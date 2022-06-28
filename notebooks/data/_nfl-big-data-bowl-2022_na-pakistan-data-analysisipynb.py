#!/usr/bin/env python
# coding: utf-8

# # National Assembly of Pakistan Data Analysis
# 
# The dataset contains data of all the members of National Assembly of Pakistan. It contains around 340 rows which have columns like NASeat, PersonName, Party, Profession, Contact details.
# 
# We will try to answer the following questions based on the data
# 
# 1) Which party contains how many members in the assembly. Furthermore,
# 
# 2) Which party contains the most educated members.
# 
# We will initiate our analysis from importing libraries, loading data, doing exploratory data analysis and then data visualization based on the given data
# 
# ## Step1: Import Useful Libraries

# In[ ]:


# For data analysis, we import pandas library
import pandas as pd

# For numerical computation, we will use numpy library
import numpy as np

# For data visualization, we will use matplotlib and seaborn library
import matplotlib.pyplot as plt
import seaborn as sns


# ## Step2: Load and explore data

# **If openpyxl error occurs, then install it using the following syntax**
# 
# > pip install openpyxl

# In[ ]:


#load data from excel file
df = pd.read_excel('../input/national-assembly-of-pakistan/NA_list.xlsx')

#Display first five rows of data
df.head()


# In[ ]:


# Check shape of data
rows, columns = df.shape

print(f'In national assembly of pakistan dataset there are {rows} rows and {columns} columns')


# In[ ]:


# Check Structure of data
df.info()


# In[ ]:


# Check missing values in dataset
df.isnull().sum()


# **As we saw there are 89 missing values in contact columns, and in our dataset the presence of contact column making no sense, there we will remove this column**

# In[ ]:


#Remove contact column
df.drop('Contact', axis = 1, inplace = True)


# In[ ]:


# check 10 random sample values from our data
df.sample(10)


# ## Step3: Data Analysis and Data Visualization

# ### 1- Which party contains how many members in the assembly ?

# In[ ]:


# Check unique values in party column
list(df['Party'].unique())


# In[ ]:


#groupby the members using the party column
party_na = df.groupby('Party')['NA Seat'].count().sort_values(ascending = False)

#Convert the groupby output to dataframe
party_df = pd.DataFrame({'Party' : party_na.index, 'Members' : party_na.values})

#view dataframe
party_df


# **We have arranged the data in a way that how many national assembly members belongs to each party. Lets plot it**

# In[ ]:


#set the figure size
plt.figure(figsize=(20, 9))

#set theme
sns.set_theme(style="whitegrid")

#Bar plot
sns.barplot(x = 'Party', y= 'Members', data = party_df)
plt.title('Total Number of National Assembly Member based on political parties', size = 20)
plt.xlabel('Political Parties', size = 15)
plt.ylabel('Total NA Members', size = 15)
plt.show()


# **We saw that PTI has the maximum numbers of national assembly members i.e. 155, PML-N has 84 members and PPP has 56 members**

# ### 2- Which party contains the most educated national assembly members
# 
# To give the answers with respect to given data, we individually visualize the data of most common party members qualification and education.
# 
# #### 2.1- Qualifications of National Assembly Members of PTI

# In[ ]:


plt.figure(figsize=(20, 9))
df.groupby('Party')['Profession/Education'].value_counts().loc['PTI'].plot(kind = 'bar')
plt.title('Qualifications of National Assembly members of PTI', size = 20)
plt.show()


# **PTI has the most educated members of national assembly of pakistan in different subjects from which majority are graduated, LLB and BA**

# #### 2.2- Qualifications of National Assembly Members of PML-N

# In[ ]:


plt.figure(figsize=(20, 9))
df.groupby('Party')['Profession/Education'].value_counts().loc['PML-N'].plot(kind = 'bar')
plt.title('Qualifications of National Assembly members of PML-N', size = 20)
plt.show()


# **After PTI, the PML-N National Assembly members qualification are less as compared to PTI, but they majority did graduation and LLB**

# #### 2.3- Qualifications of National Assembly Members of PPPP

# In[ ]:


plt.figure(figsize=(20, 9))
df.groupby('Party')['Profession/Education'].value_counts().loc['PPPP'].plot(kind = 'bar')
plt.title('Qualifications of National Assembly members of PPPP', size = 20)
plt.show()


# **PPP contains less number of national assembly members with regards to qualification as compared to PML-N. They usually did bachelors in arts and graduation**

# #### 2.4- Qualifications of National Assembly Members of MNAP

# In[ ]:


plt.figure(figsize=(20, 9))
df.groupby('Party')['Profession/Education'].value_counts().loc['MMAP'].plot(kind = 'bar')
plt.title('Qualifications of National Assembly members of MMAP', size = 20)
plt.show()


# #### 2.5- Qualifications of National Assembly Members of MQMP

# In[ ]:


plt.figure(figsize=(20, 9))
df.groupby('Party')['Profession/Education'].value_counts().loc['MQMP'].plot(kind = 'bar')
plt.title('Qualifications of National Assembly members of MQMP', size = 20)
plt.show()


# ## Conclusion
# 
# **I have analyze the data of national assembly members of pakistan. I have explore and deal with the data, shape the data according to the questions and visualize the results. The results concluded are mostly members in national assembly of pakistan are of PTI, then PML-N, then PPP while the most educated members with different qualifications are of PTI.**
# 
# **I hope you have liked my work. If you like work, then please upvote the notebook.**
# 
# **Thankyou.**
