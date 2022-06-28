#!/usr/bin/env python
# coding: utf-8

# <img src="https://dpbnri2zg3lc2.cloudfront.net/en/wp-content/uploads/2019/12/data-analyst-colleagues-in-office.jpeg" width="1000">

# #### In this notebook I tried some data cleaning and visualizations. This dataset contains more than 3900 job listing for data scientist positions, with features such as: Salary Estimate, where are they Located, where's the Head Quarters of these Company, Job Description and many more. I am working to add insights from the outputs.

# In[ ]:


#importing libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# 

# In[ ]:


get_ipython().system('pip install -U kaleido')


# In[ ]:


#importing the dataset

df = pd.read_csv('../input/data-scientist-jobs/DataScientist.csv')
df.head()


# In[ ]:


print(df.isnull().sum()) 
print(df.info()) 


# In[ ]:


df['Easy Apply'] = df['Easy Apply'].fillna(False).astype(bool) #Easy Apply column has -1 values, replacing them with boolean value False
df['Easy Apply'].value_counts() 


# # **1. Data Cleaning**

# **Replacing -1 with nan**

# In[ ]:


df.replace(['-1'], [np.nan], inplace=True)
df.replace(['-1.0'], [np.nan], inplace=True)
df.replace([-1], [np.nan], inplace=True)


# In[ ]:


df.isnull().sum()  #After replacing -1 with nan, we can see that there are null values in the dataset


# **Creating separate columns of Salary Estimate as minimum and maximum salary**

# In[ ]:


df_salary = df['Salary Estimate'].str.split("-",expand=True,)

minimum_salary = df_salary[0]
minimum_salary = minimum_salary.str.replace('K',' ')


maximum_salary = df_salary[1].str.replace('(Glassdoor est.)', ' ')
maximum_salary = maximum_salary.str.replace('(', ' ')
maximum_salary = maximum_salary.str.replace(')', ' ')
maximum_salary = maximum_salary.str.replace('K', ' ')
maximum_salary = maximum_salary.str.replace('Employer est.', ' ')
maximum_salary = maximum_salary.str.replace('Per Hour', ' ')

maximum_salary = maximum_salary.str.replace('$', ' ').fillna(0).astype(int)
minimum_salary = minimum_salary.str.replace('$', ' ').fillna(0).astype(int)


# In[ ]:


maximum_salary.value_counts()


# In[ ]:


df['Minimum Salary'] = minimum_salary
df['Maximum Salary'] = maximum_salary

df.drop('Salary Estimate',axis = 1,inplace = True)
df['Company Name'] = df['Company Name'].str.replace('\n.*', ' ')
df['Est_Salary']= (df['Minimum Salary']+df['Maximum Salary'])/2


# **Making city and state columns for both Location and Headquaters**

# In[ ]:


Location = df['Location'].str.split(",",expand=True,)
Location_City = Location[0]
Location_State = Location[1]
df['Location City'] = Location_City
df['Location State'] = Location_State
df.drop('Location',axis = 1, inplace = True)


HQ = df['Headquarters'].str.split(",",expand=True)
Headquarters_City = HQ[0]
Headquarters_State = HQ[1]
df['Headquarters City'] = Headquarters_City
df['Headquarters State'] = Headquarters_State
df.drop('Headquarters',axis = 1, inplace = True)


# **Separating department and from job title column**

# In[ ]:


department = df['Job Title'].str.split(',', expand = True)
df['Job Title'], df['Department'] = department[0],department[1]


# Since, department has too many missing values (2023/2253), it can be dropped.

# In[ ]:


df.drop('Department',1, inplace = True)


# In[ ]:


df['Job Title'].value_counts()


# In[ ]:


df['Job Title'] = df['Job Title'].str.replace('Sr.', 'Senior')
df.head()


# Checking values from the columns for cleaning

# In[ ]:


df.info()


# In[ ]:


df['Type of ownership'].value_counts()


# In[ ]:


df['Industry'].value_counts()


# In[ ]:


df['Sector'].value_counts()


# In[ ]:


df['Revenue'].value_counts()


# **Cleaning the Revenue column**

# In[ ]:


df['Revenue'] = df['Revenue'].replace('Unknown / Non-Applicable', None)
# data['Revenue']=data['Revenue'].replace('Unknown / Non-Applicable', None)


# In[ ]:


df['Revenue'] = df['Revenue'].str.replace('$', ' ')
df['Revenue'] = df['Revenue'].str.replace('(USD)', ' ')
df['Revenue'] = df['Revenue'].str.replace('(', ' ')
df['Revenue'] = df['Revenue'].str.replace(')', ' ')
df['Revenue'] = df['Revenue'].str.replace(' ', '')


# In[ ]:


df['Revenue'].value_counts()


# In[ ]:


df['Revenue'] = df['Revenue'].str.replace('2to5billion', '2billionto5billion')
df['Revenue'] = df['Revenue'].str.replace('5to10billion ', '5billionto10billion ')


# In[ ]:


df['Revenue'].value_counts()


# In[ ]:


df['Revenue'] = df['Revenue'].replace('million', ' ')
df['Revenue'] = df['Revenue'].replace('10+billion', '10billionto11billion')
df['Revenue'] = df['Revenue'].str.replace('Lessthan1million', '0millionto1million')


# In[ ]:


df['Revenue'].value_counts()


# In[ ]:


df['Revenue'] = df['Revenue'].str.replace('million', ' ')
df['Revenue'] = df['Revenue'].str.replace('billion', '000 ')
df['Revenue'] = df['Revenue'].replace('Unknown/Non-Applicable', np.nan)


# In[ ]:


df['Revenue'].value_counts()


# In[ ]:


Revenue = df['Revenue'].str.split("to",expand=True)


# In[ ]:


df['Revenue'].value_counts()


# **Creating two separate columns of Revenue as Minimum and Maximum Revenue**

# In[ ]:


df['Minimum Revenue'] = Revenue[0]
df['Maximum Revenue'] = Revenue[1]


# In[ ]:


df['Maximum Revenue'] = pd.to_numeric(df['Maximum Revenue'])
df['Minimum Revenue'] = pd.to_numeric(df['Minimum Revenue'])


# In[ ]:


df.drop('Revenue',1,inplace=True)
df.head()


# **Cleaning the Size column**

# In[ ]:


df['Size'].value_counts()


# In[ ]:


df['Size'] = df['Size'].str.replace('employees', '')


# In[ ]:


df['Size'] = df['Size'].str.replace('+', 'plus')
df['Size'] = df['Size'].replace('Unknown', None)


# In[ ]:


df['Size'] = df['Size'].str.replace('10000plus', '10000 to 10001')


# In[ ]:


size = df['Size'].str.split("to",expand=True)


# **Creating separate columns of Size as minimum and maximum size**

# In[ ]:


df['Minimum Size'] = size[0]
df['Maximum Size'] = size[1]
df.head()


# In[ ]:


df.drop('Size',1,inplace = True)


# # 2. Statistics

# Distribution of minimum and maximum salary of all Data Science job titles

# In[ ]:





# In[ ]:


df['Minimum Size'] = df['Minimum Size'].astype('float')
df['Maximum Size'] = df['Maximum Size'].astype('float')


# In[ ]:


f, axes = plt.subplots(1, 2, figsize=(25, 4), sharex=True)
sns.boxplot(x = df['Minimum Size'], ax = axes[0]);
sns.boxplot(x = df['Maximum Size'], ax = axes[1],palette='Set2');


# In[ ]:


f, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True)
sns.despine(left=True)
sns.distplot(df['Minimum Salary'],color = 'darkred',ax = axes[0])
sns.distplot(df['Maximum Salary'],color = 'g' , ax = axes[1])
plt.legend();


# # 3. Data Visualization

# In[ ]:


"""

plt.subplots(figsize=(10,10))
splot = sns.barplot(x=df['Job Title'].value_counts()[0:20].index,y=df['Job Title'].value_counts()[0:20], palette = 'winter_r')
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 15), textcoords = 'offset points')

plt.xlabel('Job Title',fontsize=15)
plt.ylabel('Job Count',fontsize=15)
plt.xticks(rotation=90)
plt.yticks(fontsize=15)
plt.title('Top 20 Job Title Counts',fontsize=25);

"""


# In[ ]:


plt.rcParams['figure.figsize'] = (12,9)
color = plt.cm.viridis(np.linspace(0,1,20))
df["Job Title"].value_counts().sort_values(ascending=False).head(20).plot.bar(color=color)
plt.title("Top 20 Data Science Job",fontsize=20)
plt.xlabel("Job Title",fontsize=15)
plt.ylabel("Count",fontsize=15)
plt.show()


# In[ ]:


plt.rcParams["figure.figsize"] = (12,9)
color = plt.cm.plasma(np.linspace(0,1,20))
df["Company Name"].value_counts().sort_values(ascending=False).head(20).plot.bar(color=color)
plt.title("Top 20 Company with Highest number of Jobs in Data Science",fontsize=20)
plt.xlabel("Company Name",fontsize=15)
plt.ylabel("Count",fontsize=15)
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (12,9)
color = plt.cm.cool(np.linspace(0,1,20))
df["Headquarters City"].value_counts().sort_values(ascending=False).head(20).plot.bar(color=color)
plt.title("Top 20 Head Quarter City of Data Science Job Holder Company",fontsize=20)
plt.xlabel("Headquarters City",fontsize=15)
plt.ylabel("Count",fontsize=15)
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (12,9)
color = plt.cm.hot(np.linspace(0,1,20))
df["Location City"].value_counts().sort_values(ascending=False).head(20).plot.bar(color=color)
plt.title("Top 20 City for Data Science Job",fontsize=20)
plt.xlabel("Location City",fontsize=15)
plt.ylabel("Count",fontsize=15)
plt.show()


# In[ ]:


df_by_city=df.groupby('Location City')['Job Title'].count().reset_index().sort_values( 
    'Job Title',ascending=False).head(20).rename(columns={'Job Title':'Hires'})
Sal_by_city = df_by_city.merge(df,on='Location City',how='left')

sns.set(style="white")
f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))
sns.barplot(x='Hires',y='Location City',data=Sal_by_city,ax=ax_bar, palette='rocket').set(ylabel="")
sns.pointplot(x='Est_Salary',y='Location City',data=Sal_by_city, join=False,ax=ax_point, color = 'red').set(
    ylabel="",xlabel="Salary ($'000)")
plt.suptitle('Top 20 Cities Hiring Data Science Jobs',fontsize=20,color='darkblue')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# Creating 'Average Revenue' column

# In[ ]:


df['Average Revenue'] = df[['Minimum Revenue','Maximum Revenue']].mean(axis=1)


# In[ ]:


avg_rev = df['Average Revenue'][0:20]
avg_rev


# In[ ]:


data = df.groupby('Location City')[['Minimum Salary', 'Maximum Salary']].mean().sort_values(['Maximum Salary','Minimum Salary'],ascending=False).head(25)
data


# In[ ]:


import plotly.graph_objs as go
fig = go.Figure()
fig.add_trace(go.Bar(
   x = data.index,
   y = data['Minimum Salary'],
   name = 'Minimum Salary'
))

fig.add_trace(go.Bar(
   x = data.index,
   y = data['Maximum Salary'],
   name = 'Maximum Salary'
))

#data1 = [plot1,plot2]
fig.update_layout(title = 'Minimum and Maximum salaries of top 25 cities', barmode = 'group')
#fig = go.Figure(data = data, layout = layout)

fig.show()


# In[ ]:


data1 = df.groupby('Job Title')[['Minimum Salary', 'Maximum Salary']].mean().sort_values(['Maximum Salary','Minimum Salary'],ascending=False).head(25)
data1.head()


# In[ ]:


import plotly.graph_objs as go
fig = go.Figure()
fig.add_trace(go.Bar(
   x = data1.index,
   y = data1['Minimum Salary'],
   name = 'Minimum Salary'
))

fig.add_trace(go.Bar(
   x = data1.index,
   y = data1['Maximum Salary'],
   name = 'Maximum Salary'
))


fig.update_layout(title = 'Minimum and Maximum salaries of top 25 job titles', barmode = 'stack')
fig.show()


# In[ ]:


df['Average Salary'] = df[['Minimum Salary', 'Maximum Salary']].mean(axis = 1)


# In[ ]:


import plotly.express as px
fig = px.scatter(df, x=df['Rating'], y= df['Average Salary'], color = 'Average Salary')
fig.update_layout(title = 'Relation between average salary and rating of companies')
fig.show()


# In[ ]:


data3 = df.groupby('Founded')[['Average Revenue']].mean().sort_values(['Average Revenue'],ascending=False).tail(25)
data3.head()


# In[ ]:


fig = px.line(x=data3['Average Revenue'], y=data3.index, labels={'x':'Average Revenue', 'y':'Year founded'})
fig.update_layout(title = 'Relation between the average revenue and year the company was founded', template='plotly_dark')
fig.show()


# In[ ]:


data4 = pd.DataFrame(df['Sector'].value_counts())
data4.head()


# In[ ]:


import plotly.express as px
fig = px.pie(data4, values=data4['Sector'], names=data4.index)
fig.update_layout(title = 'Percentage of Different Sectors with requirement of Data Scientist  Roles')
fig.show()


# In[ ]:


data5 = pd.DataFrame(df['Industry'].value_counts().head(25))
data5


# In[ ]:


fig = px.pie(data5, values=data5['Industry'], names=data5.index)
fig.update_layout(title = 'Percentage of top 25 Industries with requirement of Data Analyst Roles')
fig.show()


# In[ ]:


data6 = pd.DataFrame(df['Type of ownership'].value_counts())
data6
fig = px.pie(data6, values=data6['Type of ownership'], names=data6.index)
fig.update_layout(title = 'Type of ownership')
fig.show()


# In[ ]:


data7 = pd.DataFrame(df['Headquarters City'].value_counts().head(20))
data7
fig = px.pie(data7, values=data7['Headquarters City'], names=data7.index)
fig.update_layout(title = 'Top 20 Headquarter City')
fig.show()


# In[ ]:


data8 = pd.DataFrame(df['Location City'].value_counts().head(20))
data8

fig = px.pie(data8, values=data8['Location City'], names=data8.index)
fig.update_layout(title = 'Top 20 Job Locations')
fig.show()


# <h2> Word Cloud of Job Titles <h2>

# In[ ]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
stopwords = set(STOPWORDS)


# <h3>1. Job Title </h3>

# In[ ]:


plt.subplots(figsize=(15,15))
wc = WordCloud()
text = df['Job Title']
wc.generate(str(' '.join(text)))
plt.imshow(wc)
plt.axis("off")
plt.title("Most available Job Title")
plt.show()


# So, Data Scientist, Data Engineer, Data Analyst, Senior Data Scientist are the most available jobs.
# <h3>2. Company Name</h3>

# In[ ]:


plt.subplots(figsize=(15,15))
wc = WordCloud()
text = df["Company Name"]
wc.generate(str(' '.join(text)))
plt.imshow(wc)
plt.axis("off")
plt.title("Most available Company")
plt.show()


# <h3>3. Head Quarters</h3>

# In[ ]:


wordcloud = WordCloud(height =3000,width = 6000).generate(str(df["Headquarters City"]))
plt.rcParams['figure.figsize'] = (15,15)
plt.axis("off")
plt.imshow(wordcloud)
plt.title("Frequently cities")
plt.show()

