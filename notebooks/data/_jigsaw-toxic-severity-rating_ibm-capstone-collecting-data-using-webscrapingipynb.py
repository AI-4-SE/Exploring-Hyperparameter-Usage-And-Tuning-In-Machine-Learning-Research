#!/usr/bin/env python
# coding: utf-8

# **IBM CAPSTONE - webscraping**
# 
# scrape data of programming languages and the associated average salary from a URL

# import the modules needed

# In[ ]:


from bs4 import BeautifulSoup # this module helps in web scrapping.
import requests  # this module helps us to download a web page
import pandas as pd


# In[ ]:


#this url contains the data you need to scrape
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/labs/datasets/Programming_Languages.html"


# In[ ]:


data = requests.get(url).text


# In[ ]:


soup=BeautifulSoup(data, "html5lib")


# In[ ]:


table=soup.find('table')
def get_data(table):
    language_salary_list = []
    for row in table.find_all('tr'):
        cols=row.find_all('td')
        language_name = cols[1].getText()
        annual_average_salary = cols[3].getText()
    #     print("{}:{}".format(language_name, annual_average_salary))
        language_salary_list.append([language_name, annual_average_salary])
    language_salary_list.remove(['Language', 'Average Annual Salary'])
    return language_salary_list

data=get_data(table)
data


# In[ ]:


df = pd.DataFrame(data, columns=['Language', 'Avg Salary'])
df


# Q1. Which language are developers paid the most according to the output of the web scraping lab?

# In[ ]:


q1 = df.copy()
q1['Avg Salary'] = q1['Avg Salary'].str.replace('$', '').str.replace(',', '').astype(int)
q1.sort_values('Avg Salary', ascending = False)


# Q2. Code segments to scrape images: soup.find_all(“img”)
