#!/usr/bin/env python
# coding: utf-8

# This is my **first kaggle notebook** and I think it is very suitable for **beginners** as my code is really **basic**.
# I will be grateful to receive some constructive advice that helps me to continue improving analyzing data.
# Thanks for your attention!

# The Chinese empire, through this Debt-Trap is increasing its influence and impact around the world like the Americans did since the begginings of the 20th century.
# This is the new era of modern conquerors

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Imports and load data

# In[ ]:


import numpy as np
import pandas as pd


# Save both CSV in different variables

# In[ ]:


df = pd.read_csv('/kaggle/input/chinese-debt-trap/chinese debt trap all over the world. - projects.csv')
df2 = pd.read_csv("/kaggle/input/chinese-debt-trap/chinese debt trap in Africa (sector wise).csv")


# ## Analyze the data

# In[ ]:


df.head()


# In[ ]:


df2.head()


# It is important to know that the number of data that we are going to use matches.
# Algo be sure to understand the type of the data

# In[ ]:


df.info() #We can obvserve that 'BORROWER' has null data, but since it won't be used it doesn't matter
          #We can also see that price is an object so we will have to deal with this later


# In[ ]:


df2.info()


# # Total money spent in other countries by China each year

# ## Clean data

# Here I created both dataframes with the columns that will be useful to calculate the money spent by China each year

# In[ ]:


perYear = df[["YEAR" ,"AMOUNT"]]
perYear2 = df2[["Year", "$ Allocation"]]


# In[ ]:


perYear.head()


# In[ ]:


#I decided to convert the columns to Numpy arrays to better treat the data
npAmount = np.array(perYear['AMOUNT'])
npAllocation = np.array(perYear2['$ Allocation'])


# Convert object to INT

# In[ ]:


#To convert 'Amount' to INT we need to clean the data
#I made a for loop to iterate over the list created previously and replace the letters and symbols with 0's if needed (Everything is converted to Millions)

lista = []

for i in npAmount:
    if i[-1] == 'B':
        i = i[:-1]
        i = i.replace('.','')
        i = i + '000'
        i = i[1:]
        lista.append(i)
    else:
        i = i[:-1]
        i = i[1:]
        lista.append(i)
intAmount = [int(x) for x in lista]

#Here are the first 5 examples of the result

print(npAmount[:5])
print(intAmount[:5])


# In[ ]:


#Also need to convert YEAR from object to INT

npYear = np.array(df['YEAR'])
list = []
for i in npYear:
    list.append(i)

intYear = [int(x) for x in npYear]

print(intYear)


# In[ ]:


#Create a new dataframe containing the list created before
data = {'YEAR': intYear,
        'AMOUNT': intAmount}

perYearClean = pd.DataFrame(data)
perYearClean


# In[ ]:


#Combine the repeated Year values and sum them to calculate the total spent every year

perYearGrouped = perYearClean.groupby("YEAR").agg(sum)
perYearGrouped


# In[ ]:


#We repeat the procedure with the other CSV file

perYear2.head()


# In[ ]:


npYear2 = np.array(df2['Year'])

intYear2 = [int(x) for x in npYear2]

print(intYear2)


# In[ ]:


lista2 = []

for i in npAllocation:
    if i[-1] == 'B':
        i = i[:-1]
        i = i.replace('.','')
        i = i + '000'
        i = i[1:]
        lista2.append(i)
    else:
        i = i[:-1]
        i = i[1:]
        i = i.replace(',','')
        lista2.append(i)

for i in range(len(lista2)):
    intAllocation = [int(x) for x in lista2]

print(intAllocation)



# In[ ]:


data2 = {'YEAR': intYear2,
        'AMOUNT': intAllocation}

perYearClean2 = pd.DataFrame(data2)
perYearClean2.head()


# In[ ]:


perYearGrouped2 = perYearClean2.groupby("YEAR").agg(sum)
perYearGrouped2


# ## Append dataframes

# In[ ]:


#Combine both cleaned dataframes to get the total money spent each year

perYearCombined = perYearGrouped.append(perYearGrouped2)

perYearCombined["AMOUNT"] = perYearCombined["AMOUNT"].astype(int)
perYearCombinedFinal = perYearCombined.groupby(["YEAR"]).sum()

perYearCombinedFinal


# ## Plot the results

# In[ ]:


#Simple Panda plot

perYearCombinedFinal.plot(kind = 'bar')


# # Money invested by China in other countries
# 
# 
# 

# ## Clean data

# In[ ]:


#Select the columns to use in the new dataframe

datos = [df['Country'], perYearClean['AMOUNT']]
headers = ["COUNTRY", "AMOUNT"]
perCountry = pd.concat(datos, axis=1, keys=headers)
perCountry.head()


# In[ ]:


#Add repeated values to calculate the total for each country

perCountryGrouped = perCountry.groupby("COUNTRY").agg(sum)
perCountryGrouped.head()


# In[ ]:


#Do the same with the other CSV file

datos2 = [df2['Country'], perYearClean2['AMOUNT']]
headers2 = ["COUNTRY", "AMOUNT"]
perCountry2 = pd.concat(datos2, axis=1, keys=headers)
perCountry2.head()


# In[ ]:


perCountryGrouped2 = perCountry2.groupby("COUNTRY").agg(sum)
perCountryGrouped2.head()


# ## Append dataframes

# In[ ]:


perCountryCombined = perCountryGrouped.append(perCountryGrouped2)
perCountryCombined["AMOUNT"] = perCountryCombined["AMOUNT"].astype(int)
perCountryCombinedFinal = perCountryCombined.groupby(["COUNTRY"]).sum()

perCountryCombinedFinal


# ## Plot the results

# In[ ]:


perCountryCombinedFinal.plot(kind = 'bar', width=0.8, figsize=(94,8))

