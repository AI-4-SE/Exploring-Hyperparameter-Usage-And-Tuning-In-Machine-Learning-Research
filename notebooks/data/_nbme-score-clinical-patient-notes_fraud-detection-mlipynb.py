#!/usr/bin/env python
# coding: utf-8

# > >  > We will going to create the notebook which will show us how to use the *machine learning* techniques on the banking dataset and find out the **fraud detection methods** , with the help of *logistic model*.

# > > We will be using the banking dataset and building the model aroung it.

# > **Exploratory Data Analysis in Python**

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


#loading the dataset into the environment
data1 = pd.read_csv("../input/creditcardfraud/creditcard.csv")
#finding the head of the dataset
data1.head()


# In[ ]:


data1.shape #finding the shape of the dataset


# In[ ]:


print(data1.describe())


# In[ ]:


data1.dtypes #finding out the types of the different features and input in our dataset


# In[ ]:


#finding out the number of missing values in the dataset
data1.count()


# In[ ]:


data1.info


# In[ ]:


#finding out the duplicate values in the dataset
data1.duplicated()


# In[ ]:


#plotting the label of the dataset
data1.Class.plot(kind='line') #it kind of provides us the idea that how the data is classified in our dataset


# In[ ]:


data1.isna() #finding out the missing values in the dataset
#as we can see there are no missing values in the dataeset


# In[ ]:


data1.isna().sum()#this is the more good form of the above function


# # **Class imbalance problem**

# In[ ]:


#lets vizualize some more of the dataset
data1.Class.hist()
plt.show()
#as we can see there is very few number of people who are classified as fraud by the system
#this can also be seen as the class imbalance problem
#going ahead we need to use the SMOTE methods to overcome this problem


# In[ ]:


corr = data1.corr()
corr.style.background_gradient(cmap='coolwarm')
#as we can see that there is no large correlation present in the dataset , that we need to be taken care off.


# In[ ]:


#it is showing the zero values because there are many values which are negatively correalted to each other 
cor_matrix = data1.corr().abs()
print(cor_matrix.head())


# In[ ]:


#we will be calcualting the correlations matrix with different correlation coefficients,
#spearman is calculated for the monotonic dataset, not for the linear dataset
Spearman_1 = data1.corr(method="spearman")
print(Spearman_1.head())


# In[ ]:


#so the good thing about it is that, the dataset contains the Numerically encoded variables V1 to V28 which are the principal components obtained from a PCA transformation. 
#Due to confidentiality issues, no background information about the original features was provided.


# In[ ]:


#lets find the number of frauds in our dataset
data1['Class'].value_counts()


# In[ ]:


#Let's calculate the percentage of fraudulent transactions over the total number of 
#transactions in our dataset
(data1['Class'].value_counts()*100/len(data1)).convert_dtypes()
#as we can see the percentage of fruadulent transaction in dataset is less than 1/2%


# In[ ]:


#next we will going to plot the dataset
def prep_data(df):
    X = df.iloc[:, 1:28]
    X = np.array(X).astype(float)
    y = df.iloc[:, 29]
    y = np.array(y).astype(float)
    return X, y

def plot_data(X, y):
    plt.scatter(X[y==0, 0], X[y==0, 1], label='Class #0', alpha=0.5, linewidth=0.15)
    plt.scatter(X[y==1, 0], X[y==1, 1], label='Class #1', alpha=0.5, linewidth=0.15, c='r')
    plt.legend()
    return plt.show()


# In[ ]:


#we will going to pass the values of the dataset from the funciton that we have just made
X, y = prep_data(data1)


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


plot_data(X,y)


# In[ ]:


#we need to rebalance the data by using the SMOTE to rebalance the dataset
#from imblearn.over_sampling import SMOTE

#method = SMOTE()
#X_resampled, y_resampled = method.fit_resample(X, y)


# In[ ]:


#plot_data(X_resampled,y_resampled)

