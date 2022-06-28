#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:LightSkyBlue;">
#     <h1><strong>Books and Wings</strong></h1>
#     This EDA has the goal of finding books with sales potential. Books and Wings is a fictional company created by an fictional investment group. The complete project is available on <a href="https://github.com/Wallis16/Data-Science-Portfolio/tree/main/Books%20%26%20Wings" class="alert-link">Github link</a>.
#     <h4>A famous investment group decided to open an e-commerce company aiming to sell books, and they ask us for recommending books with sale potential. The goal of this project is to collect and analyze data about bestseller books from big companies such as Amazon, eBay, and Barnes&Noble. We intend to use this data to get insights into what would be more relevant for our client.</h4>
# </div>
# 

# <img align="center" width="40%" src="https://github.com/Wallis16/Data-Science-Portfolio/blob/main/Books%20&%20Wings/Figures/pexels-pixabay-159751.jpg?raw=true">

# <img align="center" width="80%" src="https://github.com/Wallis16/Data-Science-Portfolio/blob/main/Books%20&%20Wings/Figures/presentation_image.jpg?raw=true">
# 

# <h1 style="background-color:Yellow;"><strong>Exploratory Data Analysis - Amazon</strong></h1>

# <img align="center" width="30%" src="https://github.com/Wallis16/Data-Science-Portfolio/blob/main/Books%20&%20Wings/Figures/amazon_logo.png?raw=true">

# ### Importing packages

# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


# ### Loading amazon dataset

# In[ ]:


df_book = pd.read_csv("../input/bestsellers-books-amazon-ebay-and-barnesnoble/amazon_products_cleaned.csv")


# ## Data Analysis

# In[ ]:


df_book.info()


# In[ ]:


df_book.describe()


# ##### **Some interesting aspects of bestsellers: are low prices and high ratings.**

# #### Verifying duplicates

# In[ ]:


# duplicated rows
df_book.duplicated().sum()


# #### Verifying null values

# In[ ]:


# rows that contain null values
df_book.shape[0]-df_book.dropna(axis = 0).shape[0]


# #### Analyzing relation among the features

# In[ ]:


df_book.corr()


# ##### **As we can see price practically does not have a relation with a rating or the number of customers who evaluate the book.**

# #### Book cover distribution

# In[ ]:


ax = sns.histplot(data=df_book, x="Book_cover", color="#0728FC")
sns.set(font_scale = 1)

ax.set_xlabel('Book cover', labelpad=15, color='#0728FC')
ax.set_ylabel('Quantity', labelpad=15, color='#0728FC')
ax.set_title('Histogram of book cover distribution', pad=15, color='#0728FC',
             weight='bold')

for p in ax.patches:
    ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='blue', xytext=(0, 5),
                textcoords='offset points')


# #### Price distribution

# In[ ]:


ax = sns.histplot(data=df_book, x="Price", color="#28DC2A", bins=10)
sns.set(font_scale = 1)

ax.set_xlabel('Price', labelpad=15, color='#28DC2A')
ax.set_ylabel('Quantity', labelpad=15, color='#28DC2A')
ax.set_title('Histogram of price distribution', pad=15, color='#28DC2A',
             weight='bold')

for p in ax.patches:
    ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='green', xytext=(0, 5),
                textcoords='offset points')


# #### Rating distribution

# In[ ]:


ax = sns.histplot(data=df_book, x="Rating", color="#FC0707", bins=10)
sns.set(font_scale = 1)

ax.set_xlabel('Rating', labelpad=15, color='#FC0707')
ax.set_ylabel('Quantity', labelpad=15, color='#FC0707')
ax.set_title('Histogram of rating distribution', pad=15, color='#FC0707',
             weight='bold')

for p in ax.patches:
    ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='red', xytext=(0, 5),
                textcoords='offset points')


# #### Authors distribution

# In[ ]:


plt.figure(figsize=(15,10))
ax = sns.histplot(data=df_book, y="Author", color="#E6F31A", bins=10)
sns.set(font_scale = 1)

ax.set_xlabel('Quantity', labelpad=15, color='#E6F31A')
ax.set_ylabel('Author', labelpad=15, color='#E6F31A')
ax.set_title('Histogram of authors distribution', pad=15, color='#E6F31A',
             weight='bold')


# #### Number of ratings by Author 

# In[ ]:


plt.figure(figsize=(15,10))
ax = sns.barplot(y="Author", x="Customers_Rated", data=df_book)

sns.set(font_scale = 1)

ax.set_xlabel('Quantity', labelpad=15, color='#000000')
ax.set_ylabel('Author', labelpad=15, color='#000000')
ax.set_title('Histogram of rating quantity distribution by author', pad=15, color='#000000',
             weight='bold')


# #### Rating and the quantity associated

# In[ ]:


sns.scatterplot(data=df_book, x="Rating", y="Customers_Rated")


# #### Top 10 most evaluated books

# In[ ]:


df_book.sort_values('Customers_Rated', ascending=False)[:10]


# #### **Principal insights:**
# 
# ##### **Most part of bestsellers have hardcover and paperback**
# 
# ##### **The books are not expensive (none of them costs more than 35 dollars)**
# 
# ##### **All the books have high rating (the lowest value is 4.2)**
# 
# ##### **The authors Shannon Bream, Tieghan Gerard, Questions about me (group of authors), and Colleen Hoover have respectively 2, 2, 2, and 4 books among the bestsellers. Colleen Hoover's books have a huge potential to be profitable.**
# 
# ##### **Unfortunately, we do not have data about the number of people that bought these books, however, we have the number of ratings as an interesting indicator of how much people bought that book. Among the top most evaluated authors are Delia Owens (Where the Crawdads Sing), Matt Haig (The Midnight Library: A Novel), and Colleen Hoover (It Ends with Us: A Novel (1)).**

# <h1 style="background-color:Red;"><strong>Exploratory Data Analysis - eBay</strong></h1>

# <img align="center" width="30%" src="https://github.com/Wallis16/Data-Science-Portfolio/blob/main/Books%20&%20Wings/Figures/ebay%20logo.jpg?raw=true">

# In[ ]:


df_book = pd.read_csv("../input/bestsellers-books-amazon-ebay-and-barnesnoble/ebay_products_cleaned.csv")


# In[ ]:


df_book.head()


# In[ ]:


df_book.info()


# In[ ]:


df_book.describe()


# In[ ]:


# duplicated rows
df_book.duplicated().sum()


# In[ ]:


# rows that contain null values
df_book.shape[0]-df_book.dropna(axis = 0).shape[0]


# In[ ]:


df_book.corr()


# In[ ]:


ax = sns.histplot(data=df_book, y="Book_cover", color="#0728FC")
sns.set(font_scale = 1)

ax.set_xlabel('Book cover', labelpad=15, color='#0728FC')
ax.set_ylabel('Quantity', labelpad=15, color='#0728FC')
ax.set_title('Histogram of book cover distribution', pad=15, color='#0728FC',
             weight='bold')


# In[ ]:


ax = sns.histplot(data=df_book, x="price_new", color="#28DC2A", bins=10)
sns.set(font_scale = 1)

ax.set_xlabel('Price', labelpad=15, color='#28DC2A')
ax.set_ylabel('Quantity', labelpad=15, color='#28DC2A')
ax.set_title('Histogram of price distribution', pad=15, color='#28DC2A',
             weight='bold')

for p in ax.patches:
    ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='green', xytext=(0, 5),
                textcoords='offset points')


# In[ ]:


ax = sns.histplot(data=df_book, x="price_used", color="#28DC2A", bins=10)
sns.set(font_scale = 1)

ax.set_xlabel('Price (used books)', labelpad=15, color='#28DC2A')
ax.set_ylabel('Quantity', labelpad=15, color='#28DC2A')
ax.set_title('Histogram of price (used books) distribution', pad=15, color='#28DC2A',
             weight='bold')

for p in ax.patches:
    ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='green', xytext=(0, 5),
                textcoords='offset points')


# In[ ]:


ax = sns.histplot(data=df_book, x="Rating", color="#FC0707", bins=10)
sns.set(font_scale = 1)

ax.set_xlabel('Rating', labelpad=15, color='#FC0707')
ax.set_ylabel('Quantity', labelpad=15, color='#FC0707')
ax.set_title('Histogram of rating distribution', pad=15, color='#FC0707',
             weight='bold')

for p in ax.patches:
    ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='red', xytext=(0, 5),
                textcoords='offset points')


# In[ ]:


plt.figure(figsize=(15,10))
ax = sns.histplot(data=df_book, y="Author", color="#E6F31A", bins=10)
sns.set(font_scale = 1)

ax.set_xlabel('Quantity', labelpad=15, color='#E6F31A')
ax.set_ylabel('Author', labelpad=15, color='#E6F31A')
ax.set_title('Histogram of authors distribution', pad=15, color='#E6F31A',
             weight='bold')


# In[ ]:


plt.figure(figsize=(15,10))
ax = sns.barplot(y="Author", x="Customers_Rated", data=df_book)

sns.set(font_scale = 1)

ax.set_xlabel('Quantity', labelpad=15, color='#000000')
ax.set_ylabel('Author', labelpad=15, color='#000000')
ax.set_title('Histogram of rating quantity distribution by author', pad=15, color='#000000',
             weight='bold')


# In[ ]:


ax = df_book["Year"].value_counts().sort_index().plot.bar()

ax.set_xlabel('Year', labelpad=15, color='#000000')
ax.set_ylabel('Quantity', labelpad=15, color='#000000')
ax.set_title('Histogram of year distribution', pad=15, color='#000000',
             weight='bold')

for p in ax.patches:
    ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='blue', xytext=(0, 5),
                textcoords='offset points')


# In[ ]:


sns.scatterplot(data=df_book, x="Rating", y="Customers_Rated")


# In[ ]:


df_book.sort_values('Customers_Rated', ascending=False)[:10]


# #### **Principal insights:**
# 
# ##### **Most bestsellers have the hardcover, paperback, and trade paperback. "[Trade paperbacks are larger and more durable, and mass-market paperbacks, which are discussed in greater detail below, are smaller and less durable.](https://www.julesbuono.com/paperback-vs-mass-market-paperback/)"**
# 
# ##### **The books are not expensive, however, different from Amazon (none of them costs more than 35 dollars),  now we have a bestseller costing 70 dollars (but if you opt for a used book, this value down by half).**
# 
# ##### **All the books have a high rating (the lowest value is 4.3)**
# 
# ##### **The author Colleen Hoover has 2 books among the bestsellers (only Colleen has more than one bestseller and again is the top author in terms of quantity). Colleen Hoover's books have a huge potential to be profitable.**
# 
# ##### **Unfortunately, we do not have data about the number of people that bought these books, however, we have the number of ratings as an interesting indicator of how much people bought that book. Among the top evaluated authors are George Orwell (1984), Claude Davis (The Lost Book of Herbal Remedies), and Mark Manson (The Subtle Art of Not Giving an F_ck: A Counterintuitive Approach to Living a Good Life).** 

# <h1 style="background-color:LightGreen;"><strong>Exploratory Data Analysis - Barns&Noble</strong></h1>

# <img align="center" width="30%" src="https://github.com/Wallis16/Data-Science-Portfolio/blob/main/Books%20&%20Wings/Figures/Barnes-Noble-Logo.png?raw=true">

# In[ ]:


df_book = pd.read_csv("../input/bestsellers-books-amazon-ebay-and-barnesnoble/barnes_noble_products_cleaned.csv")


# In[ ]:


df_book.head()


# In[ ]:


df_book.info()


# In[ ]:


df_book.describe()


# In[ ]:


# duplicated rows
df_book.duplicated().sum()


# In[ ]:


# rows that contain null values
df_book.shape[0]-df_book.dropna(axis = 0).shape[0]


# In[ ]:


df_book.corr()


# In[ ]:


ax = sns.histplot(data=df_book, x="Book_cover", color="#0728FC")
sns.set(font_scale = 1)

ax.set_xlabel('Book cover', labelpad=15, color='#0728FC')
ax.set_ylabel('Quantity', labelpad=15, color='#0728FC')
ax.set_title('Histogram of book cover distribution', pad=15, color='#0728FC',
             weight='bold')

for p in ax.patches:
    ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='blue', xytext=(0, 5),
                textcoords='offset points')


# In[ ]:


ax = sns.histplot(data=df_book, x="Price", color="#28DC2A", bins=10)
sns.set(font_scale = 1)

ax.set_xlabel('Price', labelpad=15, color='#28DC2A')
ax.set_ylabel('Quantity', labelpad=15, color='#28DC2A')
ax.set_title('Histogram of price distribution', pad=15, color='#28DC2A',
             weight='bold')

for p in ax.patches:
    ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='green', xytext=(0, 5),
                textcoords='offset points')


# In[ ]:


ax = df_book["Year"].value_counts().sort_index().plot.bar()

ax.set_xlabel('Year', labelpad=15, color='#000000')
ax.set_ylabel('Quantity', labelpad=15, color='#000000')
ax.set_title('Histogram of year distribution', pad=15, color='#000000',
             weight='bold')

for p in ax.patches:
    ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')


# In[ ]:


plt.figure(figsize=(15,12))
ax = sns.histplot(data=df_book, y="Author", color="#E6F31A", bins=10)
sns.set(font_scale = 1)

ax.set_xlabel('Quantity', labelpad=15, color='#E6F31A')
ax.set_ylabel('Author', labelpad=15, color='#E6F31A')
ax.set_title('Histogram of authors distribution', pad=15, color='#E6F31A',
             weight='bold')


# #### **Principal insights:**
# 
# ##### **All the bestsellers have two types of bookcover: paperback and hardcover.**
# 
# ##### **The books are not expensive, except by one book that costs more than 60 dollars.**
# 
# ##### **The authors Alice Oseman (2), Shannon Bream (2), Emily Henry (2), E. L. James (2), Tatsuki Fujimoto (2), Julia Quinn (2), Gege Akutami (3), Colleen Hoover (6), and Tatsuya Endo (7) are among authors with more bestsellers. Again, Colleen Hoover is in our top.**

# ## Observations
# 
# #### **Each data collected is different according to the site on which we made the web scraping. In Barnes&Noble, for example, we do not have several evaluations. Therefore, some points could not be compared among the three sites. Our goal in using different sources is to avoid bias and have data more statistically significant. Of course, each site has different considerations about the bestsellers, amazon can update the site hourly and eBay only daily, for example. To check how this can influence our results, we need to know how volatile the books market is and what makes a book a bestseller by these sites. Finishing, we spotlight that the number of ratings on amazon is massively bigger than on eBay. The most number of ratings on eBay has around 400, and amazon's topmost evaluated book has more than 200000 ratings. Again, we need to check more details about this difference.**
#  
# #### **After the discussion above, let's see the insights we would like to share about what books can be more profitable.** 
# 
# ### **Insights:**
# 
# ##### **The books are cheap.** 
# ##### **People prefer hardcover, paperback, and trade paperback.**
# ##### **Colleen Hoover is the author of the moment. She has more than one book considered a bestseller by each one of the sites analyzed.**
# ##### **Beyond Colleen Hoover, authors who received a great number of ratings are also interesting.**
# ##### **In eBay and Barnes&Noble we can see that most bestsellers are recent books.**
# ##### **Amazon and eBay also show that bestsellers have high ratings (no less than 4 stars).**

# # **Thank you!!!**
