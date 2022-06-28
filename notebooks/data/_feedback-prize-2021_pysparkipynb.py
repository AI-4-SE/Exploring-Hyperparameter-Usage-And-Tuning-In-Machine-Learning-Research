#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


from pyspark.sql import SparkSession 


# In[ ]:


spark = SparkSession.builder.master("local[*]").appName("example_spark").getOrCreate()


# In[ ]:


data = [(1,'virat','kohli'),(2,'rohit','sharma'),(3,'jasprit','bumrah')]
headers = ('Rank', 'Name', 'Surmane')
df = spark.createDataFrame(data,headers)
df.show()


# In[ ]:


# df2 = spark.read.format('com.crelytics.spark.excel') \
# .option('header','true') \
# .option('inferschema','true') \
# .load('../input/date-fruit-datasets/Date_Fruit_Datasets/Date_Fruit_Datasets.xlsx')
# df2.show()

# import pyspark
# df2 = pyspark.pandas.read_excel('../input/date-fruit-datasets/Date_Fruit_Datasets/Date_Fruit_Datasets.xlsx')
# df2.show()


# In[ ]:


import pandas as pd


# In[ ]:


get_ipython().system('pip install openpyxl')


# In[ ]:


import openpyxl


# In[ ]:


df2 = pd.read_excel('../input/date-fruit-datasets/Date_Fruit_Datasets/Date_Fruit_Datasets.xlsx',sheet_name=0)
df2.head()


# In[ ]:


df3 = spark.createDataFrame(df2)
df3.show()


# In[ ]:


df3.columns


# In[ ]:


df2['Class'].value_counts()


# In[ ]:


df3.groupBy('class').sum('AREA').show()


# In[ ]:


df4 = df3.groupBy('Class').agg({'AREA':'sum','PERIMETER':'max'})
df4.show()


# In[ ]:


df5 = df4.withColumnRenamed('sum(AREA)','sum_area').withColumnRenamed('max(PERIMETER)','max_perimeter')
df5.show()
# ['Class','sum_area','max_perimeter']


# In[ ]:


df5.show()


# In[ ]:


df5.write.parquet('./output_df.parquet')


# In[ ]:




