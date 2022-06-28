#!/usr/bin/env python
# coding: utf-8

# **Disclaimer:** In this notebook I am going to merge all the available datasets to perform the analysis at the following levels:
# 
# * customer
# * product
# * seller
# * payments
# * freight time

# ### Listing available datasets

# In[ ]:


get_ipython().system('ls -ltr /kaggle/input/brazilian-ecommerce/*.csv')


# In[ ]:


ROOT_PATH = "/kaggle/input/brazilian-ecommerce"


# ## a. Pandas

# ![Pandas Logo](https://numfocus.org/wp-content/uploads/2016/07/pandas-logo-300.png)

# In[ ]:


import pandas as pd

pd.set_option("display.max_rows", 50, "display.max_columns", 50)


# ### 1. Reading the datasets

# In[ ]:


customer_order_df_pd = pd.read_csv(f"{ROOT_PATH}/olist_customers_dataset.csv")

order_df_pd = pd.read_csv(f"{ROOT_PATH}/olist_orders_dataset.csv",
                       parse_dates=[
                           "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                           "order_delivered_customer_date", "order_estimated_delivery_date"
                       ])

order_items_df_pd = pd.read_csv(f"{ROOT_PATH}/olist_order_items_dataset.csv",
                             parse_dates=["shipping_limit_date"])

payments_df_pd = pd.read_csv(f"{ROOT_PATH}/olist_order_payments_dataset.csv")

products_df_pd = pd.read_csv(f"{ROOT_PATH}/olist_products_dataset.csv")

sellers_df_pd = pd.read_csv(f"{ROOT_PATH}/olist_sellers_dataset.csv")

product_category_df_pd = pd.read_csv(f"{ROOT_PATH}/product_category_name_translation.csv")


# ### 2. Viewing the dataframes

# In[ ]:


display(customer_order_df_pd.head())


# In[ ]:


display(order_df_pd.head())


# In[ ]:


display(order_items_df_pd.head())


# In[ ]:


display(payments_df_pd.head())


# In[ ]:


display(products_df_pd.head())


# In[ ]:


display(sellers_df_pd.head())


# In[ ]:


display(product_category_df_pd.head())


# ### 3. Merging the dataframes

# In[ ]:


cust_order_id_df_pd = customer_order_df_pd.merge(order_df_pd, on=["customer_id"], how="inner")
cust_product_id_df_pd = cust_order_id_df_pd.merge(order_items_df_pd, on=["order_id"], how="inner")
cust_payments_df_pd = cust_product_id_df_pd.merge(payments_df_pd, on=["order_id"], how="inner")
cust_products_df_pd = cust_payments_df_pd.merge(products_df_pd, on=["product_id"], how="inner")
cust_sellers_df_pd = cust_products_df_pd.merge(sellers_df_pd, on=["seller_id"], how="inner")
merged_df_pd = cust_sellers_df_pd.merge(product_category_df_pd, on=["product_category_name"], how="inner")


# In[ ]:


display(merged_df_pd.head())


# In[ ]:


# Renaming the columns
merged_df_pd = merged_df_pd.rename(
    columns={
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
        "product_category_name_english": "product_category_name_en"
    })


# In[ ]:


# Subsetting columns
column_order = [
    "customer_id", "customer_unique_id", "order_id", "product_id", "seller_id",
    "order_purchase_timestamp", "order_approved_at","order_estimated_delivery_date", 
    "shipping_limit_date", "order_delivered_carrier_date", "order_delivered_customer_date",
    "customer_zip_code_prefix", "seller_zip_code_prefix",
    "customer_city", "seller_city", "customer_state", "seller_state",
    "order_status", "order_item_id", "product_category_name_en", 
    "product_name_length", "product_description_length", "product_photos_qty",
    "product_length_cm", "product_width_cm", "product_height_cm", "product_weight_g",
    "price", "payment_value", "freight_value",
    "payment_sequential", "payment_type", "payment_installments"
]

merged_df_pd = merged_df_pd[column_order]

display(merged_df_pd.head())


# ## b. PySpark

# ![PySpark Logo](https://databricks.com/wp-content/uploads/2018/12/PySpark-1024x164.png)

# ### Setting up PySpark

# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


get_ipython().system('pip install findspark')


# In[ ]:


import findspark

findspark.init()


# In[ ]:


findspark.find()


# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType, DateType


# In[ ]:


# Creating spark session
spark = SparkSession.builder.appName("olist_ecommerce").getOrCreate()


# ### 1. Reading the datasets

# In[ ]:


customer_order_df = spark.read.format("csv") \
                         .options(header=True, inferSchema=True) \
                         .load(f"{ROOT_PATH}/olist_customers_dataset.csv")

order_df = spark.read.format("csv") \
                .options(header=True, inferSchema=True, timestampFormat="y-M-d H:m:s") \
                .load(f"{ROOT_PATH}/olist_orders_dataset.csv")

order_items_df = spark.read.format("csv") \
                      .options(header=True, inferSchema=True, timestampFormat="y-M-d H:m:s") \
                      .load(f"{ROOT_PATH}/olist_order_items_dataset.csv")

payments_df = spark.read.format("csv") \
                   .options(header=True, inferSchema=True) \
                   .load(f"{ROOT_PATH}/olist_order_payments_dataset.csv")

products_df = spark.read.format("csv") \
                   .options(header=True, inferSchema=True) \
                   .load(f"{ROOT_PATH}/olist_products_dataset.csv")

sellers_df = spark.read.format("csv") \
                  .options(header=True, inferSchema=True) \
                  .load(f"{ROOT_PATH}/olist_sellers_dataset.csv")

product_category_df = spark.read.format("csv") \
                           .options(header=True, inferSchema=True) \
                           .load(f"{ROOT_PATH}/product_category_name_translation.csv")


# In[ ]:


order_df_columns = [column if column != "order_estimated_delivery_date" else F.col(column).cast(DateType()) for column in order_df.columns]

order_df = order_df.select(order_df_columns)


# ### 2. Viewing the dataframes

# In[ ]:


customer_order_df.show(5)


# In[ ]:


order_df.show(5)


# In[ ]:


order_items_df.show(5)


# In[ ]:


payments_df.show(5)


# In[ ]:


products_df.show(5)


# In[ ]:


sellers_df.show(5)


# In[ ]:


product_category_df.show(5)


# ### 3. Merging the dataframes

# In[ ]:


cust_order_id_df = customer_order_df.join(order_df, on=["customer_id"], how="inner")
cust_product_id_df = cust_order_id_df.join(order_items_df, on=["order_id"], how="inner")
cust_payments_df = cust_product_id_df.join(payments_df, on=["order_id"], how="inner")
cust_products_df = cust_payments_df.join(products_df, on=["product_id"], how="inner")
cust_sellers_df = cust_products_df.join(sellers_df, on=["seller_id"], how="inner")
merged_df = cust_sellers_df.join(product_category_df, on=["product_category_name"], how="inner")


# In[ ]:


merged_df = merged_df.select(
    "customer_id", "customer_unique_id", "order_id", "product_id", "seller_id",
    "order_purchase_timestamp", "order_approved_at","order_estimated_delivery_date",
    "shipping_limit_date", "order_delivered_carrier_date", "order_delivered_customer_date",
    "customer_zip_code_prefix", "seller_zip_code_prefix",
    "customer_city", "seller_city", "customer_state", "seller_state",
    "order_status", "order_item_id",
    F.col("product_category_name_english").alias("product_category_name_en"),
    F.col("product_name_lenght").alias("product_name_length"),
    F.col("product_description_lenght").alias("product_description_length"),
    "product_photos_qty", "product_length_cm", "product_width_cm", "product_height_cm", "product_weight_g",
    "price", "payment_value", "freight_value", "payment_sequential", "payment_type", "payment_installments"
)


# In[ ]:


merged_df.show(5)

