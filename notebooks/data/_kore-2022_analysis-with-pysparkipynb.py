#!/usr/bin/env python
# coding: utf-8

# # PySpark - Fundamentos e aplicações
# 
# A ideia deste *notebook* é apresentar os fundamentos de uso da API *Python* para *Spark*. Vamos verificar os principais comandos, e entender o que são ações, transformações, execução *lazy* e as famosas DAG's geradas pelo *Spark* (essa última parte de uma forma mais superficial).
# 
# Tudo isso será feito dentro de um ambiente *Databricks*, o qual conta com algumas funcionalidades que facilitam nosso trabalho diário.
# 
# Após essa breve introdução e explanação dos conceitos, nós iremos simular uma utilização em um problema de negócio, para facilitar a assimilação de tudo que foi visto.
# 
# ## Pontos importantes sobre essa interface
# 
# A aba na lateral direita nos permite:
# 
# - criar *notebooks*, a partir da opção *create*;
# - importar dados, também na opção *create*. Essa importação é feita a partir de uma interface gráfica, muito simples de ser utilizada (vale a pena o teste);
# - criação de um *cluster* na opção *compute*. Todo *notebook* deve ser associado a um *cluster* para execução.
# 
# ## Dados utilizados
# 
# Os dados utilizados estão dispobilizados no *Kaggle*:
# 
# - Link para *download*: [Conjunto de dados](https://www.kaggle.com/olistbr/brazilian-ecommerce?select=olist_order_payments_dataset.csv)
# 
# Nesse projeto vamos utilizar as tabelas:
# 
# - *olist_customers_dataset.csv*;
# - *olist_order_payments_dataset.csv*;
# - *olist_orders_dataset.csv*.
# 
# ### Contexto dos dados
# 
# "*This dataset was generously provided by Olist, the largest department store in Brazilian marketplaces. Olist connects small businesses from all over Brazil to channels without hassle and with a single contract. Those merchants are able to sell their products through the Olist Store and ship them directly to the customers using Olist logistics partners. After a customer purchases the product from Olist Store a seller gets notified to fulfill that order. Once the customer receives the product, or the estimated delivery date is due, the customer gets a satisfaction survey by email where he can give a note for the purchase experience and write down some comments.*"
# 
# #### Link entre os dados
# 
# ![databases](https://i.imgur.com/HRhd2Y0.png)
# 
# ## Referências úteis
# 
# - Documentação: [Documentação PySpark](https://spark.apache.org/docs/latest/api/python/index.html);
# - Criando tabelas: [Criação de tabelas no databricks](https://docs.databricks.com/data/tables.html).

# Mãos a obra
# Agora que temos um conhecimento sobre os principais métodos dessa API, vamos aplicá-los para resolvermos alguns problemas de negócio.A área de negócio nos fez as seguintes indagações:
# 
# 1 - Quantidade de ordens agrupadas por ANO/MÊS/STATUS (eles precisam de um arquivo .CSV contendo todas essas informações);
# 
# 2 - Quantidade de usuários por estado (para identificarem onde precisam focar os esforços de marketing);
# 
# 3 - Além da quantidade, a área de negócio precisa do ranking de cada estado (qual é o primeiro, segundo, etc) em termos da quantidade de usuários;
# 
# 4 - Quantidade de usuários que tiveram mais de três ordens;
# 
# 5 - Dos usuários que tiveram pelo menos três ordens, quantos dias isso (ter a terceira ordem) levou em relação a primeira ordem de compra;Devemos saber o seguinte sobre o conjunto de dados:
# - customer_id: ID do cliente;
# - customer_unique_id: ID único de cada cliente (esse ID engloba vários customer_id);
# - order_purchase_timestamp: timestamp da data da ordem;
# - customer_state: estado do cliente;payment_type: tipo de pagamento;
# 
# Endereço das BasesUtilizar bases:
# - olist_customers_dataset.csv
# - olist_orders_dataset.csv
# - olist_order_payments_dataset.csv

# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


import pyspark.sql.types as T         #Define os tipos nativos do PySpark
import pyspark.sql.functions as F     #Importa as funções nativas do Spark para manipulação dos dados
from pyspark.sql.window import Window #Importa a função utilizada para criação de janelas

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Analysis with pyspark").getOrCreate()


# In[ ]:


customers_df = spark.read.option("delimeter","|").csv('../input/brazilian-ecommerce/olist_customers_dataset.csv',header = True)
customers_df = customers_df.na.drop()
customers_df.printSchema()
customers_df.show()


# In[ ]:


orders_df = spark.read.option("delimeter","|").csv('../input/brazilian-ecommerce/olist_orders_dataset.csv',header = True)
orders_df = orders_df.withColumn('order_purchase_timestamp', F.to_timestamp('order_purchase_timestamp')).withColumn('order_delivered_carrier_data', F.to_timestamp('order_delivered_carrier_date')).withColumn('order_approved_at',F.to_timestamp('order_approved_at')).withColumn('order_delivered_customer_date',F.to_timestamp('order_delivered_customer_date')).withColumn('order_estimed_delivery_date',F.to_timestamp('order_estimated_delivery_date'))
orders_df.printSchema()
orders_df.show()


# In[ ]:


payment_df = spark.read.option("delimeter","|").csv('../input/brazilian-ecommerce/olist_order_payments_dataset.csv',header = True)
payment_df = payment_df.na.drop()
payment_df.printSchema()
payment_df.show()


# 1 - Quantidade de ordens agrupadas por ANO/MÊS/STATUS (eles precisam de um arquivo .CSV contendo todas essas informações)

# In[ ]:


orders_df = orders_df.withColumn("order_year_month", F.date_format(F.col("order_purchase_timestamp"), format="y-M"))
answer_1 = orders_df.groupby("order_year_month", "order_status").count().withColumnRenamed('count','No_of_orders_year_month')
answer_1.show(3)


# 2 - Quantidade de usuários por estado (para identificarem onde precisam focar os esforços de marketing);

# In[ ]:


answer_2 = customers_df.groupby("customer_state").count().withColumnRenamed('count','No_of_customers_state')
answer_2.show(2)


# 3 - Além da quantidade, a área de negócio precisa do ranking de cada estado (qual é o primeiro, segundo, etc) em termos da quantidade de usuários;

# In[ ]:


answer_3 = answer_2.orderBy('No_of_customers_state',ascending=False)
answer_3 = answer_3.withColumn('rank',F.monotonically_increasing_id()+1)
answer_3.show()


# 4 - Quantidade de usuários que tiveram mais de três ordens;

# In[ ]:


orders_customers = orders_df.join(customers_df, on="customer_id", how="left")
answer_4 = (orders_customers.groupby("customer_unique_id").count().where(F.col("count") >= 3))

print(f"Quantidade de clientes: {answer_4.count()}.")


# https://www.geeksforgeeks.org/groupby-and-filter-data-in-pyspark/

# 5 - Dos usuários que tiveram pelo menos três ordens, quantos dias isso (ter a terceira ordem) levou em relação a primeira ordem de compra;

# In[ ]:




