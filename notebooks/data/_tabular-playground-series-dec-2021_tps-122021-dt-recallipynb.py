#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
data=pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
data.head()


# In[ ]:


data.info()


# In[ ]:


Wilderness_Area_cols = [col for col in data.columns if 'Wilderness_Area' in col]
Wilderness_Area_cols


# In[ ]:


Wilderness_Area=data[Wilderness_Area_cols]
Wilderness_Area.head()


# In[ ]:


df_n = Wilderness_Area.apply(lambda x: x.idxmax(), axis = 1)
df_n.head()


# In[ ]:


Wilderness_Area=pd.DataFrame(df_n, columns=['Wilderness'])
Wilderness_Area.head()


# In[ ]:


Wilderness_Area.Wilderness.value_counts()


# In[ ]:


Wilderness_Area.Wilderness=Wilderness_Area.Wilderness.map({'Wilderness_Area1':'Area1',
                                                           'Wilderness_Area2':'Area2',
                                                          'Wilderness_Area3':'Area3',
                                                          'Wilderness_Area4':'Area4'})
Wilderness_Area.Wilderness.value_counts()


# In[ ]:


data=data.drop(['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'],axis=1)
data.head()


# In[ ]:


data=pd.concat([data, Wilderness_Area], axis=1)
data.head()


# In[ ]:


Soil_Type_cols = [col for col in data.columns if 'Soil_Type' in col]
Soil_Type_cols


# In[ ]:


Soil_Type=data[Soil_Type_cols]
Soil_Type.head()


# In[ ]:


df_n = Soil_Type.apply(lambda x: x.idxmax(), axis = 1)
df_n.head()


# In[ ]:


Soil_Type=pd.DataFrame(df_n, columns=['Soil'])
Soil_Type.head()


# In[ ]:


Soil_Type.Soil.value_counts()


# In[ ]:


Soil_Type.Soil = Soil_Type.Soil.str.replace('Soil_', '')
Soil_Type.Soil.value_counts()


# In[ ]:


data=data.drop(Soil_Type_cols,axis=1)
data.head()


# In[ ]:


data=pd.concat([data, Soil_Type], axis=1)
data.head()


# In[ ]:


data.info()


# In[ ]:


data.nunique()


# In[ ]:


round(data.Cover_Type.value_counts()*100/len(data),2)


# In[ ]:


data.Cover_Type.value_counts()


# In[ ]:


data=data[data.Cover_Type!=5]
data=data[data.Cover_Type!=4]
data=data[data.Cover_Type!=6]
data=data[data.Cover_Type!=7]
data=data.sample(n=250000)
data.reset_index(drop=True, inplace=True)
data.shape


# In[ ]:


round(data.Cover_Type.value_counts()*100/len(data),2)


# In[ ]:


data.Cover_Type=data.Cover_Type.map({1:0,2:1,3:2})
round(data.Cover_Type.value_counts()*100/len(data),2)


# In[ ]:


data.Cover_Type.unique()


# In[ ]:


pd.options.display.float_format = '{:.2f}'.format
data.describe()


# In[ ]:


data.nunique().sort_values(ascending=False)


# In[ ]:


get_ipython().system('pip install pycaret[full]')


# In[ ]:


from pycaret.classification import *


# In[ ]:


forest= setup(data=data,
          target = "Cover_Type",  session_id=42,
          normalize=True,
          train_size = 0.8, # training over 80% of available data
          handle_unknown_categorical = True, 
          remove_multicollinearity = True, #drop one of the two features that are highly correlated with each other
          ignore_low_variance = True,#all categorical features with statistically insignificant variances are removed from the dataset.    
          ignore_features=['Id'],
          categorical_features=['Wilderness'],
              high_cardinality_features=['Soil'],
          combine_rare_levels = True,
          fix_imbalance = True,
          unknown_categorical_method= 'most_frequent',
          transformation = True,silent=True
         )


# In[ ]:


model_dt_Recall = tune_model(create_model('dt'),optimize = 'Recall')
model_dt_Recall 


# In[ ]:


plot_model(model_dt_Recall,plot = 'confusion_matrix',use_train_data=True)


# In[ ]:


plot_model(model_dt_Recall,plot = 'confusion_matrix')


# In[ ]:


plot_model(model_dt_Recall,plot = 'class_report',use_train_data=True)


# In[ ]:


plot_model(model_dt_Recall,plot = 'class_report')


# In[ ]:


predict_model(model_dt_Recall)


# In[ ]:


final_dt = finalize_model(model_dt_Recall)
final_dt


# In[ ]:


test=pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")
test.head()


# In[ ]:


Wilderness_Area_cols = [col for col in test.columns if 'Wilderness_Area' in col]
Wilderness_Area_cols


# In[ ]:


Wilderness_Area=test[Wilderness_Area_cols]
Wilderness_Area.head()


# In[ ]:


df_n = Wilderness_Area.apply(lambda x: x.idxmax(), axis = 1)
df_n.head()


# In[ ]:


Wilderness_Area=pd.DataFrame(df_n, columns=['Wilderness'])
Wilderness_Area.head()


# In[ ]:


Wilderness_Area.Wilderness.value_counts()


# In[ ]:


Wilderness_Area.Wilderness=Wilderness_Area.Wilderness.map({'Wilderness_Area1':'Area1',
                                                           'Wilderness_Area2':'Area2',
                                                          'Wilderness_Area3':'Area3',
                                                          'Wilderness_Area4':'Area4'})
Wilderness_Area.Wilderness.value_counts()


# In[ ]:


test=test.drop(['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'],axis=1)
test.head()


# In[ ]:


test=pd.concat([test, Wilderness_Area], axis=1)
test.head()


# In[ ]:


Soil_Type_cols = [col for col in test.columns if 'Soil_Type' in col]
Soil_Type_cols


# In[ ]:


Soil_Type=test[Soil_Type_cols]
Soil_Type.head()


# In[ ]:


df_n = Soil_Type.apply(lambda x: x.idxmax(), axis = 1)
df_n.head()


# In[ ]:


Soil_Type=pd.DataFrame(df_n, columns=['Soil'])
Soil_Type.head()


# In[ ]:


Soil_Type.Soil = Soil_Type.Soil.str.replace('Soil_', '')
Soil_Type.Soil.value_counts()


# In[ ]:


test=test.drop(Soil_Type_cols,axis=1)
test.head()


# In[ ]:


test=pd.concat([test, Soil_Type], axis=1)
test.head()


# In[ ]:


predictions=predict_model(final_dt,data =test)
predictions.head()


# In[ ]:


predictions=predictions[['Id','Label']]
predictions.head()


# In[ ]:


predictions.Label.value_counts()


# In[ ]:


predictions.Label=predictions.Label.map({0:1,1:2,2:3})
predictions.Label.value_counts()


# In[ ]:


predictions.columns = ['Id', 'Cover_Type']
predictions.head()


# In[ ]:


predictions.to_csv('./predictions.csv',index=False)

