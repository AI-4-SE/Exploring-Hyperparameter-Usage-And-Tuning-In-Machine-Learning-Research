#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from prophet import Prophet


# Read the csv with pandas

# In[ ]:


df = pd.read_csv('../input/madrid-weather-dataset-by-hours-20192022/weather_madrid_2019-2022.csv',  parse_dates=['time'])  
df = df.rename(columns={'time': 'ds'})
df = df.rename(columns={'temperature': 'y'})

df = df.drop(['Unnamed: 0'], axis=1)
df['ds'] = df['ds'].dt.tz_localize(None)


# Visualize the data

# In[ ]:


df


# In[ ]:


df.head()
plt.xlabel('')
plt.ylabel('temperature')
plt.plot(df['y'])


# Train test split

# In[ ]:


train = df[(df['ds'] < '2022-01-29')]
test = df[(df['ds'] > '2022-01-30')]


# Model definition

# In[ ]:


m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=True)


# Add the regresor and mode

# In[ ]:


m.add_regressor('wind_speed', prior_scale=1, mode='additive')
m.add_regressor('wind_direction', prior_scale=0.5, mode='additive')
m.add_regressor('humidity', prior_scale=1, mode='additive')
m.add_regressor('solar_radiation', prior_scale=1, mode='additive')
m.add_regressor('precipitation', prior_scale=0.7, mode='additive')


# Fit the multivariable model

# In[ ]:


m.fit(train)


# In[ ]:


future = m.make_future_dataframe(periods=72, freq='H')
final_df = future
final_df


# For predict with a multivariable model we need to predict the regresors with a single variable model, with this loop we iterate through all the columns, generate and save the forecast 

# In[ ]:


list = ['y',
'wind_speed',
'wind_direction',
'humidity',
'barometric_pressure',
'solar_radiation',
'precipitation']
df_list = [] 


for col in list:
    df_aux = train[['ds', col]]
    df_aux = df_aux.rename(columns={col: 'y'})
    m1 = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=True)
    m1.fit(df_aux)
    forecast = m1.predict(future)
    final_df[col] = forecast['yhat']
    


# In[ ]:


final_df


# Now we predict with the multivariable model

# In[ ]:


forecast = m.predict(final_df)


# In[ ]:


forecast = forecast[(forecast['ds'] > '2022-01-30')]


test = test.set_index('ds')
forecast = forecast.set_index('ds')


# Visualize our forecast vs the test

# In[ ]:


test['y'].plot(figsize = (12, 5), legend = True)
forecast['yhat'].plot(legend = True)


# With prophet.diagnostics we can visualize the rmse and mae.

# In[ ]:


from prophet.diagnostics import cross_validation
df_cv = cross_validation(m, initial='1000 days', period='30 days', horizon = '2 days')


# In[ ]:


from prophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p.head()


# In[ ]:


from prophet.plot import plot_cross_validation_metric

fig = plot_cross_validation_metric(df_cv, metric='rmse')


# In[ ]:


fig = plot_cross_validation_metric(df_cv, metric='mae')


# In[ ]:


from prophet.utilities import regressor_coefficients

regressor_coefficients(m)


# In[ ]:




