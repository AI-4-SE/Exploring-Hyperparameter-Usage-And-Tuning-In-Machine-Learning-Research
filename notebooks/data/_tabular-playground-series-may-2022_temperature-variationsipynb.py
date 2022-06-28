#!/usr/bin/env python
# coding: utf-8

# The target variable in this competition is the so called '**Indoor_temperature_room**'. This is the indoor temperature in the bedroom situated at the south-west corner of the house. We are required to predict a time series with 15 minutes increment. Thank you very much Abir for posting this competition !
# 

# # Indoor Temperature room

# In[ ]:


import pandas as pd
df=pd.read_csv('../input/smart-homes-temperature-time-series-forecasting/train.csv')
df.dtypes


# In[ ]:


df['Indoor_temperature_room'].describe()


# In[ ]:


import matplotlib.pyplot as plt
df.plot(x='Date',y='Indoor_temperature_room')
plt.xticks(rotation = 70)
plt.show()


# The room temperature daily cycle is the most distinct pattern: cold during the night and warm during the day.
# During the training period, the coldest temperature was 11.076 and the warmest 24.944.
# There is an additional pattern though, with a multi-day period, which seems to be caused by the weather. Let's have a closer look.

# In[ ]:


df['MA'] = df['Indoor_temperature_room'].rolling(window=192).mean()
df.plot(x='Date',y='MA',label='Room Temperature 2 days moving avg')
plt.xticks(rotation = 70)
plt.show()


# The 2 days moving average graph smooths the day/night variations and exhibits the weather factor. Let's now graph the weather, specifically the irradiance and rain:

# In[ ]:


df.plot(x='Date',y='Meteo_Sun_irradiance')
plt.xticks(rotation = 70)
plt.show()


# In[ ]:


df.plot(x='Date',y='Meteo_Rain')
plt.xticks(rotation = 70)
plt.show()


# The rain and clouds (low irradiance) are associated with the drop in the moving average room temperature.
# What about the correlations ?

# In[ ]:


import seaborn as sns
df_extract = df[['Indoor_temperature_room','Meteo_Sun_irradiance','Meteo_Rain']]
corrMatrix = df_extract.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# The room temperature is positively correlated with the sun irradiance and negatively with the rain. However the relations are weak. Can we find something better ? We can try the temperature variations instead.

# # Room temperature variations

# In[ ]:


df['room_temp_variation'] = df['Indoor_temperature_room'].diff()
df.plot(x='Date',y='room_temp_variation',label='Room Temperature variations')
plt.xticks(rotation = 70)
plt.show()


# Again, the 24 hour cycle appears clearly.

# In[ ]:


sns.displot(df,x='room_temp_variation',kde=True)
plt.show()


# In[ ]:


from scipy.stats import skew

print('Temperature change Average: {:.5f}, Skew: {:.2f} '.format(df['room_temp_variation'].mean() ,df['room_temp_variation'].skew()))
print ('Temperature going up, count ratio: {:.2f}'.format(len(df[df['room_temp_variation']>0])/len(df)))


# 2/3 of the time, the temperature decreases but with smaller moves than when it increases. In total the average is close to zero.

# In[ ]:


df_extract = df[['room_temp_variation','Meteo_Sun_irradiance','Meteo_Rain']]
corrMatrix = df_extract.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# The room temperature variation is very strongly correlated with the sun irradiance. It is slightly negatively correlated with the rain.
# Let's have a closer look at the room temperature variation / sun irradiance relationship.

# In[ ]:


plt.figure(figsize = (20,15))
g=sns.scatterplot(data=df,x='Meteo_Sun_irradiance',y='room_temp_variation')
g.set(ylim=(-0.5,0.75))
plt.show()


# There are two interesting parts in this graph.
# When the sun irradiance is positive (during the day), the temperature variation follows a linear relation with the light received. The slope is actually the Heat Capacity of the house. The light transfers heat to the house and specifically the room and increases the temperature.
# When the sun irradiance is null (during the night), the house heat is transferred to the exterior which is colder and the room temperature variations are negative.

# # Stationarity

# As the temperature variation seems to have a stronger physical meaning than the temperature itself, we may be tempted to predict the variation instead. The usual test to determine this question is the stationarity of the time series. Let's draw it again. 

# In[ ]:


df.plot(x='Date',y='Indoor_temperature_room')
plt.xticks(rotation = 70)
plt.show()


# We don't really see a long term trend here. There is about a month of observations, in this period, the weather fluctuations are stronger than the seasonal trend. This would mean that in the train data window, the time series is stationary. Let's confirm this with an Augmented Dickey-Fuller test.

# In[ ]:


from statsmodels.tsa.stattools import adfuller

df=pd.read_csv('../input/smart-homes-temperature-time-series-forecasting/train.csv')
result = adfuller(df['Indoor_temperature_room'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# The p-value is well below 0.05 which indicates a stationary time series. 
