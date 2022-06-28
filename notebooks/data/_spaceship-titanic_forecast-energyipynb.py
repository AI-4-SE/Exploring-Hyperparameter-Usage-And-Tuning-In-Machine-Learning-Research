#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import seaborn as sns



# ## Data imports

# In[ ]:


train=pd.read_csv('../input/rte-forecast-energy-consumption-in-french-areas/train.csv',index_col='date',parse_dates=True)
test=pd.read_csv('../input/rte-forecast-energy-consumption-in-french-areas/test.csv',index_col='date',parse_dates=True)
temp=pd.read_csv('../input/rte-forecast-energy-consumption-in-french-areas/weather_power_nasa.csv',index_col='date',parse_dates=True)
piv=pd.pivot_table(train, values='energy_consumption', index='date',columns='metropolitan_area_code')


# In[ ]:


piv.resample('W').sum().describe()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


merged=pd.merge(train['energy_consumption'].resample('W').sum(),temp['T2M'].resample('d').mean(),on='date')


# In[ ]:


fig,ax=plt.subplots(figsize=(20,5))
ax.set(xlabel='time',ylabel='energy',title='energy consumption over time')




plt.grid()
plt.plot(train['energy_consumption'].resample('W').sum())

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

date_form=DateFormatter('%Y-%b')
ax.xaxis.set_major_formatter(date_form)
plt.xticks(rotation = 45)


plt.show()


# In[ ]:


fig,ax=plt.subplots(figsize=(20,5))
ax.set(xlabel='time',ylabel='temp',title='t2m temperature')


plt.grid()
plt.plot(merged['T2M'])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

date_form=DateFormatter('%Y-%b')
ax.xaxis.set_major_formatter(date_form)
plt.xticks(rotation = 45)


plt.show()


# In[ ]:


train['energy_consumption'].resample('W').agg(['min','mean','max']).plot(subplots=True,figsize=(10,10))
plt.show()


# ## Seasonal decomposition

# In[ ]:


import statsmodels.api as sm


# In[ ]:


decomposition=sm.tsa.seasonal_decompose(train['energy_consumption'].resample('W').sum(),model='additive')


# ## Trend

# In[ ]:


figtrend,ax=plt.subplots(figsize=(20,7))

plt.grid()
ax.set(xlabel='time',ylabel='energy',title=' trend of energy consumption over time')

plt.plot(train.resample('W')['energy_consumption'].sum(),c='blue')
plt.plot(decomposition.trend.index, decomposition.trend, c='red')

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

date_form=DateFormatter('%Y-%b')
ax.xaxis.set_major_formatter(date_form)
plt.xticks(rotation = 45)



plt.legend(["energy consumption","trend of energy consumption"])
plt.show()
figtrend.savefig('pouet')



# In[ ]:


figseason,ax=plt.subplots(figsize=(20,7))
plt.grid()
ax.set(xlabel='time',ylabel='energy',title=' seasonal component of energy consumption')

plt.plot(train.resample('W')['energy_consumption'].sum(),c='blue')
plt.plot(decomposition.seasonal.index, decomposition.seasonal, c='red')

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))


date_form=DateFormatter('%Y-%b')
ax.xaxis.set_major_formatter(date_form)
plt.xticks(rotation = 45)

plt.legend(["energy consumption","seasonal component"])
plt.show()



# ## seasonal component over one year

# In[ ]:


figseason,ax=plt.subplots(figsize=(20,7))
plt.grid()
ax.set(xlabel='time',ylabel='energy',title=' seasonal component of energy consumption')


plt.plot(decomposition.seasonal.loc['2020'].index, decomposition.seasonal.loc['2020'], c='red')

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))


date_form=DateFormatter('%Y-%b')
ax.xaxis.set_major_formatter(date_form)
plt.xticks(rotation = 45)

plt.legend(["energy consumption","seasonal component"])
plt.show()


# **we can see that we obtained one 1-year seasonal component, energy consumption  is at his highest in december and january**

# ## Predictions

# In[ ]:


from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# **Testing holt winters**

# In[ ]:


X=train.resample('W')['energy_consumption'].sum()

X_train=X['2018':'2021-01-03']
X_train2=X['2021-01-03':]

model = ExponentialSmoothing(X_train, trend='add',seasonal='add').fit()
pred = model.predict(start=X_train2.index[0], end=X_train2.index[-1])

predfig=plt.figure(figsize=(15,7))
plt.plot(X_train.index, X_train, label='Train')
plt.plot(X_train2.index, X_train2, label='Test')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best')
plt.show()


# ## SARIMA

# In[ ]:


X=train['energy_consumption'].resample('w').sum()


# In[ ]:


df = pd.DataFrame({
    'energy': train['energy_consumption'].resample('w').sum()    
})
#df.index = pd.DatetimeIndex(df.index).to_period('w')


# In[ ]:


pip install pmdarima;


# In[ ]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import pacf


# ## autocorrelation

# **ploting autocorr**

# In[ ]:


autocorrfig=plt.figure(figsize=(15,10))
# Adding plot title.
plt.title("Autocorrelation Plot")
 
# Providing x-axis name.
plt.xlabel("Lags")
 

plt.subplot(2,1,1)
plt.grid()
plt.title('acorr of diff(1)')
plt.acorr(df['energy'].loc['2018':], maxlags = 52)
plt.xlabel('lags')
plt.ylabel('autocorrelation')

plt.subplot(2,1,2)
plt.grid()
plt.title('diff(1)')
plt.plot(df['energy'].loc['2018':])
plt.xlabel('time')
plt.ylabel('diff(1) energy')


fig.tight_layout(pad=10.0)
 
# Displaying the plot.
print("The Autocorrelation plot for the data is:")

 
plt.show()


# **differencing d=1**

# In[ ]:


autocorrfig=plt.figure(figsize=(15,10))
# Adding plot title.
plt.title("Autocorrelation Plot")
 
# Providing x-axis name.
plt.xlabel("Lags")
 
# Plotting the Autocorrelation plot.
#plt.acorr(df['energy'].loc['2018':], maxlags = 52)
#plot_acf(df['energy'].loc['2018':])

plt.subplot(2,1,1)
plt.grid()
plt.title('acorr of diff(1)')
plt.acorr(df['energy'].loc['2018':].diff(1)[1:], maxlags = 52)
plt.xlabel('lags')
plt.ylabel('autocorrelation')

plt.subplot(2,1,2)
plt.grid()
plt.title('diff(1)')
plt.plot(df['energy'].loc['2018':].diff(1)[1:])
plt.xlabel('time')
plt.ylabel('diff(1) energy')


fig.tight_layout(pad=10.0)
 
# Displaying the plot.
print("The Autocorrelation plot for the data is:")

 
plt.show()


# In[ ]:


def ad_test(df):
    dftest=adfuller(df,autolag='AIC')
    print("1. ADF ",dftest[0])
    print("2. p-value ",dftest[1])
 

print("d=0")
ad_test(df['energy'].loc['2018':])
print("\n")
print("d =1")
ad_test(df['energy'].loc['2018':].diff(1)[1:])


# **we can see we can consider  ou differenciated series as stationnary with d=1**

# In[ ]:


from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


"""
stepwise_fit=auto_arima(df['energy'].loc['2018':'2021-12-26'],seasonal=True,m=52,trace=True,suppress_warnings=True)
stepwise_fit.summary()
"""


# ## Checking on 2022

# In[ ]:


mod=sm.tsa.statespace.SARIMAX(df['energy'].loc['2018':'2021-12-26'],order=(3,0, 1),seasonal_order=(2, 0, 0, 52))
res = mod.fit(disp=False)
print(res.summary())

pred=res.predict(start='2021-01-03',end='2022-01',typ='levels')

predfig=plt.figure(figsize=(15,7))
plt.plot(df['energy'].loc['2018':].index,df['energy'].loc['2018':],label='energy consumption')
plt.plot(pred.index, pred, label='sarima')
plt.legend(loc='best')
plt.grid()
plt.title('sarima energy forecasting')
plt.xlabel('time')
plt.ylabel('energy consumption')
plt.show()


# In[ ]:


mod=sm.tsa.statespace.SARIMAX(df['energy'].loc['2018':'2021-12-26'],order=(3,0, 1),seasonal_order=(2, 0, 0, 52))
res = mod.fit(disp=False)
pred=res.predict(start='2021-12-26',end='2023-01-02')

predfig=plt.figure(figsize=(15,7))
plt.plot(df['energy'].loc['2018':].index,df['energy'].loc['2018':],label='energy consumption')
plt.title('sarima energy forecasting')
plt.xlabel('time')
plt.ylabel('energy consumption')
plt.grid()
plt.plot(pred.index, pred , label='sarimax predictions')
plt.legend(loc='best')
plt.show()


# ## ML models

# In[ ]:


X_train=np.array(decomposition.trend.loc[:'2021-01-03'].dropna()).reshape(-1,1)
X_test=np.array(decomposition.trend.loc['2021-01-03':].dropna()).reshape(-1,1)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
scaled_train = scaler.transform(X_train)
scaled_test = scaler.transform(X_test)


# In[ ]:


from keras.preprocessing.sequence import TimeseriesGenerator
# define generator
n_input = 52
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()


# fit model
model.fit(generator,epochs=15)


# In[ ]:


plt.figure(figsize=(15,7))
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()


# In[ ]:


test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(X_test)):
    
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]
    
    # append the prediction into the array
    test_predictions.append(current_pred) 
    
    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[ ]:


true_predictions = scaler.inverse_transform(test_predictions)
true_predictions
test_pred=pd.Series(data=true_predictions.reshape(len(X_test),),index=decomposition.trend.loc['2021-01-03':].dropna().index)


# In[ ]:


plt.figure(figsize=(15,7))
decomposition.trend.loc[:'2021-01-03'].dropna().plot(label='train')
decomposition.trend.loc['2021-01-03':].dropna().plot(label='test')
test_pred.plot(label='pred')
plt.grid()
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(15,7))
df['energy'].loc[:'2021-01-03'].plot(label='energy')
df['energy'].loc['2021-01-03':'2021-07-04'].plot(label='true value')
(test_pred+decomposition.seasonal.loc['2021-01-03':'2021-07-04'].dropna()).plot(label='prediction')
plt.title('prediction using mixed model')
plt.grid()
plt.legend()
plt.show()


# **rmse :**

# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

sqrt(mean_squared_error(test_pred+decomposition.seasonal.loc['2021-01-03':'2021-07-04'].dropna(),df['energy'].loc['2021-01-03':'2021-07-04']))


# **we're still missing the residual part**

# In[ ]:


X_train_resid=np.array(decomposition.resid.loc[:'2021-01-03'].dropna()).reshape(-1,1)
X_test_resid=np.array(decomposition.resid.loc['2021-01-03':].dropna()).reshape(-1,1)

scaler_resid = MinMaxScaler()
scaler_resid.fit(X_train_resid)
scaled_train_resid = scaler_resid.transform(X_train_resid)
scaled_test_resid = scaler_resid.transform(X_test_resid)

n_input = 52
n_features = 1
generator = TimeseriesGenerator(scaled_train_resid, scaled_train_resid, length=n_input, batch_size=1)

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# fit model
model.fit(generator,epochs=15)


# In[ ]:


test_predictions = []

first_eval_batch = scaled_train_resid[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(X_test_resid)):
    
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]
    
    # append the prediction into the array
    test_predictions.append(current_pred) 
    
    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[ ]:


true_predictions = scaler_resid.inverse_transform(test_predictions)
test_pred_resid=pd.Series(data=true_predictions.reshape(len(X_test_resid),),index=decomposition.resid.loc['2021-01-03':].dropna().index)


# In[ ]:


decomposition.resid.loc['2021-01-03':].dropna().plot()
test_pred_resid.plot()


# In[ ]:


plt.figure(figsize=(15,7))
df['energy'].loc[:'2021-01-03'].plot(label='energy')
df['energy'].loc['2021-01-03':'2021-07-04'].plot(label='true value')
(test_pred+test_pred_resid+decomposition.seasonal.loc['2021-01-03':'2021-07-04'].dropna()).plot(label='prediction')
plt.title('prediction using mixed model')
plt.grid()
plt.legend()
plt.show()


# ----------

# ## 2023 predictions
