#!/usr/bin/env python
# coding: utf-8

# In[19]:


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import datetime as dt
from datetime import datetime
from nsepy import get_history

from sklearn.preprocessing import MinMaxScaler

### For Creating the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import Dropout

import numpy as np
import pandas as pd


# In[97]:


Stock = str(input("enter stock ticker :"))

rawdata=get_history(symbol=Stock,start=dt.date(2015,8,29),end=dt.date(2021,8,29))
realdata=rawdata
rawdata=rawdata.drop(["Symbol","Series","Prev Close","Open","High","Low","Last","VWAP","Volume","Turnover","Trades","Deliverable Volume","%Deliverble"],axis=1)



rawdata


# In[114]:



#Visualizing the closing price history for the training dataset


plt.title(Stock+" original data")

plt.plot(rawdata.index, rawdata['Close'])


# In[ ]:





# In[99]:


rawdata['Close'] = rawdata['Close'].fillna(0)
rawdata=rawdata[rawdata['Close'] != 0]
rawdata.shape


# In[100]:


#preparing training and testing data
dataset_final = rawdata.values
train_ds = dataset_final[0:1000,:]
valid_ds = dataset_final[1000:,:]


# In[101]:


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset_final)


# In[102]:


#creating sliding window of 60 days
x_train, y_train = [], []
for i in range(60,len(train_ds)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)


# In[103]:


#Converting to 3D
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


# In[104]:


assert not np.any(np.isnan(x_train))


# In[105]:


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1),activation="relu"))
model.add(Dropout(rate = 0.01))

model.add(LSTM(units=50, return_sequences = True,activation="relu"))
model.add(Dropout(rate = 0.01))

model.add(LSTM(units=50, return_sequences = True,activation="relu"))
model.add(Dropout(rate = 0.01))

model.add(LSTM(units=50, return_sequences = False,activation="relu"))
model.add(Dropout(rate = 0.01))

model.add(Dense(1))

model.summary()


# In[106]:


model.compile(loss='mean_squared_error', optimizer='adam')


# In[107]:


model.fit(x_train, y_train, epochs=100, batch_size=64)


# In[108]:


#predicting test data values, using past 60 from the train data
inputs = rawdata[len(rawdata) - len(valid_ds)-60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)
inputs


# In[109]:


X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test.shape, X_test


# In[110]:


X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
preds = model.predict(X_test)
preds = scaler.inverse_transform(preds)

X_test.shape, preds.shape


# In[111]:


rms=np.sqrt(np.mean(np.power((valid_ds-preds),2)))
rms


# In[113]:


#for plotting
train = rawdata[0:1000]
valid = rawdata[1000:]
valid['Predictions'] = preds
plt.figure(figsize=(20,8))
plt.plot(train['Close'])
plt.plot(valid['Close'], color = 'blue', label = 'Real Price')
plt.plot(valid['Predictions'], color = 'red', label = 'Predicted Price')
plt.title(Stock+" price prediction")
plt.legend()

plt.show()


# In[132]:


look_back=30
inputs = rawdata[-look_back:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)


# In[133]:


forecast_list=[]
for _ in range(7):
  x = inputs[-look_back:]
  x = np.reshape(x, (x.shape[1],x.shape[0],1))
  forecast = model.predict(x)
  forecast = scaler.inverse_transform(forecast)
  inputs = scaler.inverse_transform(inputs)
  inputs = np.append(inputs,forecast[0][0])
  inputs = inputs.reshape(-1,1)
  inputs  = scaler.transform(inputs)
  forecast_list.append(forecast[0][0])


# In[134]:


last_date = realdata.index.values[-1]
prediction_dates = pd.date_range(last_date, periods=31).tolist()
prediction_dates = prediction_dates[1:]
prediction_dates


# In[138]:


fpred=pd.DataFrame(list(zip(prediction_dates, forecast_list)), 
               columns =['Date', 'Forecasted Price']) 
print("Forcasted price for the next week:")
fpred


# In[ ]:





# In[ ]:




