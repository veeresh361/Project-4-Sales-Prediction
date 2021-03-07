#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Reading The Data

# In[102]:


d=pd.read_csv('perrin-freres-monthly-champagne-.csv',sep=',',parse_dates=True,index_col='Month')


# In[103]:


d.head()


# In[104]:


d.columns=['Champagne_sales']


# In[105]:


d.head()


# # Visualizing the Data

# In[106]:


d.plot(figsize=(12,8))


# In[107]:


d.shape


# In[108]:


train_data=d.iloc[:80]


# In[109]:


test_data=d.iloc[80:]


# In[110]:


plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
train_data['Champagne_sales'].plot()
plt.subplot(1,2,2)
test_data['Champagne_sales'].plot()


# In[111]:


from keras.layers import LSTM,Dense
from keras.models import Sequential


# In[112]:


from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler


# In[113]:


scaler=MinMaxScaler()


# In[114]:


train_data=scaler.fit_transform(train_data)
test_data=scaler.transform(test_data)


# # Data Prepration

# In[115]:


generator=TimeseriesGenerator(train_data,train_data,length=12,batch_size=1)


# In[116]:


generator[0]


# # Model Building

# In[89]:


model=Sequential()
model.add(LSTM(100,activation='relu',input_shape=(12,1)))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')


# In[117]:


model.fit_generator(generator,epochs=25)


# In[118]:


first_batch=train_data[-12:]
batch=first_batch.reshape(1,12,1)
test_prediction=[]
for i in range(len(test_data)):
    y_pred=model.predict(batch)[0]
    test_prediction.append(y_pred)
    batch=np.append(batch[:,1:,:],[[y_pred]],axis=1)


# In[92]:


import numpy as np


# In[119]:


test_prediction=scaler.inverse_transform(test_prediction)


# In[120]:


prediction=pd.DataFrame(test_prediction,columns=['Prediction'])


# In[95]:


prediction


# In[121]:


test_data=scaler.inverse_transform(test_data)


# In[122]:


prediction['Actual']=test_data


# In[124]:


new_pred.plot(figsize=(12,8),legend=True)


# In[123]:


new_pred=prediction.iloc[:-3,:]


# # Final Prediction

# In[99]:


prediction


# In[68]:


test_data=pd.DataFrame(test_data)


# In[69]:


test_data.plot()


# In[ ]:




