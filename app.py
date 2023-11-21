import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model('Stock_prediction_model.keras')

st.header('Stock Prediction App')

stock= st.text_input("Enter the stock symbol", "MRF.NS")

start = '2012-01-01'
end = '2022-12-21'

data = yf.download(stock, start, end)
st.subheader('Last 10 days data')
st.write(data.tail(10))

#calculate the moving average over 50,100,200 days
ma_50 = data['Close'].rolling(50).mean()
ma_100 = data['Close'].rolling(100).mean()
ma_200 = data['Close'].rolling(200).mean()

st.subheader('Stock Price vs Moving Averages')
fig1 = plt.figure(figsize=(12,8))
plt.plot(data['Close'],label='Stock Price')
plt.plot(ma_50,'r',label='50 day Moving Average')
plt.plot(ma_100,'b',label='100 day Moving Average')
plt.plot(ma_200,'g',label='200 day Moving Average')
plt.legend()
plt.show()
st.pyplot(fig1)


#adding the first 80% of the data values into training set and the rest 20% into test set
data_train = data['Close'][0:int(len(data)*0.80)]
data_test = data['Close'][int(len(data)*0.80):len(data)]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data.test =pd.concat([pas_100_days,data_test],ignore_index=True)

data_test_scale = scaler.fit_transform(np.array(data_test).reshape(-1,1))

#appending the last 100 days data into a list to feed to the model
x=[]
y=[]
for i in range(100,data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x= np.array(x)
y= np.array(y)

predict = model.predict(x)

predict = predict*(1/scaler.scale_)
y = y*(1/scaler.scale_)

st.subheader('Stock Price vs Predicted Stock Price')
fig2 = plt.figure(figsize=(12,8))
plt.plot(y,label='Stock Price')
plt.plot(predict,'r',label='Predicted Stock Price')
plt.legend()
plt.xlabel('Time in days')
plt.ylabel('Stock Price')
plt.show()
st.pyplot(fig2)
