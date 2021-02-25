#Author: K.T.Shreya
#Code: Stock price prediction

#Data Collection & Analysis
import pandas as pd
import numpy as np
import quandl
quandl.ApiConfig.api_key = "QpYW5gRhevJbuy71yxsi"
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

data = quandl.get("WIKI/AAPL")
print (data)

df = data
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Close']*100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open']*100.0
print (df)

df_new = df[['Adj. Close','PCT_change', 'HL_PCT', 'Adj. Volume']]

df_new.fillna(value = -99999, inplace = True)
print (df_new)

forecast_col = 'Adj. Close'
forecast = int(math.ceil(0.01 * len(df)))

df_new['label'] = df[forecast_col].shift(-forecast)
df_new.dropna(inplace = True)
print (df_new)

feature_cols = ['Adj. Close', 'PCT_change', 'HL_PCT', 'Adj. Volume']
X = df_new[feature_cols]
y = df_new.label

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)

rm = LinearRegression()
rm = rm.fit(X_train, y_train)

y_predict = rm.predict(X_test)
print(y_predict)

mse = mean_squared_error(y_test, y_predict)
print ('MSE = ', mse)
