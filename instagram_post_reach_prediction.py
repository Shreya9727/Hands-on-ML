#Author: K.T.Shreya Parthasarathi
#Date: 19/01/2021
#Code Title: Instagram Post Reach Prediction

#Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Read file
file = pd.read_csv('instagram_reach.csv')
print(file.head())

#Replace hours to empty string
file_new = file.replace(["hours","hour"], " ", regex = True)
print (file_new.head())

#Typecasting
file_changed = file_new.astype({"Time since posted":'int64'})
print (file_changed.info())

#Data splitting
feature_cols = ['Followers','Time since posted']
X = file_changed[feature_cols]
y = file_changed.Likes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Model Building
rm = LinearRegression()
rm = rm.fit(X_train, y_train)

#Prediction
prediction = rm.predict(X_test)
print (prediction)
mse = mean_squared_error(y_test, prediction)
print ('MSE: ', mse)

