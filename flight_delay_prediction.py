#Author: K.T.Shreya Parthasarathi
#Date: 14/01/2021
#Purpose: Flight delay Prediction

#Reading file and sample extraction
import pandas as pd
file = pd.read_csv('flights.csv', low_memory = False)
file_sample = file.head(100000)
print (file_sample.info())
print(file_sample['DIVERTED'].value_counts())

#Feature relation visualisation
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(file_sample)
plt.show()

#Correlation calculation
file_sample.corr()

#Data Cleaning
new_file = file_sample.drop(['YEAR','FLIGHT_NUMBER','AIRLINE','DISTANCE','TAIL_NUMBER','ORIGIN_AIRPORT','DESTINATION_AIRPORT','TAXI_OUT','SCHEDULED_TIME','DEPARTURE_TIME','WHEELS_OFF','ELAPSED_TIME','AIR_TIME','WHEELS_ON','DAY_OF_WEEK','TAXI_IN','ARRIVAL_TIME','CANCELLATION_REASON'], axis =1)
new_file.fillna(new_file.mean(), inplace= True)

#Add new result column
import numpy as np
new_file['RESULT'] = np.where(new_file['ARRIVAL_DELAY'] > 15, 1,0)
print (new_file.info())
print (new_file['RESULT'].value_counts())

#Split dataset
from sklearn.model_selection import train_test_split
feature_cols = ['MONTH','DAY','SCHEDULED_DEPARTURE','DEPARTURE_DELAY','SCHEDULED_ARRIVAL','DIVERTED','CANCELLED','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY']
X = new_file[feature_cols]
y = new_file.RESULT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

#Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#Model building using decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train_scaled, y_train)

#Prediction
y_pred = clf.predict(X_test_scaled)

#Accuracy calculation
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test, y_pred)
print (score)
