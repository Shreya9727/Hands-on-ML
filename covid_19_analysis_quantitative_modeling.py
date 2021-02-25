##Author: K.T.Shreya Parthasarathi
##Date: 06/01/2021
##Code title: COVID-19 Analysis (Quantitative Modeling)

#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read file
file = pd.read_csv('covid19.csv')
#print (file)

#Groupby Dates
grouped_file = file.groupby(['Date'], sort = False).sum()
#print (grouped_file.head())
new_grouped = grouped_file.drop(['Sno'], axis = 1)
#print (new_grouped.head())

#Sum Rows
total = new_grouped.sum(axis = 1)
print (total)

#Plotting 
cases = np.array(total)
dates = total.index
plt.plot(dates, cases, linestyle = 'solid')
plt.gcf().autofmt_xdate()
plt.xticks(rotation=90)
plt.show()

#Select specified dates
new_file = total.loc['04/03/20' : '21/03/20']
#print (new_file)

#Final prediction calculations
cases = np.array(new_file)
#print (cases)
rate = [(cases[i+1]-cases[i])/cases[i] for i in range(1, len(cases)-1)]
r = (sum(rate)/len(rate))

#Test
P0 = 31
t = 26
Pt = P0 * (2.303**(r*t))
print (Pt)
    

