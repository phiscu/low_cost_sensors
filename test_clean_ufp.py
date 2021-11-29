import pandas as pd
import numpy as np
import datetime
import seaborn as sbn
import matplotlib.pyplot as plt



path = "20180719_tour1_raw.csv"

data = pd.read_csv(path, sep = ",", header = 0,  na_values=(16777200), parse_dates=['Time'])

print(data['Time'] - datetime.datetime(2018, 7, 19))


data['Time'] = data['Time'] - pd.Timedelta(days = 1229, hours=2)

print(data[1545:1568])

data_c = data.copy()

print(data_c.iloc[1547,0])
print(data_c.iloc[1547,1])

for i in data_c.index:
       if data_c.iloc[i,1] == 0:
              data_c.iloc[i:i+7,1] = np.nan


print(data_c[1545:1568])


for i in reversed(data.index):
       if data.iloc[i, 1] == 0:
              data.iloc[i:i+7,1] = np.nan

print(data[1545:1568])

data = data.dropna()

print(data[1545:1568])


print(data.describe())

sbn.distplot(data['Concentration'], kde=False, color='blue', bins=100)
plt.title('UFP-Konzentration entlang der Messroute', fontsize=18)
plt.xlabel('Konzentration [#/cm^3]', fontsize=16)
plt.ylabel('Häufigkeitsdichte', fontsize=16)
plt.show()

data.to_csv("20180719_tour1_edit.csv", index=False)