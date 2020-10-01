import pandas as pd
from scipy.sparse.construct import random

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from datetime import datetime


df = pd.read_csv('weatherHistory.csv')
df = df.rename(columns={'Apparent Temperature (C)': 'temp', 'Formatted Date': 'date'})

df['date'] = pd.to_datetime(df['date'])

startdate = pd.to_datetime("2006-4-1")
enddate = pd.to_datetime("2006-4-28")

df = df.loc[startdate:enddate].head(500)

X, y = df.loc[:, ['Humidity']], df['temp']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

print(f'Training score: {lr.score(X_train, y_train)}')
print(f'Test score: {lr.score(X_test, y_test)}')
