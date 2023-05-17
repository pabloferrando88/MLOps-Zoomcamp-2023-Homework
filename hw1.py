# -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:13:03 2023

@author: P.FERRANDOVILLALBA
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error


df1 = pd.read_parquet('yellow_tripdata_2022-01.parquet')
df2 = pd.read_parquet('yellow_tripdata_2022-02.parquet')
q1 = len(df1.columns)

df1['duration'] = (df1['tpep_dropoff_datetime'] - df1['tpep_pickup_datetime']).astype('timedelta64[m]')

q2 = df1['duration'].std()
cond = (df1['duration'] >= 1) & (df1['duration'] <= 60)
q3 = sum(cond)/len(df1)

df1 = df1[cond]
y1 = df1['duration']


df1 = df1[['PULocationID', 'DOLocationID']]
for col in df1.columns:
    df1[col] = df1[col].astype(str)

dict1 = df1.to_dict('records')

dv = DictVectorizer()

X1 = dv.fit_transform(dict1)
q4 = X1.shape[1]

model = LinearRegression()
model.fit(X1, y1)


q5 = mean_squared_error(y1, model.predict(X1), squared=False)

df2['duration'] = (df2['tpep_dropoff_datetime'] - df2['tpep_pickup_datetime']).astype('timedelta64[m]')
df2 = df2[(df2['duration'] >= 1) & (df2['duration'] <= 60)]
y2 = df2['duration']
df2 = df2[['PULocationID', 'DOLocationID']]
dict2 = df2.to_dict('records')

X2 = dv.transform(dict2)
q6 = mean_squared_error(y2, model.predict(X2), squared=False)
