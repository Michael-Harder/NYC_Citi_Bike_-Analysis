# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Full_Concat.csv")

#Label Encoder
le = LabelEncoder()

df_usertype = pd.DataFrame(le.fit_transform(df['usertype'].to_numpy().reshape(-1,1)))


#One Hot Encoder

#start station name
enc = OneHotEncoder(sparse=False)

df_start_station = pd.DataFrame(enc.fit_transform(df['start station name'].to_numpy().reshape(-1,1)))


#end station name
df_end_station = pd.DataFrame(enc.fit_transform(df['end station name'].to_numpy().reshape(-1,1)))

#birth year
df['birth year'] = df['birth year'].fillna(0).astype('int32')
df_birth_year = pd.DataFrame(enc.fit_transform(df['birth year'].to_numpy().reshape(-1,1)))

#Standard Scaler

#trip durration
scaler = StandardScaler()
df_trip_durration = pd.DataFrame(scaler.fit_transform(df['tripduration'].to_numpy().reshape(-1,1)))


#merging dfs
df_all = pd.concat([df_usertype, df_start_station, df_end_station,
                    df_birth_year, df_trip_durration],axis = 1)

df1 = df

df1 = df1.drop(['usertype', 'start station name', 'end station name',
                'birth year', 'tripduration', 'start station id',
                'end station id', 'bikeid'], axis = 1)

df_final = pd.concat([df1, df_all],axis = 1)

df_final.to_csv('Preprocessed_Citi_Bike_Data.csv')



