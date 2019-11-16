# Script for bureau_balance.csv

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Defining the directory

DATA_PATH = 'C:\\Users\\KALPAW01\\Dropbox\\Data_for_ML_course'
br_balance = os.path.join(DATA_PATH, 'bureau_balance.csv')

train_df = pd.read_csv(br_balance)


# identifying the Nan
nas = train_df.isna()
nacnt = {}
for f in train_df.columns:
    n = nas[f].sum()
    nacnt[n] = []
for f in train_df.columns:
    n = nas[f].sum()
    nacnt[n].append(f)

print(nacnt)

# Memory usage prior to optimization

mem_use = train_df.memory_usage().sum() / 1024**3
print('Memory usage of dataframe is {:.2f} GB'.format(mem_use))

# Converting to lower format to optimize memory usage

train_df['SK_ID_BUREAU'] = train_df['SK_ID_BUREAU'].astype(np.int32)
train_df['MONTHS_BALANCE'] = train_df['MONTHS_BALANCE'].astype(np.int8)

# Memory usage after optimization

mem_use = train_df.memory_usage().sum() / 1024**3
print('Memory usage of dataframe is {:.2f} GB'.format(mem_use))

# Encoding remaining object with more than 5 categories using label encoder
le = LabelEncoder()

le.fit_transform(train_df['STATUS'])
train_df['STATUS'].factorize(sort=True)
print(le.classes_)
