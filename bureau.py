# This is bureau script!

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Defining the directory

DATA_PATH = 'C:\\Users\\KALPAW01\\Dropbox\\Data_for_ML_course'
bureau = os.path.join(DATA_PATH, 'bureau.csv')

train_df = pd.read_csv(bureau)


# identifying the Nan
nas = train_df.isna()
nacnt = {}
for f in train_df.columns:
    n = nas[f].sum()
    nacnt[n]=[]
for f in train_df.columns:
    n = nas[f].sum()
    nacnt[n].append(f)

# replacing the Nan by the mean. When data is normally distributed, we can use average if the data has a skewed
# distribution, we should use median instead

train_df['DAYS_CREDIT_ENDDATE'].fillna(train_df['DAYS_CREDIT_ENDDATE'].mean(), inplace=True)
train_df['DAYS_ENDDATE_FACT'].fillna(train_df['DAYS_ENDDATE_FACT'].mean(), inplace=True)
train_df['AMT_CREDIT_MAX_OVERDUE'].fillna(train_df['AMT_CREDIT_MAX_OVERDUE'].mean(), inplace=True)
train_df['AMT_CREDIT_SUM'].fillna(train_df['AMT_CREDIT_SUM'].mean(), inplace=True)
train_df['AMT_CREDIT_SUM_DEBT'].fillna(train_df['AMT_CREDIT_SUM_DEBT'].mean(), inplace=True)
train_df['AMT_CREDIT_SUM_LIMIT'].fillna(train_df['AMT_CREDIT_SUM_LIMIT'].mean(), inplace=True)
train_df['AMT_ANNUITY'].fillna(train_df['AMT_ANNUITY'].mean(), inplace=True)

# Memory usage prior to optimization

mem_use = train_df.memory_usage().sum() / 1024**3
print('Memory usage of dataframe is {:.2f} GB'.format(mem_use))

# Picking the proper format

# What is the Type, the Max and the Min of each feature

for f in train_df:
    print(" {}: Type:{} Max:{} Min:{}".format(f, train_df[f].dtype, train_df[f].max(), train_df[f].min()))

# Float16 Max and Min

print("Float16 Max:", np.finfo(np.float16).max, "Float16 Max:", np.finfo(np.float16).min)

# Float32 Max and Min

print("Float32 Max:", np.finfo(np.float32).max, "Float32 Max:", np.finfo(np.float32).min)

# Int16 Max and Min
print("Int8 Max:", np.iinfo(np.int8).max, "Int8 Max:", np.iinfo(np.int8).min)

# Int16 Max and Min
print("Int16 Max:", np.iinfo(np.int16).max, "Int16 Max:", np.iinfo(np.int16).min)

# Int32 Max and Min
print("Int32 Max:", np.iinfo(np.int32).max, "Int32 Max:", np.iinfo(np.int32).min)

# Converting to lower format to optimize memory usage

train_df['SK_ID_CURR'] = train_df['SK_ID_CURR'].astype(np.int32)
train_df['SK_ID_BUREAU'] = train_df['SK_ID_BUREAU'].astype(np.int32)
train_df['DAYS_CREDIT'] = train_df['DAYS_CREDIT'].astype(np.int16)
train_df['CREDIT_DAY_OVERDUE'] = train_df['CREDIT_DAY_OVERDUE'].astype(np.int16)
train_df['DAYS_CREDIT_ENDDATE'] = train_df['DAYS_CREDIT_ENDDATE'].astype(np.float16)
train_df['DAYS_ENDDATE_FACT'] = train_df['DAYS_ENDDATE_FACT'].astype(np.float16)
train_df['AMT_CREDIT_MAX_OVERDUE'] = train_df['AMT_CREDIT_MAX_OVERDUE'].astype(np.float32)
train_df['CNT_CREDIT_PROLONG'] = train_df['CNT_CREDIT_PROLONG'].astype(np.int8)
train_df['AMT_CREDIT_SUM'] = train_df['AMT_CREDIT_SUM'].astype(np.float32)
train_df['AMT_CREDIT_SUM_DEBT'] = train_df['AMT_CREDIT_SUM_DEBT'].astype(np.float32)
train_df['AMT_CREDIT_SUM_LIMIT'] = train_df['AMT_CREDIT_SUM_LIMIT'].astype(np.float32)
train_df['AMT_CREDIT_SUM_OVERDUE'] = train_df['AMT_CREDIT_SUM_OVERDUE'].astype(np.float32)
train_df['DAYS_CREDIT_UPDATE'] = train_df['DAYS_CREDIT_UPDATE'].astype(np.int32)
train_df['AMT_ANNUITY'] = train_df['AMT_ANNUITY'].astype(np.float32)


# Memory usage after optimization

mem_use = train_df.memory_usage().sum() / 1024**3
print('Memory usage of dataframe is {:.2f} GB'.format(mem_use))

# Encoding the objects by using one-hot-coding
temp_df = pd.get_dummies(train_df['CREDIT_ACTIVE'], prefix='CREDIT_ACTIVE')
train_df = pd.concat([train_df, temp_df], axis=1)

temp_df = pd.get_dummies(train_df['CREDIT_CURRENCY'], prefix='CREDIT_CURRENCY')
train_df = pd.concat([train_df, temp_df], axis=1)

# Encoding remaining object with more than 5 categories using label encoder
le = LabelEncoder()

le.fit_transform(train_df['CREDIT_TYPE'])
train_df['CREDIT_TYPE'].factorize(sort=True)

# dropping the one-hot-coded columns (CREDIT_ACTIVE AND CREDIT_CURRENCY)
train_df.drop(['CREDIT_ACTIVE', 'CREDIT_CURRENCY'], axis=1)

print(train_df.shape)

for f in train_df:
    print(" {}: Type:{} Max:{} Min:{}".format(f, train_df[f].dtype, train_df[f].max(), train_df[f].min()))