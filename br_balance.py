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

# Encoding the objects by using one-hot-coding
temp_df = pd.get_dummies(train_df['STATUS'], prefix='STATUS')
train_df = pd.concat([train_df, temp_df], axis=1)

# Memory usage prior to optimization

mem_use = train_df.memory_usage().sum() / 1024**3
print('Memory usage of dataframe is {:.2f} GB'.format(mem_use))

for f in train_df:
    print(" {}: Type:{} Max:{} Min:{}".format(f, train_df[f].dtype, train_df[f].max(), train_df[f].min()))

# dropping the one-hot-coded columns (CREDIT_ACTIVE AND CREDIT_CURRENCY)
train_df.drop(['STATUS'], axis=1)

# Converting to lower format to optimize memory usage

train_df['SK_ID_BUREAU'] = train_df['SK_ID_BUREAU'].astype(np.int32)
train_df['MONTHS_BALANCE'] = train_df['MONTHS_BALANCE'].astype(np.int8)
train_df['STATUS_0'] = train_df['STATUS_0'].astype(np.int8)
train_df['STATUS_1'] = train_df['STATUS_1'].astype(np.int8)
train_df['STATUS_2'] = train_df['STATUS_2'].astype(np.int8)
train_df['STATUS_3'] = train_df['STATUS_3'].astype(np.int8)
train_df['STATUS_4'] = train_df['STATUS_4'].astype(np.int8)
train_df['STATUS_5'] = train_df['STATUS_5'].astype(np.int8)
train_df['STATUS_C'] = train_df['STATUS_C'].astype(np.int8)
train_df['STATUS_X'] = train_df['STATUS_X'].astype(np.int8)

# Memory usage after optimization

mem_use = train_df.memory_usage().sum() / 1024**3
print('Memory usage of dataframe is {:.2f} GB'.format(mem_use))

# Aggregating the features

client_ids = list(train_df['SK_ID_BUREAU'].unique())
temp_df1 = train_df[train_df['SK_ID_BUREAU'].isin(client_ids)]

agg_dict = {
    'MONTHS_BALANCE': ['mean', 'max', 'min', 'std']
    }

agg_df = temp_df1.groupby('SK_ID_BUREAU').agg(agg_dict)
# Arrange columns names to be more readable
agg_df.columns = ['br_bureau_{}_{}'.format(x[0], x[1]) for x in agg_df.columns.tolist()]

temp_df2 = temp_df1.merge(agg_df, on='SK_ID_BUREAU', how='left')

print(temp_df1.shape)
print(temp_df1.head)
# Encoding remaining object with more than 5 categories using label encoder
# le = LabelEncoder()
#
# le.fit_transform(train_df['STATUS'])
# train_df['STATUS'].factorize(sort=True)
# print(le.classes_)
#
# # Exporting result to CSV
# export_csv = train_df.to_csv(r'C:\Users\KALPAW01\OneDrive - Air Canada\Desktop\Testing_br_Bureau.csv', index=None, header=True)
