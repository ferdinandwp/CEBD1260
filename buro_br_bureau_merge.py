# This is the script to merge Left join bureau_bl.csv to bureau.csv

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder

# Importing multiple .csv

DATA_PATH = 'C:\\Users\\KALPAW01\\Dropbox\\Data_for_ML_course\\'
flnm_df = glob.glob(DATA_PATH+"*.csv")



# this is part of the bureau script right before aggregation train_df was replaced to avoid conflict between the files

# Defining the directory
#
# DATA_PATH = 'C:\\Users\\KALPAW01\\Dropbox\\Data_for_ML_course'
# bureau = os.path.join(DATA_PATH, 'bureau.csv')

# This references the bureau balanced csv
brbl_df = pd.read_csv(flnm_df[2])

# This references the bureau csv
train_df = pd.read_csv(flnm_df[1])


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

temp_df = pd.get_dummies(train_df['CREDIT_TYPE'], prefix='CREDIT_TYPE')
train_df = pd.concat([train_df, temp_df], axis=1)


# # Encoding remaining object with more than 5 categories using label encoder
# le = LabelEncoder()
#
# le.fit_transform(train_df['CREDIT_TYPE'])
# train_df['CREDIT_TYPE'].factorize(sort=True)

# dropping the one-hot-coded columns (CREDIT_ACTIVE AND CREDIT_CURRENCY)
train_df.drop(['CREDIT_ACTIVE', 'CREDIT_CURRENCY','CREDIT_TYPE'], axis=1)

print(train_df.shape)

for f in train_df:
    print(" {}: Type:{} Max:{} Min:{}".format(f, train_df[f].dtype, train_df[f].max(), train_df[f].min()))

# Aggregating the features

client_ids = list(train_df['SK_ID_CURR'].unique())
temp_df1 = train_df[train_df['SK_ID_CURR'].isin(client_ids)]

agg_dict = {
    'CREDIT_ACTIVE': ['nunique'],
    'CREDIT_CURRENCY': ['nunique'],
    'DAYS_CREDIT': ['mean', 'max', 'min'],
    'CREDIT_DAY_OVERDUE': ['mean', 'max', 'min'],
    'DAYS_CREDIT_ENDDATE': ['mean', 'max', 'min'],
    'DAYS_ENDDATE_FACT': ['mean', 'max', 'min'],
    'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max', 'min'],
    'CNT_CREDIT_PROLONG': ['mean', 'max', 'min'],
    'AMT_CREDIT_SUM': ['mean', 'max', 'min'],
    'AMT_CREDIT_SUM_DEBT': ['mean', 'max', 'min'],
    'AMT_CREDIT_SUM_LIMIT': ['mean', 'max', 'min'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean', 'max', 'min'],
    'CREDIT_TYPE': ['nunique'],
    'DAYS_CREDIT_UPDATE': ['mean', 'max', 'min'],
    'AMT_ANNUITY': ['mean', 'max', 'min']
}

agg_df = temp_df1.groupby('SK_ID_CURR').agg(agg_dict)
# Arrange columns names to be more readable
agg_df.columns = ['BURO_{}_{}'.format(x[0], x[1]) for x in agg_df.columns.tolist()]

temp_df1 = temp_df1.merge(agg_df, on='SK_ID_CURR', how='left')

print(temp_df1.shape)

for f in temp_df1:
    print(" {}: Type:{} Max:{} Min:{}".format(f, temp_df1[f].dtype, temp_df1[f].max(), temp_df1[f].min()))


################################################################################

# This is the br_balance script with aggregation

# Defining the directory
#
# DATA_PATH = 'C:\\Users\\KALPAW01\\Dropbox\\Data_for_ML_course'
# br_balance = os.path.join(DATA_PATH, 'bureau_balance.csv')
#
# br_balance_df = pd.read_csv(br_balance)

#
# Creating the group by
gp = brbl_df.groupby('SK_ID_BUREAU')

# Aggregating the MONTHS_BALANCE by using median
add_df = pd.DataFrame(gp['MONTHS_BALANCE'].median())

# Renamed the column to make it easier to ID after joining to Main table
add_df = add_df.rename({'MONTHS_BALANCE': 'BRBL_MONTHS_BALANCE_median'}, axis=1)

# Using a pivot table to get the frequency of every STATUS
STS_Agg = brbl_df.pivot_table(index="SK_ID_BUREAU", columns= "STATUS", aggfunc="count")
# Arrange column names to be more readable
STS_Agg.columns = ['BURO_{}_{}'.format(x[0],x[1]) for x in STS_Agg.columns.tolist()]
STS_Agg

# Joining the pivot table (STS_Agg) with the aggregated MONTH_BALANCE (add_df)
add1_df = add_df.merge(STS_Agg,on='SK_ID_BUREAU',how='left')

# identifying the Nan
nas = add1_df.isna()
nacnt = {}
for f in add1_df.columns:
    n = nas[f].sum()
    nacnt[n] = []
for f in add1_df.columns:
    n = nas[f].sum()
    nacnt[n].append(f)

print(nacnt)

# Replacing the Nan by 0s
add1_df['BURO_MONTHS_BALANCE_0'].fillna(0,inplace=True)
add1_df['BURO_MONTHS_BALANCE_1'].fillna(0,inplace=True)
add1_df['BURO_MONTHS_BALANCE_2'].fillna(0,inplace=True)
add1_df['BURO_MONTHS_BALANCE_3'].fillna(0,inplace=True)
add1_df['BURO_MONTHS_BALANCE_4'].fillna(0,inplace=True)
add1_df['BURO_MONTHS_BALANCE_5'].fillna(0,inplace=True)
add1_df['BURO_MONTHS_BALANCE_C'].fillna(0,inplace=True)
add1_df['BURO_MONTHS_BALANCE_X'].fillna(0,inplace=True)

# Renaming the columns created by the pivot table
add1_df.rename({'BURO_MONTHS_BALANCE_0': 'BURO_STATUS_0', 'BURO_MONTHS_BALANCE_1': 'BURO_STATUS_1',
                'BURO_MONTHS_BALANCE_2': 'BURO_STATUS_2', 'BURO_MONTHS_BALANCE_3': 'BURO_STATUS_3',
                'BURO_MONTHS_BALANCE_4': 'BURO_STATUS_4', 'BURO_MONTHS_BALANCE_5': 'BURO_STATUS_5',
                'BURO_MONTHS_BALANCE_C': 'BURO_STATUS_C', 'BURO_MONTHS_BALANCE_X': 'BURO_STATUS_X'}, axis=1, inplace=True)

print(add1_df.shape)
print(temp_df1.shape)

##########################################################################

# Joining aggregated bureau_balance to aggregated Bureau using left join

train_brbl_merge = temp_df1.merge(add1_df,on='SK_ID_BUREAU',how='left')

print(train_brbl_merge.shape)
