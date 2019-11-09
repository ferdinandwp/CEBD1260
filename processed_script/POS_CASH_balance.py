# This is for POS Cash balance
# Import library
import numpy as np  
import pandas as pd
import os
import glob

# Read data
path = '/Users/ferdinand/Desktop/data'
file_name = os.path.join(path,'POS_CASH_balance.csv')
POS_CASH_balance_df = pd.read_csv(file_name)

# Check memory size
mem_use = POS_CASH_balance_df.memory_usage().sum()/1024**2 # Convert to MB
print('--------------------------------------------------')
print('Initial dataset memory usage for application_train: {:2f} MB'.format(mem_use))
print('--------------------------------------------------')

# Check data types
print('INITIAL DATATYPE:')
print(POS_CASH_balance_df.dtypes)
print('--------------------------------------------------')

# Get non object features
features = [f for f in POS_CASH_balance_df.columns.values]

nb_features = []
for f in features:
    if str(POS_CASH_balance_df[f].dtype) in ['int64','float64']:
        nb_features.append(f)

print('ALL FEATURES WITH INT64 & FLOAT64:')
print(nb_features)
print('--------------------------------------------------')

# Find min and max for each features in int64 & float64  
print('MAX AND MID FOR INT64 & FLOAT64 FEATURES:')
for f in nb_features:
    print("{}: max= {}, min= {}".format(f,POS_CASH_balance_df[f].max(),POS_CASH_balance_df[f].min()))

# Fill missing values
POS_CASH_balance_df['CNT_INSTALMENT'].fillna(POS_CASH_balance_df['CNT_INSTALMENT'].mean(),inplace=True)
POS_CASH_balance_df['CNT_INSTALMENT_FUTURE'].fillna(POS_CASH_balance_df['CNT_INSTALMENT_FUTURE'].mean(),inplace=True)

# Change datatype for memory optimization
POS_CASH_balance_df['MONTHS_BALANCE'] = POS_CASH_balance_df['MONTHS_BALANCE'].astype(np.int8)
POS_CASH_balance_df['CNT_INSTALMENT'] = POS_CASH_balance_df['CNT_INSTALMENT'].astype(np.int8)
POS_CASH_balance_df['CNT_INSTALMENT_FUTURE'] = POS_CASH_balance_df['CNT_INSTALMENT_FUTURE'].astype(np.int8)
POS_CASH_balance_df['SK_DPD'] = POS_CASH_balance_df['SK_DPD'].astype(np.int16)
POS_CASH_balance_df['SK_DPD_DEF'] = POS_CASH_balance_df['SK_DPD_DEF'].astype(np.int16)

print('--------------------------------------------------')

# Change object into category
POS_CASH_balance_df['NAME_CONTRACT_STATUS'] = POS_CASH_balance_df['NAME_CONTRACT_STATUS'].astype('category')

# Use label encoder
POS_CASH_balance_df['CAT_NAME_CONTRACT_STATUS'] = POS_CASH_balance_df['NAME_CONTRACT_STATUS'].cat.codes

# Remove object feature from dataset
POS_CASH_balance_df = POS_CASH_balance_df.drop(columns=['NAME_CONTRACT_STATUS'])

print('--------------------------------------------------')
print(POS_CASH_balance_df.head())