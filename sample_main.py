# Import library
import numpy as np  
import pandas as pd
import os
import glob

# Read data
path = '/Users/ferdinand/Desktop/data'
file_name = os.path.join(path,'application_train.csv')

# Create application_train dataset
train_df = pd.read_csv(file_name)
print(train_df.shape)

# Check memory size
mem_use = train_df.memory_usage().sum()#/1024**2 # Convert to MB
print('Initial dataset memory usage for application_train: {:2f} MB'.format(mem_use))

chunk = pd.read_csv(file_name,chunksize=2000,iterator=True)
features = [f for f in train_df.columns.values]

# for i,df in enumerate(chunk):
#     if i < 2:
#         print(df.shape)

# for f in features:
#     print("{}: {}".format(f,train_df.dtypes))

# float64_features = []
# for f in features:
#     if str(train_df[f].dtype) in ['float64']:
#         float64_features.append(f)

# Optimize memory dataset
train_df['AMT_INCOME_TOTAL'] = train_df['AMT_INCOME_TOTAL'].astype(np.float32)
train_df['AMT_CREDIT'] = train_df['AMT_CREDIT'].astype(np.float32)
train_df['AMT_ANNUITY'] = train_df['AMT_ANNUITY'].astype(np.float32)
train_df['AMT_ANNUITY'] = train_df['AMT_ANNUITY'].astype(np.float32)
train_df['REGION_POPULATION_RELATIVE'] = train_df['REGION_POPULATION_RELATIVE'].astype(np.float16)
train_df['DAYS_REGISTRATION'] = train_df['DAYS_REGISTRATION'].astype(np.float16)
train_df['OWN_CAR_AGE'] = train_df['OWN_CAR_AGE'].astype(np.float16)
train_df['CNT_FAM_MEMBERS'] = train_df['CNT_FAM_MEMBERS'].astype(np.float16)
train_df['EXT_SOURCE_1'] = train_df['EXT_SOURCE_1'].astype(np.float16)
train_df['EXT_SOURCE_2'] = train_df['EXT_SOURCE_2'].astype(np.float16)
train_df['EXT_SOURCE_3'] = train_df['EXT_SOURCE_3'].astype(np.float16)
train_df['APARTMENTS_AVG'] = train_df['APARTMENTS_AVG'].astype(np.float16)
train_df['BASEMENTAREA_AVG'] = train_df['BASEMENTAREA_AVG'].astype(np.float16)
train_df['YEARS_BEGINEXPLUATATION_AVG'] = train_df['YEARS_BEGINEXPLUATATION_AVG'].astype(np.float16)
train_df['YEARS_BUILD_AVG'] = train_df['YEARS_BUILD_AVG'].astype(np.float16)
train_df['COMMONAREA_AVG'] = train_df['COMMONAREA_AVG'].astype(np.float16)
train_df['ELEVATORS_AVG'] = train_df['ELEVATORS_AVG'].astype(np.float16)
train_df['ENTRANCES_AVG'] = train_df['ENTRANCES_AVG'].astype(np.float16)
train_df['FLOORSMAX_AVG'] = train_df['FLOORSMAX_AVG'].astype(np.float16)
train_df['FLOORSMIN_AVG'] = train_df['FLOORSMIN_AVG'].astype(np.float16)
train_df['LANDAREA_AVG'] = train_df['LANDAREA_AVG'].astype(np.float16)
train_df['LIVINGAPARTMENTS_AVG'] = train_df['LIVINGAPARTMENTS_AVG'].astype(np.float16)
train_df['LIVINGAREA_AVG'] = train_df['LIVINGAREA_AVG'].astype(np.float16)
train_df['NONLIVINGAPARTMENTS_AVG'] = train_df['NONLIVINGAPARTMENTS_AVG'].astype(np.float16)
train_df['NONLIVINGAREA_AVG'] = train_df['NONLIVINGAREA_AVG'].astype(np.float16)
train_df['APARTMENTS_MODE'] = train_df['APARTMENTS_MODE'].astype(np.float16)
train_df['BASEMENTAREA_MODE'] = train_df['BASEMENTAREA_MODE'].astype(np.float16)
train_df['YEARS_BEGINEXPLUATATION_MODE'] = train_df['YEARS_BEGINEXPLUATATION_MODE'].astype(np.float16)
train_df['YEARS_BUILD_MODE'] = train_df['YEARS_BUILD_MODE'].astype(np.float16)
train_df['COMMONAREA_MODE'] = train_df['COMMONAREA_MODE'].astype(np.float16)
train_df['ELEVATORS_MODE'] = train_df['ELEVATORS_MODE'].astype(np.float16)
train_df['ENTRANCES_MODE'] = train_df['ENTRANCES_MODE'].astype(np.float16)
train_df['FLOORSMAX_MODE'] = train_df['FLOORSMAX_MODE'].astype(np.float16)
train_df['FLOORSMIN_MODE'] = train_df['FLOORSMIN_MODE'].astype(np.float16)
train_df['LANDAREA_MODE'] = train_df['LANDAREA_MODE'].astype(np.float16)
train_df['LIVINGAPARTMENTS_MODE'] = train_df['LIVINGAPARTMENTS_MODE'].astype(np.float16)
train_df['LIVINGAREA_MODE'] = train_df['LIVINGAREA_MODE'].astype(np.float16)
train_df['NONLIVINGAPARTMENTS_MODE'] = train_df['NONLIVINGAPARTMENTS_MODE'].astype(np.float16)
train_df['NONLIVINGAREA_MODE'] = train_df['NONLIVINGAREA_MODE'].astype(np.float16)
train_df['APARTMENTS_MEDI'] = train_df['APARTMENTS_MEDI'].astype(np.float16)
train_df['BASEMENTAREA_MEDI'] = train_df['BASEMENTAREA_MEDI'].astype(np.float16)
train_df['YEARS_BEGINEXPLUATATION_MEDI'] = train_df['YEARS_BEGINEXPLUATATION_MEDI'].astype(np.float16)
train_df['YEARS_BUILD_MEDI'] = train_df['YEARS_BUILD_MEDI'].astype(np.float16)
train_df['COMMONAREA_MEDI'] = train_df['COMMONAREA_MEDI'].astype(np.float16)
train_df['ELEVATORS_MEDI'] = train_df['ELEVATORS_MEDI'].astype(np.float16)
train_df['ENTRANCES_MEDI'] = train_df['ENTRANCES_MEDI'].astype(np.float16)
train_df['FLOORSMAX_MEDI'] = train_df['FLOORSMAX_MEDI'].astype(np.float16)
train_df['FLOORSMIN_MEDI'] = train_df['FLOORSMIN_MEDI'].astype(np.float16)
train_df['LANDAREA_MEDI'] = train_df['LANDAREA_MEDI'].astype(np.float16)
train_df['LIVINGAPARTMENTS_MEDI'] = train_df['LIVINGAPARTMENTS_MEDI'].astype(np.float16)
train_df['LIVINGAREA_MEDI'] = train_df['LIVINGAREA_MEDI'].astype(np.float16)
train_df['NONLIVINGAPARTMENTS_MEDI'] = train_df['NONLIVINGAPARTMENTS_MEDI'].astype(np.float16)
train_df['NONLIVINGAREA_MEDI'] = train_df['NONLIVINGAREA_MEDI'].astype(np.float16)
train_df['TOTALAREA_MODE'] = train_df['TOTALAREA_MODE'].astype(np.float16)
train_df['OBS_30_CNT_SOCIAL_CIRCLE'] = train_df['OBS_30_CNT_SOCIAL_CIRCLE'].astype(np.float16)
train_df['DEF_30_CNT_SOCIAL_CIRCLE'] = train_df['DEF_30_CNT_SOCIAL_CIRCLE'].astype(np.float16)
train_df['OBS_60_CNT_SOCIAL_CIRCLE'] = train_df['OBS_60_CNT_SOCIAL_CIRCLE'].astype(np.float16)
train_df['DEF_60_CNT_SOCIAL_CIRCLE'] = train_df['DEF_60_CNT_SOCIAL_CIRCLE'].astype(np.float16)
train_df['DAYS_LAST_PHONE_CHANGE'] = train_df['DAYS_LAST_PHONE_CHANGE'].astype(np.float16)
train_df['AMT_REQ_CREDIT_BUREAU_HOUR'] = train_df['AMT_REQ_CREDIT_BUREAU_HOUR'].astype(np.float16)
train_df['AMT_REQ_CREDIT_BUREAU_DAY'] = train_df['AMT_REQ_CREDIT_BUREAU_DAY'].astype(np.float16)
train_df['AMT_REQ_CREDIT_BUREAU_WEEK'] = train_df['AMT_REQ_CREDIT_BUREAU_WEEK'].astype(np.float16)
train_df['AMT_REQ_CREDIT_BUREAU_MON'] = train_df['AMT_REQ_CREDIT_BUREAU_MON'].astype(np.float16)
train_df['AMT_REQ_CREDIT_BUREAU_QRT'] = train_df['AMT_REQ_CREDIT_BUREAU_QRT'].astype(np.float16)
train_df['AMT_REQ_CREDIT_BUREAU_YEAR'] = train_df['AMT_REQ_CREDIT_BUREAU_YEAR'].astype(np.float16)

print(train_df.dtypes)
print('Initial dataset memory usage for application_train: {:2f} MB'.format(mem_use))

# Order attributes that has null values
nan_info = pd.DataFrame(train_df.isnull().sum()).reset_index()
nan_info.columns = ['col','nan_cnt']
nan_info.sort_values(by='nan_cnt',ascending=False,inplace=True)

cols_with_missing = nan_info.loc[nan_info.nan_cnt > 0].col.values

for f in cols_with_missing:
    print(" {}: {}".format(f,train_df[f].dtype))

train_df['NONLIVINGAPARTMENTS_MODE'].min(),train_df['NONLIVINGAPARTMENTS_MODE'].max(),train_df['NONLIVINGAPARTMENTS_MODE'].isna().sum()/len(train_df)
train_df['NONLIVINGAPARTMENTS_AVG'].min(),train_df['NONLIVINGAPARTMENTS_AVG'].max(),train_df['NONLIVINGAPARTMENTS_AVG'].isna().sum()/len(train_df)
train_df['NONLIVINGAPARTMENTS_MEDI'].min(),train_df['NONLIVINGAPARTMENTS_MEDI'].max(),train_df['NONLIVINGAPARTMENTS_MEDI'].isna().sum()/len(train_df)

train_df['LIVINGAPARTMENTS_MODE'].min(),train_df['LIVINGAPARTMENTS_MODE'].max(),train_df['LIVINGAPARTMENTS_MODE'].isna().sum()/len(train_df)
train_df['LIVINGAPARTMENTS_AVG'].min(),train_df['LIVINGAPARTMENTS_AVG'].max(),train_df['LIVINGAPARTMENTS_AVG'].isna().sum()/len(train_df)
train_df['LIVINGAPARTMENTS_MEDI'].min(),train_df['LIVINGAPARTMENTS_MEDI'].max(),train_df['LIVINGAPARTMENTS_MEDI'].isna().sum()/len(train_df)

train_df['FLOORSMIN_AVG'].min(),train_df['FLOORSMIN_AVG'].max(),train_df['FLOORSMIN_AVG'].isna().sum()/len(train_df)
train_df['FLOORSMIN_MODE'].min(),train_df['FLOORSMIN_MODE'].max(),train_df['FLOORSMIN_MODE'].isna().sum()/len(train_df)
train_df['FLOORSMIN_MEDI'].min(),train_df['FLOORSMIN_MEDI'].max(),train_df['FLOORSMIN_MEDI'].isna().sum()/len(train_df)

train_df['YEARS_BUILD_MEDI'].min(),train_df['YEARS_BUILD_MEDI'].max(),train_df['YEARS_BUILD_MEDI'].isna().sum()/len(train_df)
train_df['YEARS_BUILD_MODE'].min(),train_df['YEARS_BUILD_MODE'].max(),train_df['YEARS_BUILD_MODE'].isna().sum()/len(train_df)
train_df['YEARS_BUILD_AVG'].min(),train_df['YEARS_BUILD_AVG'].max(),train_df['YEARS_BUILD_AVG'].isna().sum()/len(train_df)

train_df['LANDAREA_MEDI'].min(),train_df['LANDAREA_MEDI'].max(),train_df['LANDAREA_MEDI'].isna().sum()/len(train_df)
train_df['LANDAREA_MODE'].min(),train_df['LANDAREA_MODE'].max(),train_df['LANDAREA_MODE'].isna().sum()/len(train_df)
train_df['LANDAREA_AVG'].min(),train_df['LANDAREA_AVG'].max(),train_df['LANDAREA_AVG'].isna().sum()/len(train_df)

train_df['BASEMENTAREA_MEDI'].min(),train_df['BASEMENTAREA_MEDI'].max(),train_df['BASEMENTAREA_MEDI'].isna().sum()/len(train_df)
train_df['BASEMENTAREA_AVG'].min(),train_df['BASEMENTAREA_AVG'].max(),train_df['BASEMENTAREA_AVG'].isna().sum()/len(train_df)
train_df['BASEMENTAREA_MODE'].min(),train_df['BASEMENTAREA_MODE'].max(),train_df['BASEMENTAREA_MODE'].isna().sum()/len(train_df)

train_df['NONLIVINGAREA_MODE'].min(),train_df['NONLIVINGAREA_MODE'].max(),train_df['NONLIVINGAREA_MODE'].isna().sum()/len(train_df)
train_df['NONLIVINGAREA_AVG'].min(),train_df['NONLIVINGAREA_AVG'].max(),train_df['NONLIVINGAREA_AVG'].isna().sum()/len(train_df)
train_df['NONLIVINGAREA_MEDI'].min(),train_df['NONLIVINGAREA_MEDI'].max(),train_df['NONLIVINGAREA_MEDI'].isna().sum()/len(train_df)

train_df['ELEVATORS_MEDI'].min(),train_df['ELEVATORS_MEDI'].max(),train_df['ELEVATORS_MEDI'].isna().sum()/len(train_df)
train_df['ELEVATORS_AVG'].min(),train_df['ELEVATORS_AVG'].max(),train_df['ELEVATORS_AVG'].isna().sum()/len(train_df)
train_df['ELEVATORS_MODE'].min(),train_df['ELEVATORS_MODE'].max(),train_df['ELEVATORS_MODE'].isna().sum()/len(train_df)

train_df['APARTMENTS_MEDI'].min(),train_df['APARTMENTS_MEDI'].max(),train_df['APARTMENTS_MEDI'].isna().sum()/len(train_df)
train_df['APARTMENTS_AVG'].min(),train_df['APARTMENTS_AVG'].max(),train_df['APARTMENTS_AVG'].isna().sum()/len(train_df)
train_df['APARTMENTS_MODE'].min(),train_df['APARTMENTS_MODE'].max(),train_df['APARTMENTS_MODE'].isna().sum()/len(train_df)

train_df['ENTRANCES_MEDI'].min(),train_df['ENTRANCES_MEDI'].max(),train_df['ENTRANCES_MEDI'].isna().sum()/len(train_df)
train_df['ENTRANCES_AVG'].min(),train_df['ENTRANCES_AVG'].max(),train_df['ENTRANCES_AVG'].isna().sum()/len(train_df)
train_df['ENTRANCES_MODE'].min(),train_df['ENTRANCES_MODE'].max(),train_df['ENTRANCES_MODE'].isna().sum()/len(train_df)

train_df['COMMONAREA_MEDI'].fillna(-1,inplace=True)
train_df['COMMONAREA_AVG'].fillna(-1,inplace=True)
train_df['COMMONAREA_MODE'].fillna(-1,inplace=True)

train_df['NONLIVINGAPARTMENTS_MODE'].fillna(-1,inplace=True)
train_df['NONLIVINGAPARTMENTS_AVG'].fillna(-1,inplace=True)
train_df['NONLIVINGAPARTMENTS_MEDI'].fillna(-1,inplace=True)

train_df['LIVINGAPARTMENTS_MODE'].fillna(-1,inplace=True)
train_df['LIVINGAPARTMENTS_AVG'].fillna(-1,inplace=True)
train_df['LIVINGAPARTMENTS_MEDI'].fillna(-1,inplace=True)

train_df['FLOORSMIN_AVG'].fillna(-1,inplace=True)
train_df['FLOORSMIN_MODE'].fillna(-1,inplace=True)
train_df['FLOORSMIN_MEDI'].fillna(-1,inplace=True)

train_df['YEARS_BUILD_MEDI'].fillna(-1,inplace=True)
train_df['YEARS_BUILD_MODE'].fillna(-1,inplace=True)
train_df['YEARS_BUILD_AVG'].fillna(-1,inplace=True)

train_df['LANDAREA_MEDI'].fillna(-1,inplace=True)
train_df['LANDAREA_MODE'].fillna(-1,inplace=True)
train_df['LANDAREA_AVG'].fillna(-1,inplace=True)

train_df['BASEMENTAREA_MEDI'].fillna(-1,inplace=True)
train_df['BASEMENTAREA_AVG'].fillna(-1,inplace=True)
train_df['BASEMENTAREA_MODE'].fillna(-1,inplace=True)

train_df['NONLIVINGAREA_MODE'].fillna(-1,inplace=True)
train_df['NONLIVINGAREA_AVG'].fillna(-1,inplace=True)
train_df['NONLIVINGAREA_MEDI'].fillna(-1,inplace=True)

train_df['ELEVATORS_MEDI'].fillna(-1,inplace=True)
train_df['ELEVATORS_AVG'].fillna(-1,inplace=True)
train_df['ELEVATORS_MODE'].fillna(-1,inplace=True)

train_df['APARTMENTS_MEDI'].fillna(-1,inplace=True)
train_df['APARTMENTS_AVG'].fillna(-1,inplace=True)
train_df['APARTMENTS_MODE'].fillna(-1,inplace=True)

train_df['ENTRANCES_MEDI'].fillna(-1,inplace=True)
train_df['ENTRANCES_AVG'].fillna(-1,inplace=True)
train_df['ENTRANCES_MODE'].fillna(-1,inplace=True)

path = '/Users/ferdinand/Desktop/data'
file_name1 = os.path.join(path,'bureau.csv')
bureau_df = pd.read_csv(file_name1)
bureau_df.shape

train_df['SK_ID_CURR'].nunique()
client_ids = list(train_df['SK_ID_CURR'].unique())
temp_df1 = train_df[train_df['SK_ID_CURR'].isin(client_ids[:2])]
print(temp_df1.shape)

temp_df2 = bureau_df[bureau_df['SK_ID_CURR'].isin(client_ids[:2])]
temp_df2.shape
temp_df1.SK_ID_CURR.unique()
temp_df1 = temp_df1.merge(temp_df2,on='SK_ID_CURR',how='left')
print(temp_df1.shape)
print(temp_df1)















