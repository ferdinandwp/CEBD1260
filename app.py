# This is master script
# Import library
import numpy as np  
import pandas as pd
import os
import glob
import sys
sys.path.append('..')
from processed_script.main import preprocessing_main
from processed_script.prev_app import preprocessing_prev_app
from processed_script.cc_balance import preprocessing_cc_balance
from processed_script.bureau import preprocessing_bureau
from processed_script.POS_CASH_balance import preprocessing_POS_CASH_balance
from processed_script.ipay import preprocessing_ipay
# from processed_script.br_balance import preprocessing_bureau_balance


main_df = preprocessing_main()
prev_app_df = preprocessing_prev_app()
cc_balance_df = preprocessing_cc_balance()
bureau_df = preprocessing_bureau()
POS_CASH_balance_df = preprocessing_POS_CASH_balance()
ipay_df = preprocessing_ipay()
# bureau_balance_df = preprocessing_bureau_balance()

# *************** Data Joining *************** #

# 1. Join Main table to previous applicaltion table
# Find common ids to main tables
main_ids = list(main_df['MAIN_SK_ID_CURR'].unique())
prev_ids = list(prev_app_df['SK_ID_CURR'].unique())
common1_ids = set(main_ids).intersection(set(prev_ids)) #see how many ids are common to main tables

filtered_prev_df = prev_app_df.loc[prev_app_df.SK_ID_CURR.isin(main_ids)]
agg_prev_dict = {'AMT_ANNUITY':['mean'],
                 'AMT_APPLICATION':['mean'],
                 'AMT_CREDIT':['mean'],
                 'AMT_DOWN_PAYMENT':['mean'],
                 'AMT_GOODS_PRICE':['mean'],
                 'HOUR_APPR_PROCESS_START':['mean'],
                 'NFLAG_LAST_APPL_IN_DAY':['mean'],
                 'RATE_DOWN_PAYMENT':['mean'],
                 'RATE_INTEREST_PRIMARY':['mean'],
                 'RATE_INTEREST_PRIVILEGED':['mean'],
                 'DAYS_DECISION':['mean'],
                 'SELLERPLACE_AREA':['mean'],
                 'CNT_PAYMENT':['mean'],
                 'DAYS_FIRST_DRAWING':['mean'],
                 'DAYS_FIRST_DUE':['mean'],
                 'DAYS_LAST_DUE_1ST_VERSION':['mean'],
                 'DAYS_LAST_DUE':['mean'],
                 'DAYS_TERMINATION':['mean'],
                 'NFLAG_INSURED_ON_APPROVAL':['mean']
                }

prev_num_df = filtered_prev_df.groupby('SK_ID_CURR').agg(agg_prev_dict)
agg_prev_df = prev_num_df.reset_index()
agg_prev_df.columns = ['PREV_APP_{}_{}'.format(x[0],x[1]) for x in agg_prev_df.columns.tolist()]
agg_prev_df.rename({'PREV_APP_SK_ID_CURR_':'PREV_APP_SK_ID_CURR'},axis=1,inplace=True)

main_df.rename({'MAIN_SK_ID_CURR':'SK_ID_CURR'},axis=1,inplace=True)
agg_prev_df.rename({'PREV_APP_SK_ID_CURR':'SK_ID_CURR'},axis=1,inplace=True)

final_temp1_df = main_df.merge(agg_prev_df, on='SK_ID_CURR', how='left')

# 2. Join Main table to credit card balance table
cc_ids = list(cc_balance_df['SK_ID_CURR'].unique())
common2_ids = set(main_ids).intersection(set(cc_ids)) #see how many ids are common to main tables

filtered_cc_df = cc_balance_df.loc[cc_balance_df.SK_ID_CURR.isin(main_ids)]
# aggregate cc dataset
agg_cc_dict = {'MONTHS_BALANCE':['mean'],
             'AMT_BALANCE':['mean'],
             'AMT_CREDIT_LIMIT_ACTUAL':['mean'],
             'AMT_DRAWINGS_ATM_CURRENT':['mean'],
             'AMT_DRAWINGS_CURRENT':['mean'],
             'AMT_DRAWINGS_OTHER_CURRENT':['mean'],
             'AMT_DRAWINGS_POS_CURRENT':['mean'],
             'AMT_INST_MIN_REGULARITY':['mean'],
             'AMT_PAYMENT_CURRENT':['mean'],
             'AMT_PAYMENT_TOTAL_CURRENT':['mean'],
             'AMT_RECEIVABLE_PRINCIPAL':['mean'],
             'AMT_RECIVABLE':['mean'],
             'AMT_TOTAL_RECEIVABLE':['mean'],
             'CNT_DRAWINGS_ATM_CURRENT':['mean'],
             'CNT_DRAWINGS_CURRENT':['mean'],
             'CNT_DRAWINGS_OTHER_CURRENT':['mean'],
             'CNT_DRAWINGS_POS_CURRENT':['mean'],
             'CNT_INSTALMENT_MATURE_CUM':['mean']
             }
cc_num_df = filtered_cc_df.groupby('SK_ID_CURR').agg(agg_cc_dict)
agg_cc_df = cc_num_df.reset_index()
agg_cc_df.columns = ['CC_BAL_{}_{}'.format(x[0],x[1]) for x in agg_cc_df.columns.tolist()]
agg_cc_df.rename({'CC_BAL_SK_ID_CURR_':'SK_ID_CURR'},axis=1,inplace=True)

# connect main to cc balance
final_temp2_df = final_temp1_df.merge(agg_cc_df, on='SK_ID_CURR', how='left')


# 3. Join Main table to bureau table
bureau_ids = list(bureau_df['SK_ID_CURR'].unique())
common3_ids = set(main_ids).intersection(set(bureau_ids)) #see how many ids are common to main tables
filtered_bureau_df = bureau_df.loc[bureau_df.SK_ID_CURR.isin(main_ids)]

agg_bureau_dict = {
            'DAYS_CREDIT': ['mean'],
            'CREDIT_DAY_OVERDUE': ['mean'],
            'DAYS_CREDIT_ENDDATE': ['mean'],
            'DAYS_ENDDATE_FACT': ['mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'CNT_CREDIT_PROLONG': ['mean'],
            'AMT_CREDIT_SUM': ['mean'],
            'AMT_CREDIT_SUM_DEBT': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'AMT_ANNUITY': ['mean']
            }

bureau_num_df = filtered_bureau_df.groupby('SK_ID_CURR').agg(agg_bureau_dict)
agg_bureau_df = bureau_num_df.reset_index()
agg_bureau_df.columns = ['BUR_{}_{}'.format(x[0],x[1]) for x in agg_bureau_df.columns.tolist()]
agg_bureau_df.rename({'BUR_SK_ID_CURR_':'SK_ID_CURR'},axis=1,inplace=True)

# connect main to bureau balance
final_temp3_df = final_temp2_df.merge(agg_bureau_df, on='SK_ID_CURR', how='left')

# 4. Join Main table to POS Cash Balance table
POS_ids = list(POS_CASH_balance_df['SK_ID_CURR'].unique())
common4_ids = set(main_ids).intersection(set(POS_ids)) #see how many ids are common to main tables
filtered_POS_df = POS_CASH_balance_df.loc[POS_CASH_balance_df.SK_ID_CURR.isin(main_ids)]

agg_POS_dict = {'MONTHS_BALANCE': ['mean'],
                'CNT_INSTALMENT': ['mean'],
                'CNT_INSTALMENT_FUTURE': ['mean'],
                'SK_DPD': ['mean'],
                'SK_DPD_DEF': ['mean']
            }

POS_num_df = filtered_POS_df.groupby('SK_ID_CURR').agg(agg_POS_dict)
agg_POS_df = POS_num_df.reset_index()
agg_POS_df.columns = ['POS_{}_{}'.format(x[0],x[1]) for x in agg_POS_df.columns.tolist()]
agg_POS_df.rename({'POS_SK_ID_CURR_':'SK_ID_CURR'},axis=1,inplace=True)

# connect main to bureau balance
final_temp4_df = final_temp3_df.merge(agg_POS_df, on='SK_ID_CURR', how='left')

# 5. Join Main table to installement payment table
ipay_ids = list(ipay_df['SK_ID_CURR'].unique())
common5_ids = set(main_ids).intersection(set(ipay_ids)) #see how many ids are common to main tables
filtered_ipay_df = ipay_df.loc[ipay_df.SK_ID_CURR.isin(main_ids)]

agg_ipay_dict = {'NUM_INSTALMENT_VERSION': ['mean'],
                'NUM_INSTALMENT_NUMBER': ['mean'],
                'DAYS_INSTALMENT': ['mean'],
                'DAYS_ENTRY_PAYMENT': ['mean'],
                'AMT_INSTALMENT': ['mean'],
                'AMT_PAYMENT': ['mean']
                }

ipay_num_df = filtered_ipay_df.groupby('SK_ID_CURR').agg(agg_ipay_dict)
agg_ipay_df = ipay_num_df.reset_index()
agg_ipay_df.columns = ['IPAY_{}_{}'.format(x[0],x[1]) for x in agg_ipay_df.columns.tolist()]
agg_ipay_df.rename({'IPAY_SK_ID_CURR_':'SK_ID_CURR'},axis=1,inplace=True)

# connect main to bureau balance
final_temp5_df = final_temp4_df.merge(agg_ipay_df, on='SK_ID_CURR', how='left')

# 6. Final dataset
final_df = final_temp5_df
final_df.fillna(0)
print(final_df.shape)
# *************** Algorithm *************** #

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
kf = KFold(n_splits=5)


X_train = final_df.drop(['TARGET'],axis=1)
y_train = final_df['TARGET']

# Split dataframe by index
for i,(tr_idx, val_idx) in enumerate(kf.split(X_train,y_train)):
    print('Fold :{}'.format(i))
    tr_X = X_train.loc[tr_idx]  # training for this loop
    tr_y = y_train[tr_idx] 
    val_X = X_train.loc[val_idx]# validation data for this loop
    val_y = y_train[val_idx]
    

    model = LGBMClassifier(
                n_jobs=4,
                n_estimators=100000,
                boost_from_average='false',
                learning_rate=0.01,
                num_leaves=64,
                num_threads=4,
                max_depth=10,
                feature_fraction = 0.7,
                bagging_freq = 5,
                bagging_fraction = 0.5,
                silent=-1,
                verbose=-1
                )
    model.fit(tr_X, tr_y, eval_set=[(tr_X, tr_y), (val_X, val_y)], eval_metric = 'binary_logloss', verbose=100, early_stopping_rounds= 200)
    pred_val_y = model.predict_proba(val_X,num_iteration=model.best_iteration_)[:,1]
