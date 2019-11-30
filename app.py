# This is master script
# Import library
import numpy as np  
import pandas as pd
import os
import glob
import sys
sys.path.append('..')
# from processed_script.ipay import preprocessing_ipay
# from processed_script.cc_balance import preprocessing_cc_balance
# from processed_script.prev_app import preprocessing_prev_app
# from processed_script.POS_CASH_balance import preprocessing_POS_CASH_balance
from processed_script.main import preprocessing_main
# from processed_script.bureau import preprocessing_bureau


# ipay_df = preprocessing_ipay()
# cc_balance_df = preprocessing_cc_balance()
# prev_app_df = preprocessing_prev_app()
# POS_CASH_balance_df = preprocessing_POS_CASH_balance()
main_df = preprocessing_main()
# bureau_df = preprocessing_bureau()

# *************** Data Joining *************** #





# *************** Algorithm *************** #

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
kf = KFold(n_splits=5)


X_train = main_df.drop(['TARGET'],axis=1)
y_train = main_df['TARGET']

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
    # pred_val_y = model.predict_proba(va_X,num_iteration=model.best_iteration_)[:,1]
