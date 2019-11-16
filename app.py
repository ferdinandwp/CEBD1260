# This is master script
# Import library
import numpy as np  
import pandas as pd
import os
import glob
import sys
sys.path.append('..')
from processed_script.ipay import preprocessing_ipay
from processed_script.cc_balance import preprocessing_cc_balance
from processed_script.prev_app import preprocessing_prev_app
from processed_script.POS_CASH_balance import preprocessing_POS_CASH_balance
from processed_script.main import preprocessing_main

# ipay_df = preprocessing_ipay()
# cc_balance_df = preprocessing_cc_balance()
# prev_app_df = preprocessing_prev_app()
# POS_CASH_balance_df = preprocessing_POS_CASH_balance()
main_df = preprocessing_main()

print(main_df.head())
print(main_df.shape)


