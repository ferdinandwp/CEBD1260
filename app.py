# This is master script
# Import library
import numpy as np  
import pandas as pd
import os
import glob
import sys
sys.path.append('..')
from processed_script.ipay import preprocessing


my_df = preprocessing()
print(my_df.head())
# os.system('python processed_script/ipay.py')







