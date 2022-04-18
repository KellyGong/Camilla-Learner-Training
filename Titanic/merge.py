'''
Author: your name
Date: 2022-01-02 13:36:13
LastEditTime: 2022-01-26 10:38:44
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /titanic-new/merge.py
'''

from numpy.lib.npyio import load
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from utils import load_ndarray
import numpy as np


def check_exist(all_ndarray, one_ndarray):
    if len(all_ndarray) == 0:
        return False
    
    for ndarr in all_ndarray:
        if (ndarr == one_ndarray).all():
            return True
    
    return False


models = []
path = './result/model_prediction/'
all_df = pd.DataFrame()
test_df = pd.DataFrame()
files = os.listdir(path)
for model_type_file in files:
    model_type_path = path + model_type_file
    model_variants = os.listdir(model_type_path)
    model_type_numpy = []
    # print(f'{model_type_file}: # {len(model_variants)}')
    for model_variant in model_variants:
        if model_variant == 'aggregate.csv':
            continue
        model_variant_path = model_type_path + '/' + model_variant
        test_correct_data = load_ndarray(model_variant_path + '/test_correct.npy')
        y_correct_data = load_ndarray(model_variant_path + '/y_correct.npy')
        if not check_exist(model_type_numpy, test_correct_data):
            all_df[model_variant] = y_correct_data
            test_df[model_variant] = test_correct_data
            model_type_numpy.append(test_correct_data)
    
    model_accs = np.array([sum(model_response) / model_response.size for model_response in model_type_numpy])
    mean_accs = np.mean(model_accs)
    std_accs = np.std(model_accs)
    print(f'filterd {model_type_file}: # {len(model_type_numpy)}, mean acc: {mean_accs}, std acc: {std_accs}')
    

# test_df.to_csv('titanic.csv')
    
