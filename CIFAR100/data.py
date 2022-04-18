'''
Author: your name
Date: 2021-12-28 19:41:57
LastEditTime: 2022-01-07 16:01:30
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /pytorch-cifar100/data.py
'''

import torch
import torchvision.utils
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def load_response_matrix(path='checkpoint/cifar100_balanced'):
    p = Path(path)
    response_files = list(p.glob('**/test_correct.npy'))
    model_name_list = []
    response_matrix = []
    print(F'--------MODEL NUM: {len(response_files)}------------')
    # FIXME
    for response_file in response_files:
        model_name = response_file.parts[-3]
        model_name_list.append(model_name)
        response_matrix.append(np.load(response_file))
    response_matrix = np.vstack(response_matrix)

    # response_tuple
    rows, cols = np.where(response_matrix >= 0)
    responses = response_matrix[rows, cols]
    response_tuple = [(row, col, response) for (row, col, response) in zip(rows, cols, responses)]

    return response_matrix, response_tuple, model_name_list


def save_csv(response_matrix, model_name_list, path='test_cifar100.csv'):
    df = pd.DataFrame()
    data_id = [i+1 for i in range(response_matrix.shape[-1])]

    df['id'] = data_id
    for model_name, response in zip(model_name_list, response_matrix):
        df[model_name] = response
    
    df.to_csv(path, index=False)


if __name__ == '__main__':
    response_matrix, response_tuple_list, model_name_list = load_response_matrix()
    from collections import defaultdict
    model_num = defaultdict(int)
    for model_name in model_name_list:
        model_num[model_name] += 1
    print(len(set(model_name_list)))
    save_csv(response_matrix, model_name_list)
    print('yes')

