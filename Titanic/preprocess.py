'''
Author: your name
Date: 2021-12-31 16:50:54
LastEditTime: 2021-12-31 17:04:54
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /titanic-new/preprocess.py
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import random
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_dataset():
    train = pd.read_csv('input/titanic/train.csv')
    test_x = pd.read_csv('input/titanic/test.csv')
    test_y = pd.read_csv('input/titanic/ground_truth.csv')

    new_train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
    new_test = pd.merge(test_x, test_y, on='PassengerId')
    new_test = new_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
    # new_test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

    new_train['Age'].fillna(new_train['Age'].mean(), inplace = True)
    new_train['Embarked'].fillna('S', inplace = True)
    new_test['Age'].fillna(new_train['Age'].mean(), inplace = True)
    new_test['Fare'].fillna(new_train['Fare'].mean(), inplace = True)

    train_x = new_train.drop('Survived',  axis = 1)
    train_y = new_train['Survived']
    test_x = new_test.drop('Survived',  axis = 1)
    test_y = new_test['Survived']

    categorical_features = ['Sex', 'Embarked', 'Pclass']

    other_features = ['Age', 'Fare', 'SibSp', 'Parch']

    train_X_class_info = train_x[categorical_features]
    test_X_class_info = test_x[categorical_features]

    train_X_class_info = pd.get_dummies(train_X_class_info, columns=categorical_features, drop_first=False)
    test_X_class_info = pd.get_dummies(test_X_class_info, columns=categorical_features, drop_first=False)

    onehotencode = OneHotEncoder()
    standard_encode = StandardScaler()

    transformer = ColumnTransformer([('Encoder', onehotencode, categorical_features), ('scaler', standard_encode, other_features)], remainder = 'passthrough')
    encoded_x = transformer.fit_transform(train_x)
    test_x = transformer.fit_transform(test_x)
    
    # smote = SMOTE()
    # x_smote, y_smote = smote.fit_resample(encoded_x, train_y)

    train_X_class_info.to_csv('train_titanic_sample_class.csv', index=False)
    test_X_class_info.to_csv('test_titanic_sample_class.csv', index=False)

    return encoded_x, train_y, test_x, test_y


if __name__ == '__main__':
    prepare_dataset()