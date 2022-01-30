import pandas as pd
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold

def set_rand_seed(randseed):
    np.random.seed(randseed)
    torch.manual_seed(randseed)
    torch.cuda.manual_seed_all(randseed)
    random.seed(randseed)
    torch.backends.cudnn.deterministic = True

def process_data():
    df = pd.read_csv('./data/diamonds.csv',index_col=0)
    X = df[['carat','cut','color','clarity','table','x','y','z']]
    y = df['price']
    X = pd.get_dummies(X, columns=['cut','color','clarity'],drop_first=True, prefix='') 
    kf = KFold(n_splits=5,shuffle=True,random_state=randseed)
    for k,(train,test) in enumerate(kf.split(X,y)):
        print (k)
        x_train=X.iloc[train]
        x_test=X.iloc[test]
        y_train=y.iloc[train]
        y_test=y.iloc[test]
        x_train.to_csv('./data/x_train_'+str(k)+'.csv')
        x_test.to_csv('./data/x_test_'+str(k)+'.csv')
        y_train.to_csv('./data/y_train_'+str(k)+'.csv')
        y_test.to_csv('./data/y_test_'+str(k)+'.csv')


if __name__=='__main__':
    set_rand_seed(2021)
    randseed=2021
    process_data()