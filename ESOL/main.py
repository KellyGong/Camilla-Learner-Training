import torch
import pandas as pd
import random
import numpy as np
from dgllife.data import ESOL
from dgllife.model import load_pretrained
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer,AttentiveFPAtomFeaturizer,CanonicalBondFeaturizer,AttentiveFPBondFeaturizer
from math import sqrt
from copy import deepcopy
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from functools import partial
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor


def load_data(num,feat_size):
    x_train = np.load('./data/'+str(feat_size)+'/train_x_'+str(num)+'.npy',allow_pickle=True)
    y_train = np.load('./data/'+str(feat_size)+'/train_y_'+str(num)+'.npy',allow_pickle=True)
    x_test = np.load('./data/'+str(feat_size)+'/test_x_'+str(num)+'.npy',allow_pickle=True)
    y_test = np.load('./data/'+str(feat_size)+'/test_y_'+str(num)+'.npy',allow_pickle=True)
    index = np.load('./data/'+str(feat_size)+'/test_index_'+str(num)+'.npy',allow_pickle=True)
    return x_train,y_train,x_test,y_test,index









randseed=2021
np.random.seed(randseed)
torch.manual_seed(randseed)
torch.cuda.manual_seed_all(randseed)
random.seed(randseed)
torch.backends.cudnn.deterministic = True

models = {
    # 'LR': LinearRegression(),
    # 'Lasso': Lasso(random_state=randseed),
    # 'RF': RandomForestRegressor(random_state=randseed),
    # 'KRN': KNeighborsRegressor(),
    # 'GBR': GradientBoostingRegressor(random_state=randseed),
    # 'ETR':ExtraTreesRegressor(random_state=randseed),
    # 'Ridge':Ridge(random_state=randseed),
    # 'cb':CatBoostRegressor(random_seed=randseed,loss_function='RMSE'),
    'Ada' :AdaBoostRegressor(random_state = randseed,loss = 'square')
}








for feat_size in [256]:
    for model_name in models:
        model = models[model_name]
        dfs = []
        for i in tqdm(range(5)):
            x_train,y_train,x_test,y_test,index = load_data(i,feat_size)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            pred = pred.reshape(-1,1)
            pred = list(pred.reshape(-1))
            df = pd.DataFrame({'id':index,'prediction':pred})
            dfs.append(df)
        df = pd.concat(dfs)
        df.to_csv('./result/'+str(feat_size) + '/'+model_name+'.csv')

