import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import random
from copy import deepcopy
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
import argparse
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
randseed=2021
np.random.seed(randseed)
random.seed(randseed)
def load_data(num):
    x_train = pd.read_csv('./data/x_train_'+str(num)+'.csv',index_col=0)
    y_train = pd.read_csv('./data/y_train_'+str(num)+'.csv',index_col=0)
    x_test = pd.read_csv('./data/x_test_'+str(num)+'.csv',index_col=0)
    y_test = pd.read_csv('./data/y_test_'+str(num)+'.csv',index_col=0)

    return x_train,y_train,x_test,y_test

models = {
'SVR_1':SVR(kernel='rbf'),
'SVR_2':SVR(kernel='poly'),
'SVR_3':SVR(kernel='linear'),
'SVR_3':SVR(kernel='sigmoid'),
'LR':LinearRegression()
'Lasso':Lasso(random_state=randseed)
}


for msf in [2,4,8]:
    for mss in [2,4,8]:
        model_name = 'RF_msf_' + str(msf) + '_mss_' + str(mss)
        models[model_name] = DecisionTreeRegressor(min_samples_leaf=msf,min_samples_split=mss,random_state=randseed)

for md in [2,4,8]:
    for ne in [500,1000,2000]:
        model_name = 'lgbm_md_' + str(md) + '_ne_' + str(ne)
        models[model_name] = LGBMRegressor(learning_rate=0.05, max_depth=md, n_estimators=ne,random_state=randseed)

for mf in [2,4,8]:
    for mss in [4,8,10]:
        for ne in [200,500]:
            model_name = 'RF_mf_' + str(mf) + '_mss_' + str(mss) + '_ne_' + str(ne)
            models[model_name] = RandomForestRegressor(max_depth=6,max_features=mf,min_samples_split=mss, n_estimators=ne,random_state=randseed)

for l_s in [10,20,30,50]:
    for n_n in [3,5,7]:
        model_name = 'KNR_ls_' + str(l_s) + '_nn_' + str(n_n)
        models[model_name] = KNeighborsRegressor(leaf_size=l_s,n_neighbors=n_n)

for m_s_s in [2,4,8,10]:
    for n_e in [100,200,300,500]:
        model_name = 'GBR_mss_' + str(m_s_s) + '_ne_' + str(n_e)
        models[model_name] = GradientBoostingRegressor(min_samples_split=m_s_s,n_estimators=n_e,random_state=randseed)

for m_d in [2,3,4]:
    for m_f in [2,4,8]:
        model_name = 'ETR_md_' + str(m_d) + '_mf_' + str(m_f)
        models[model_name] = ExtraTreesRegressor(max_depth=m_d,n_estimators=200,max_features=m_f,min_samples_split=10,random_state=randseed)

for mi in [200,500,1000]:
    for t in [0.0001,0.001,0.01]:
        model_name = 'Lasso_mi_' + str(mi) + '_t_' + str(t)
        models[model_name] = Lasso(max_iter=mi,tol=t,random_state=randseed)

model_name = 'Ridge'
models[model_name] = Ridge(alpha=0.5,max_iter=500,tol=0.001,random_state=randseed)



for eta in [0.03,0.1,0.3]:
    for mcw in [1,5,10]:
        for l in [0.5,1,2]:
            model_name = 'xgb_eta_' + str(eta) + '_mcw_' + str(mcw) + '_l_' + str(l)
            models[model_name] = xgb.XGBRegressor(
                        learning_rate=eta, 
                        objective='reg:squarederror', 
                        reg_lambda=l,
                        min_child_weight=mcw,
                        seed=randseed)

for ne in [50,100,500]:
    for lr in [0.03,0.3,1.0]:
        for loss_ in ['square','linear']:
            model_name = 'Ada_ne_' + str(ne) + '_lr_' + str(lr) + '_loss_' + loss_
            models[model_name] = AdaBoostRegressor(
                n_estimators = ne,
                learning_rate = lr,
                random_state = randseed,
                loss = loss_
        )

for hls in [(32,),(64,),(32,32)]:
    for a in [0.001,0.01,0.1]:
        for lr in [0.0001,0.001,0.01]:
            model_name = 'nn_hls_'+ str(hls) + '_alpha_' + str(a) + '_lr_' + str(lr)
            models[model_name] = MLPRegressor(
                hidden_layer_sizes = hls,
                alpha = a,
                learning_rate = 'constant',
                learning_rate_init = lr,
                random_state = randseed
            )

for lr in [0.003,0.03,0.1]:
    for l2reg in [0.3,1.0,3.0]:
        for d in [4,7,10]:
            model_name = 'cb_lr_' + str(lr) + '_l2reg_' + str(l2reg) + '_depth_' + str(d)
            models[model_name] = CatBoostRegressor(
                random_seed=randseed,
                learning_rate=lr,
                l2_leaf_reg = l2reg,
                depth = d,
                loss_function='RMSE'
            )




for model_name in tqdm(models):
    model = models[model_name]
    dfs = []
    for i in range(5):
        x_train,y_train,x_test,y_test = load_data(i)
        index = list(x_test.index)
        sc_X = StandardScaler()
        # Add sc_Y in y_trans
        sc_Y = StandardScaler()
        x_train = sc_X.fit_transform(x_train)
        y_train = sc_Y.fit_transform(y_train)
        x_test = sc_X.transform(x_test)
        
        model.fit(x_train, y_train.ravel())
        pred = model.predict(x_test)
        pred = pred.reshape(-1,1)
        pred = sc_Y.inverse_transform(pred)
        pred = list(pred.reshape(-1))
        df = pd.DataFrame({'id':index,'prediction':pred})
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv('./result/y_trans/'+model_name+'.csv')



