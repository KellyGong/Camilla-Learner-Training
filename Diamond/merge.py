import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error
import os




# models = ['LR']

# for i in range(9):
#     models.append('ETR_'+str(i))
# for i in range(16):
#     models.append('GBR_'+str(i))
# for i in range(12):
#     models.append('KNR_'+str(i))
# for i in range(9):
#     models.append('Lasso_'+str(i))
# for i in range(9):
#     models.append('lgbm_'+str(i))
# for i in range(18):
#     models.append('RF_'+str(i))
# for i in range(27):
#     models.append('Ridge_'+str(i))
# for i in range(1,4):
#     models.append('SVR_'+str(i))
models = []
files = os.listdir('./result/y_trans/')
for f in files:
    if f.split('.')[-1]=='csv':
        models.append(os.path.splitext(f)[0])


dfs = []
for model in models:
    df_tmp = pd.read_csv('./result/y_trans/'+ model + '.csv',index_col=False,usecols=["id",'prediction'])
    df_tmp.columns = ['id',model]
    dfs.append(df_tmp)

df = dfs[0]
for i in range(1,len(dfs)):
    df = pd.merge(df,dfs[i],how='inner',on='id')
df = df.sort_values('id')
label_df = pd.read_csv('label.csv',usecols=["id",'label'])
label_df.columns = ['id','label']
df = pd.merge(df,label_df,how='inner',on='id')
mae,mse = [-1],[-2]
for model in models:
    mae.append(mean_absolute_error(df[model],df['label']))
    mse.append(mean_squared_error(df[model],df['label']))

mae.append(-1)
mse.append(-1)
mae = pd.DataFrame([mae],columns = df.columns)
mse = pd.DataFrame([mse],columns = df.columns)

df = df.append(mae,ignore_index = False)
df = df.append(mse,ignore_index = False)
print(df)
df.to_csv('./result/result.csv',index = False)
