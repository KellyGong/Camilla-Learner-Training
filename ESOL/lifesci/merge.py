import pandas as pd
models = ['GCN_canonical_ESOL','GCN_attentivefp_ESOL','GAT_canonical_ESOL','GAT_attentivefp_ESOL','Weave_canonical_ESOL','Weave_attentivefp_ESOL','MPNN_canonical_ESOL','MPNN_attentivefp_ESOL','AttentiveFP_canonical_ESOL','AttentiveFP_attentivefp_ESOL','gin_supervised_contextpred_ESOL','gin_supervised_infomax_ESOL','gin_supervised_edgepred_ESOL','gin_supervised_masking_ESOL']
dfs = []
for model in models:
    df_tmp = pd.read_csv('./result/'+model+'_to_merge.csv',usecols=["id", model])
    dfs.append(df_tmp)
len_ = len(models)
df = dfs[0]
for i in range(1,len_):
    df = pd.merge(df,dfs[i],how='inner',on='id')
df.to_csv('./result/merged.csv')
print(df.head())
