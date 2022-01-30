import torch
import numpy
import pandas as pd
import random
import numpy as np
from dgllife.data import ESOL
from dgllife.model import load_pretrained
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer,AttentiveFPAtomFeaturizer
from math import sqrt
from copy import deepcopy
from tqdm import tqdm
def set_rand_seed(randseed):
    np.random.seed(randseed)
    torch.manual_seed(randseed)
    torch.cuda.manual_seed_all(randseed)
    random.seed(randseed)
    torch.backends.cudnn.deterministic = True



def test(dataset,model_name):
    model = load_pretrained(model_name)
    model.eval()
    mse,mae,id_,preds = [],[],[],[]
    cnt_dirty = 0
    len_ = len(dataset)
    for i in tqdm(range(len_)):
        if model_name.split('_')[0]=='GCN':
            s,g,label = dataset[i]
            feats = deepcopy(g.ndata).pop('h')
            if dataset.min_degree[i]>0:
                pred = model(g,feats)[0][0].detach().numpy()
                label = label[0].detach().numpy()
                print(pred)
                print(label)
                return
                mae.append(abs(pred-label))
                mse.append((pred-label)*(pred-label))
                id_.append(i)
                preds.append(pred)
            else:
                cnt_dirty += 1
        else:
            pass

    mse_g = sum(mse)/(len_-cnt_dirty)
    map_g = sum(mae)/(len_-cnt_dirty)
    id_.append(-1)
    preds.append(mse_g)
    id_.append(-2)
    preds.append(map_g)
    to_save_df = pd.DataFrame({'id':id_,model_name:preds})

    return to_save_df
            
if __name__=='__main__':
    set_rand_seed(2021)
    dataset_a = ESOL(smiles_to_bigraph, AttentiveFPAtomFeaturizer())
    dataset_c = ESOL(smiles_to_bigraph, CanonicalAtomFeaturizer())
    models = ['GCN_canonical_ESOL','GCN_attentivefp_ESOL']
    #models = ['GCN_canonical_ESOL','GCN_attentivefp_ESOL','GAT_canonical_ESOL','GAT_attentivefp_ESOL','Weave_canonical_ESOL','Weave_attentivefp_ESOL','MPNN_canonical_ESOL','MPNN_attentivefp_ESOL','AttentiveFP_canonical_ESOL','AttentiveFP_attentivefp_ESOL','gin_supervised_contextpred_ESOL','gin_supervised_infomax_ESOL','gin_supervised_edgepred_ESOL','gin_supervised_masking_ESOL']

    for model in models:
        if model.split('_')[1]=='canonical':
            to_save_df = test(dataset_c,model)
        else:
            to_save_df = test(dataset_a,model)
        to_save_df.to_csv('./result/'+model+'_to_merge.csv')