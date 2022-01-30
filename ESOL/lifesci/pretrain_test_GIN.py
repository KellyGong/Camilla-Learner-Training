import torch
import numpy
import pandas as pd
import random
import numpy as np
from dgllife.data import ESOL
from dgllife.model import load_pretrained
from dgllife.utils import smiles_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from math import sqrt
from copy import deepcopy
from tqdm import tqdm
from functools import partial
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

        s,g,label = dataset[i]
        feats_n, feats_e = [g.ndata['atomic_number'],g.ndata['chirality_type']], [g.edata['bond_type'],g.edata['bond_direction_type']]

        if dataset.min_degree[i]>0:
            pred = model(g,list(feats_n),list(feats_e))[0][0].detach().numpy()
            label = label[0].detach().numpy()
            mae.append(abs(pred-label))
            mse.append((pred-label)*(pred-label))
            id_.append(i)
            preds.append(pred)
        else:
            cnt_dirty += 1

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
    dataset = ESOL(partial(smiles_to_bigraph, add_self_loop=True), PretrainAtomFeaturizer(), PretrainBondFeaturizer())
    models = ['gin_supervised_contextpred_ESOL','gin_supervised_infomax_ESOL','gin_supervised_edgepred_ESOL','gin_supervised_masking_ESOL']

    for model in models:
        to_save_df = test(dataset,model)
        to_save_df.to_csv('./result/'+model+'_to_merge.csv')
