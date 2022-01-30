import torch
import numpy
import pandas as pd
import random
import numpy as np
from dgllife.data import ESOL
from dgllife.model import load_pretrained
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer,AttentiveFPAtomFeaturizer,CanonicalBondFeaturizer,AttentiveFPBondFeaturizer
from math import sqrt
from copy import deepcopy
from tqdm import tqdm
from functools import partial

df = pd.read_csv('./result/merged.csv',index_col=0)
ids =  list(df['id'])
dataset = ESOL(smiles_to_bigraph, AttentiveFPAtomFeaturizer())
labels = []
for id_ in ids:
    s,g,label = dataset[id_]
    labels.append(label[0].detach().numpy())
df['label'] = labels
df.to_csv('./result/merged_label.csv')
