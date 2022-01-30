import numpy as np
from dgllife.data import ESOL
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


smiles, labels, FP= [],[],[]

dataset = ESOL()
len_ = len(dataset)

for i in range(len_):
    s_,_,label_ = dataset[i]
    smiles.append(s_)
    m = Chem.MolFromSmiles(s_)
    fp_ = np.array(AllChem.GetMorganFingerprintAsBitVect(m,2, nBits=256))
    FP.append(fp_)
    labels.append(label_)

randseed = 2021


kf = KFold(n_splits=5,shuffle=False)

for k,(train,test) in enumerate(kf.split(smiles,labels)):
    X_train, Y_train, X_test, Y_test = [],[],[],[]
    for i in train:
        X_train.append(FP[i])
        Y_train.append(labels[i].numpy()[0])
    for i in test:
        X_test.append(FP[i])
        Y_test.append(labels[i].numpy()[0])       
    index_train = train
    index_test = test

    np.save('./data/256/train_x_'+str(k)+'.npy',X_train)
    np.save('./data/256/train_y_'+str(k)+'.npy',Y_train)
    np.save('./data/256/test_x_'+str(k)+'.npy',X_test)
    np.save('./data/256/test_y_'+str(k)+'.npy',Y_test)
    np.save('./data/256/train_index_'+str(k)+'.npy',index_train)
    np.save('./data/256/test_index_'+str(k)+'.npy',index_test)
    


    