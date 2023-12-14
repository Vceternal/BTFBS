import torch
from torch.utils.data import Dataset
import numpy as np
CHARISOSMISET = ['A',   'G',   'C'   ,'T']

CHARISOSMILEN = 4

CHARPROTSET = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y']
CHARPROTLEN = 21


def label_smiles(line, smi_ch_ind, MAX_SEQ_LEN=100):
    fea=[]
    tem_vec=[]
    X = np.zeros(100, dtype=np.int64())
    seq=line
    for i in range(len(seq)):
        if seq[i] =='A':
            tem_vec = 0*100+i
        elif seq[i]=='G':
            tem_vec = 1*100+i
        elif seq[i]=='C':
            tem_vec = 2*100+i
        elif seq[i]=='T' or seq[i]=='U':
            tem_vec = 3*100+i
        
        #tem_vec = tem_vec +[i]
        fea.append(tem_vec)
        #fea.append(tem_vec)

    if len(fea)<100:
        for i in range(100-len(fea)):
            fea.append(0)
    #fea=np.array(fea,np.int64())
    for i in range(len(fea)):
        X[i]=fea[i]
    #print(X)
    #print(len(X))

    return X


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=600):
    fea=[]
    tem_vec=[]
    X = np.zeros(600, dtype=np.int64())
    seq=line
    for i in range(len(seq)):
        if seq[i] =='A':
            tem_vec = 0*600+i
        elif seq[i]=='C':
            tem_vec = 1*600+i
        elif seq[i]=='D':
            tem_vec = 2*600+i
        elif seq[i]=='E' or seq[i]=='U':
            tem_vec = 3*600+i
        elif seq[i]=='F':
            tem_vec = 4*600+i
        elif seq[i]=='G':
            tem_vec = 5*600+i
        elif seq[i]=='H':
            tem_vec = 6*600+i
        elif seq[i]=='I':
            tem_vec = 7*600+i
        elif seq[i]=='K':
            tem_vec = 8*600+i
        elif seq[i]=='L':
            tem_vec = 9*600+i
        elif seq[i]=='M' or seq[i]=='O':
            tem_vec = 10*600+i
        elif seq[i]=='N':
            tem_vec = 11*600+i
        elif seq[i]=='P':
            tem_vec = 12*600+i
        elif seq[i]=='Q':
            tem_vec = 13*600+i
        elif seq[i]=='R':
            tem_vec = 14*600+i
        elif seq[i]=='S':
            tem_vec = 15*600+i
        elif seq[i]=='T':
            tem_vec = 16*600+i
        elif seq[i]=='V':
            tem_vec = 17*600+i
        elif seq[i]=='W':
            tem_vec = 18*600+i
        elif seq[i]=='X' or seq[i]=='B' or seq[i]=='Z':
            tem_vec = 19*600+i
        elif seq[i]=='Y':
            tem_vec = 20*600+i
        #tem_vec = tem_vec +[i]
        fea.append(tem_vec)
        #fea.append(tem_vec)

    if len(fea)<600:
        for i in range(600-len(fea)):
            fea.append(0)
    #fea=np.array(fea,np.int64())
    for i in range(len(fea)):
        X[i]=fea[i]
    #print(fea)
    #print(X)
    #print(len(fea))

    return X



class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)


def collate_fn(batch_data):
    N = len(batch_data)
    drug_ids, protein_ids = [], []
    compound_max = 100
    protein_max = 600
    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)
    for i, pair in enumerate(batch_data):
        pair = pair.strip().split()
        drug_id, protein_id, compoundstr, proteinstr, label = pair[-5], pair[-4], pair[-3], pair[-2], pair[-1]
        drug_ids.append(drug_id)
        protein_ids.append(protein_id)
        compoundint = torch.from_numpy(label_smiles(
            compoundstr, CHARISOSMISET, compound_max))
        compound_new[i] = compoundint
        proteinint = torch.from_numpy(label_sequence(
            proteinstr, CHARPROTSET, protein_max))
        protein_new[i] = proteinint
        label = float(label)
        labels_new[i] = np.int(label)
    return (compound_new, protein_new, labels_new)
