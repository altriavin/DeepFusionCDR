import torch
import numpy as np
import pandas as pd
from os.path import splitext, basename
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset


def get_feature(batch_size, training):
    fea_CNV_file = './fea/cnv_fullgdsc.csv'
    fea_CNV = pd.read_csv(fea_CNV_file, header=0, index_col=0, sep=',')

    fea_meth_file = './fea/meth_fullgdsc.csv'
    fea_meth = pd.read_csv(fea_meth_file, header=0, index_col=0, sep=',')

    fea_mirna_file = './fea/mRNA_fullgdsc.csv'
    fea_mirna = pd.read_csv(fea_mirna_file, header=0, index_col=0, sep=',')

    fea_rna_file = './fea/mutation_fullgdsc.csv'
    fea_rna = pd.read_csv(fea_rna_file, header=0, index_col=0, sep=',')

    feature = np.concatenate((fea_CNV, fea_meth, fea_mirna, fea_rna), axis=1)
    minmaxscaler = MinMaxScaler()
    feature = minmaxscaler.fit_transform(feature)
    feature = torch.tensor(feature)

    dataloader = DataLoader(feature, batch_size=batch_size, shuffle=training)

    return dataloader



