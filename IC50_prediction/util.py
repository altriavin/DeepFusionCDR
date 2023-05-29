import math

import numpy as np
import pandas as pd
import torch
from model import MLP

import codecs
from subword_nmt.apply_bpe import BPE
from sklearn.model_selection import train_test_split


class Dataloader():
    def __init__(self, arg, cell_line_data_path, drug_data_path, IC50_data_path):
        self.cell_line_data_path = cell_line_data_path
        self.drug_data_path = drug_data_path
        self.IC50_data_path = IC50_data_path
        # self.batch_size = arg.batch_size
        self.drugname2id = {}
        self.id2drugname = {}
        self.cellname2id = {}
        self.id2cellname = {}

    def get_batch(self, data_x, data_y, batch_size):
        batch_x, batch_y = [], []
        len_data = len(data_x)
        batch_count = math.ceil(len_data / batch_size)

        perm = np.random.permutation(len_data)
        data_x, data_y = np.array(data_x), np.array(data_y)
        data_x = data_x[perm]
        data_y = data_y[perm]

        for i in range(batch_count - 1):
            if (i + 1) * batch_size <= len_data:
                batch_x.append(data_x[i * batch_size: (i + 1) * batch_size])
                batch_y.append(data_y[i * batch_size: (i + 1) * batch_size])
            else:
                batch_x.append(data_x[i * batch_size:])
                batch_y.append(data_y[i * batch_size:])

        return batch_x, batch_y

    def get_train_test_data(self):
        X, y = self.load_data_IC50()
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=5)

        return train_X, train_y, test_X, test_y

    def load_data_IC50(self):
        drug_data = self.load_data_drug()

        # print(f'drug_data: {drug_data}')

        cell_line_data = self.load_data_cell_line()

        IC50_file = pd.read_csv(self.IC50_data_path, header=None)

        X, y = [], []
        # count = 0
        for i in range(len(IC50_file.index)):
            drug_name = IC50_file.iloc[i, 0]
            cell_name = IC50_file.iloc[i, 2]
            IC50 = IC50_file.iloc[i, 4]

            cell_name = cell_name.replace("-", "").replace(".", "").lower()

            X.append([cell_line_data[cell_name], drug_data[drug_name]])
            y.append(IC50)
            # count += 1
        # print(f'count: {count}')
        return X, y

    '''
    return dict: dict[drugname -> id] = encode
    '''
    def load_data_drug(self):
        drug_smile_file = pd.read_csv(self.drug_data_path, header=None)
        drugname2smile = {}

        drug_name_count = 0
        for i in range(len(drug_smile_file.index)):
            # tmp = {}
            drug_name = drug_smile_file.iloc[i, 0]
            drug_smile = drug_smile_file.iloc[i, 2]

            # print(f'drug_name: {drug_name}')

            if drug_name not in self.drugname2id.keys():
                self.drugname2id[drug_name] = drug_name_count
                self.id2drugname[drug_name_count] = drug_name
                drug_name_count += 1

            drugname2smile[drug_name] = drug_smile
        drugname2id = self.drug2id_encoder(drugname2smile)
        # for key in drugname2smile.keys():
        #     drugname2smile[key] = list(self._drug2emb_encoder(drugname2smile[key]))

        return drugname2id

    def drug2id_encoder(self, drugname2smile):
        vocab_path = "./data/ESPF-info/drug_codes_chembl_freq_1500.txt"
        bpe_codes_drug = codecs.open(vocab_path)
        dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
        for drug_name in drugname2smile.keys():
            drugname2smile[drug_name] = dbpe.process_line(drugname2smile[drug_name]).split()

        sub_chem2id = {}
        count = 0
        for drug_name in drugname2smile.keys():
            for sub_chem in drugname2smile[drug_name]:
                if sub_chem not in sub_chem2id.keys():
                    sub_chem2id[sub_chem] = count
                    count += 1
        for drug_name in drugname2smile.keys():
            drugname2smile[drug_name] = [sub_chem2id[x] for x in drugname2smile[drug_name]]
        # print(f'sub_chem_count = {count}')
        return drugname2smile

    '''
    return: dict[cell_name -> id] = [[],[],[],[]]四个组学数据
    '''
    def load_data_cell_line(self):
        data_all_feature = []
        for sub_path in self.cell_line_data_path:
            file_content = np.array(pd.read_csv(sub_path))
            cell_name_count = 0
            tmp_result = []
            for i in file_content:
                cell_name = i[0]
                if cell_name not in self.cellname2id.keys():
                    # print(f'cell_name: {cell_name}')
                    self.cellname2id[cell_name.lower()] = cell_name_count
                    self.id2cellname[cell_name_count] = cell_name.lower()
                    cell_name_count += 1
                tmp_result.append([int(i[x]) for x in range(1, len(i))])
            data_all_feature.append(tmp_result)

        data = {}
        for j in range(len(data_all_feature[0])):
            tmp = []
            for i in range(len(data_all_feature)):
                tmp.append(data_all_feature[i][j])
            data[self.id2cellname[j]] = tmp

        return data

    def _drug2emb_encoder(self, smile):
        # vocab_path = "{}/ESPF/drug_codes_chembl_freq_1500.txt".format(self.vocab_dir)
        # sub_csv = pd.read_csv("{}/ESPF/subword_units_map_chembl_freq_1500.csv".format(self.vocab_dir))
        vocab_path = "./data/ESPF-info/drug_codes_chembl_freq_1500.txt"
        sub_csv = pd.read_csv("./data/ESPF-info/subword_units_map_chembl_freq_1500.csv")

        bpe_codes_drug = codecs.open(vocab_path)
        dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

        idx2word_d = sub_csv['index'].values
        words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

        max_d = 50
        t1 = dbpe.process_line(smile).split()  # split
        try:
            i1 = np.asarray([words2idx_d[i] for i in t1])  # index
        except:
            i1 = np.array([0])

        # print(f'i1: {i1}')

        return i1

        # l = len(i1)
        # if l < max_d:
        #     i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        #     input_mask = ([1] * l) + ([0] * (max_d - l))
        # else:
        #     i = i1[:max_d]
        #     input_mask = [1] * max_d
        # print(f'i = {i}, input_mask: {input_mask}')
        # return i, np.asarray(input_mask)

