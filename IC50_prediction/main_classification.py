import argparse
import copy

import numpy as np
import pandas as pd
import os, sys
import pandas as pd
# import pubchempy as pcp
import pickle

import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, roc_curve, auc, \
    precision_recall_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
from lifelines.utils import concordance_index

from transformer.data_utils import load_smiles_data, construct_loader
from transformer.transformer import make_model

class new_mlp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(new_mlp, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, data):
        out = self.fc(data)
        return out


class mymodel(nn.Module):
    def __init__(self, drug_model, cell_line_model):
        super(mymodel, self).__init__()
        self.drug_emb_dim = 128
        self.gene_emb_dim = 128
        self.drug_model = drug_model
        self.cell_line_model = cell_line_model
        self.predictor = nn.Sequential(
            nn.Linear(self.drug_emb_dim + self.gene_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, cell_line_data, node_features, adjacency_matrix, distance_matrix):
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        drug_emb = self.drug_model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)

        mean = torch.mean(cell_line_data, dim=0)
        std = torch.std(cell_line_data, dim=0)

        eps = 1e-6
        std += eps

        cell_line_data = (cell_line_data - mean) / std

        cell_line_emb = self.cell_line_model(cell_line_data)
        emb = torch.cat((drug_emb, cell_line_emb), 1)
        score = self.predictor(emb)
        return score, emb


def train_test(drug_model_params, train_loader, test_loader, arg, celllinefeature, train_rna,
                                    test_rna, mlp_dim):
    drug_model = make_model(**drug_model_params)
    gene_model = new_mlp(mlp_dim, 128).to(arg.device)
    model = mymodel(drug_model, gene_model).to(arg.device)
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.decay)

    best_model = copy.deepcopy(model)
    best_auc, best_aupr = 0, 0
    result = {}

    for epoch in range(arg.epochs):
        print(f'start training')
        model.train()
        emb_train = []
        for i, batch in enumerate(tqdm(train_loader)):
            adjacency_matrix, node_features, distance_matrix, labels, index = batch
            index = index.cpu().detach().numpy().astype(int)
            adjacency_matrix = adjacency_matrix.to(arg.device)
            node_features = node_features.to(arg.device)
            distance_matrix = distance_matrix.to(arg.device)
            train_rna_data = [celllinefeature[train_rna[x]] for x in index]
            train_rna_data = torch.Tensor(train_rna_data).float().to(arg.device)
            output, emb = model(train_rna_data, node_features, adjacency_matrix, distance_matrix)
            emb_train.append(emb)
            optimizer.zero_grad()

            labels = Variable(labels).float().to(arg.device)
            loss = loss_function(output, labels)  # logits: (N, T, VOCAB), y: (N, T)
            loss.backward()
            optimizer.step()

        print(f'start eval')
        model.eval()
        final_output, final_y = [], []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                adjacency_matrix, node_features, distance_matrix, labels, index = batch
                index = index.cpu().detach().numpy().astype(int)
                adjacency_matrix = adjacency_matrix.to(arg.device)
                node_features = node_features.to(arg.device)
                distance_matrix = distance_matrix.to(arg.device)

                test_rna_data = [celllinefeature[test_rna[x]] for x in index]
                test_rna_data = torch.Tensor(test_rna_data).float().to(arg.device)

                output, _ = model(test_rna_data, node_features, adjacency_matrix, distance_matrix)

                final_y.append(labels.cpu().detach().numpy().flatten())
                final_output.append(output.cpu().detach().numpy().flatten())

        final_y = np.concatenate(np.array(final_y))
        final_output = np.concatenate(np.array(final_output))
        fpr, tpr, _ = roc_curve(final_y, final_output)
        auroc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(final_y, final_output)
        aupr = auc(recall, precision)


        print(f'epoch: {epoch}')
        print(f"AUC: {auroc}, AUPR: {aupr}")

        if auroc > best_auc and aupr >= best_aupr:
            best_auc, best_aupr = auroc, aupr
            best_model = copy.deepcopy(model)
            result = {
                "AUC": auroc,
                "AUPR": aupr
            }
        torch.cuda.empty_cache()

    return best_model, result


def main(arg, drug_model_params):
    with open(arg.root_path + "ClassificationLabel1_" + str(arg.c) + ".pkl", "rb") as f:
        IC50 = pickle.load(f)
    with open(arg.root_path + "drug_encoder.pkl", "rb") as f:
        drug_encoder = pickle.load(f)
    celllinefeature = np.array(pd.read_csv(arg.root_path + "GDSC_" + str(arg.count) + ".csv", header=None, sep="\t"))
    data_x = [[x[0], x[1]] for x in IC50]
    data_label = [x[2] for x in IC50]
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_label, test_size=0.2, random_state=42)

    train_loader, d_atom, train_rna = construct_loader(X_train, y_train, drug_encoder, arg.batch_size)
    test_loader, _, test_rna = construct_loader(X_test, y_test, drug_encoder, arg.batch_size)
    drug_model_params["d_atom"] = d_atom

    best_model, result = train_test(drug_model_params, train_loader, test_loader, arg, celllinefeature, train_rna,
                                    test_rna)
    print(f'result: {result}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='mydata/GDSC/', help='root_path')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--emb_size', type=int, default=128, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--layer', type=float, default=3, help='the number of layer used')
    parser.add_argument('--epochs', type=int, default=100, help='filter incidence matrix')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--decay', type=int, default=0)
    parser.add_argument('--count', type=int, default=3000)
    parser.add_argument('--c', type=int, default=1)

    arg = parser.parse_args()
    print(arg)
    drug_model_params = {
        'd_atom': 26,
        'd_model': 128,
        'N': 8,
        'h': 16,
        'N_dense': 1,
        'lambda_attention': 0.7,
        'lambda_distance': 0.8,
        'leaky_relu_slope': 0.2,
        'dense_output_nonlinearity': 'relu',
        'distance_matrix_kernel': 'exp',
        'dropout': 0.1,
        'aggregation_type': 'sum'
    }
    main(arg, drug_model_params)