import numpy as np
import pandas as pd
import scanpy as sc
import pyranges as pr
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import PearsonCorrCoef, R2Score
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning import LightningModule, LightningDataModule
from lightning.pytorch.utilities import seed
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler
import os 
import sys
import itertools
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
tqdm.pandas()
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform
import seaborn as sns
import scipy.cluster.hierarchy as hc
from scipy.stats import spearmanr
from importlib import reload
import snapatac2 as snap
from itertools import product
import pickle
import copy
sys.path.append('/scratch/eli')
sys.path.append('/data1/normantm/eli/software')
from perturbseq import *
from sparsepca import *
from modules import *
import multiome as mo

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import math

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

class FactorizedBilinearBaseline(nn.Module):
    def __init__(self, n_chromvar, n_peak, rank):
        super().__init__()
        self.U = nn.Parameter(torch.randn(n_chromvar, rank) / math.sqrt(rank))
        self.V = nn.Parameter(torch.randn(n_peak, rank) / math.sqrt(rank))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, A, B):
        A_proj = A @ self.U
        B_proj = B @ self.V
        return A_proj @ B_proj.T + self.bias

class UpgradedFactorizedBaseline(nn.Module):
    def __init__(self, n_chromvar, n_peak, rank, hidden_dim=128, dropout=0.1):
        super().__init__()

        self.pert_bias = nn.Linear(n_chromvar, 1)
        self.gene_bias = nn.Linear(n_peak, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.pert_proj = nn.Sequential(
            nn.Linear(n_chromvar, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rank)
        )
        
        self.gene_proj = nn.Sequential(
            nn.Linear(n_peak, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rank)
        )
        
        self.pert_norm = nn.LayerNorm(rank)
        self.gene_norm = nn.LayerNorm(rank)

    def forward(self, A, B):
   
        b_A = self.pert_bias(A)
        b_B = self.gene_bias(B)
        
        A_emb = self.pert_norm(self.pert_proj(A))
        B_emb = self.gene_norm(self.gene_proj(B))
        interactions = A_emb @ B_emb.T
        
        return interactions + b_A + b_B.T + self.global_bias
    
def train_bilinear_model(
    A, B, Y_train, Y_val, Y_test,
    rank=32, lr=1e-3, weight_decay=1e-4, dropout=0.2, epochs=100000, device='cuda', patience=5000, linear=True
):
    n_chromvar = A.shape[1]
    n_peak = B.shape[1]

    model = FactorizedBilinearBaseline(n_chromvar, n_peak, rank).to(device) if linear else UpgradedFactorizedBaseline(n_chromvar, n_peak, rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    A = torch.tensor(A.values, dtype=torch.float32, device=device)
    B = torch.tensor(B.values, dtype=torch.float32, device=device)
    Y_train = torch.tensor(Y_train.values, dtype=torch.float32, device=device)
    Y_val = torch.tensor(Y_val.values, dtype=torch.float32, device=device)
    Y_test = torch.tensor(Y_test.values, dtype=torch.float32, device=device)

    mask_train = ~torch.isnan(Y_train)
    mask_val = ~torch.isnan(Y_val)
    mask_test = ~torch.isnan(Y_test)

    Y_train_safe = torch.nan_to_num(Y_train, 0.0)
    Y_val_safe = torch.nan_to_num(Y_val, 0.0)
    Y_test_safe = torch.nan_to_num(Y_test, 0.0)

    criterion = nn.MSELoss(reduction='mean')

    best_val_loss = float('inf')
    best_model_state = None
    steps_since_improvement = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        Y_pred = model(A, B)
        train_loss = criterion(Y_pred[mask_train], Y_train_safe[mask_train])
        train_loss.backward()
        optimizer.step()

        if epoch % 1000 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                Y_pred_eval = model(A, B)
                val_loss = criterion(Y_pred_eval[mask_val], Y_val_safe[mask_val])

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_model_state = copy.deepcopy(model.state_dict())
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1000

            if steps_since_improvement >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        Y_pred_final = model(A, B)
        if mask_test.sum() > 0:
            test_preds = Y_pred_final[mask_test].cpu().numpy()
            test_targets = Y_test_safe[mask_test].cpu().numpy()

            test_mse = mean_squared_error(test_targets, test_preds)
            test_r2 = r2_score(test_targets, test_preds)
            test_pearson = pearsonr(test_targets, test_preds).statistic
        else:
            test_mse, test_r2, test_pearson = float('nan'), float('nan'), float('nan')

    return test_mse, test_r2, test_pearson

if __name__ == '__main__':

    X = pd.read_parquet('/data1/normantm/eli/T7/202511_RPE1_dg50/models/genelevel_clustered/data/X_clustered.parquet')
    with open('/data1/normantm/eli/T7/202511_RPE1_dg50/models/genelevel_clustered/data/peakdict_dg50only.pkl', 'rb') as f:
        peak_dict = pickle.load(f)
    with open('/data1/normantm/eli/T7/202511_RPE1_dg50/models/genelevel_clustered/data/corr_dict.pkl', 'rb') as f:
        corr_dict = pickle.load(f)

    df_genes = pd.read_parquet('/data1/normantm/eli/T7/202511_RPE1_dg50/models/genelevel_clustered/data/df_genes.parquet')
    fg = FoldGenerator(X, peak_dict, corr_dict, resid_path = "/data1/normantm/eli/T7/202511_RPE1_dg50/analysis/intermediate_files/260506_tvals_expectation_fit.csv")

    chromvar_singles = pd.read_csv('/data1/normantm/eli/T7/202511_RPE1_dg50/models/genelevel_clustered/data/df_chromvar.csv', index_col = 0)
    df_pca = pd.DataFrame(PCA(n_components = 32).fit_transform(chromvar_singles), index = chromvar_singles.index, columns = [f"PC{i}" for i in range(32)])
    df_pca.index = df_pca.index.map(lambda i: i.replace("NKX2-5", "NKX25").replace("NKX3-1", "NKX31"))

    n = df_pca.shape[1]
    df_pairs = pd.DataFrame(index=fg.resid.index, columns=[str(i) for i in range(2 * n)])

    def get_pair_rep(pair):
        a, b = pair.split("_")
        additive = df_pca.loc[a] + df_pca.loc[b]
        multiplicative = df_pca.loc[a] * df_pca.loc[b]
        return np.concatenate([additive.values, multiplicative.values])

    for p in fg.resid.index:
        df_pairs.loc[p] = get_pair_rep(p)

    A = df_pairs.copy()
    scaler = StandardScaler()
    A = pd.DataFrame(scaler.fit_transform(A.values))

    y = fg.resid.T.copy()
    common_genes = np.intersect1d(df_genes.index, y.index)
    B = df_genes.copy().loc[common_genes]
    y = y.loc[common_genes]

    df_performance = pd.DataFrame(index = fg.folds.keys(), columns = ['linear_mse', 'linear_r2', 'linear_pearson', 'nonlinear_mse', 'nonlinear_r2', 'nonlinear_pearson'])
    for fold in tqdm(fg.folds.keys()):

        train_genes, val_genes, test_genes = fg._get_genes_by_fold(fold)
        
        y_train, y_val, y_test = y.copy(), y.copy(), y.copy()
        y_train.loc[val_genes.tolist() + test_genes.tolist()] = np.nan
        y_val.loc[train_genes.tolist() + test_genes.tolist()] = np.nan
        y_test.loc[train_genes.tolist() + val_genes.tolist()] = np.nan

        lmse, lr2, lpearson = train_bilinear_model(A.astype(np.float32), B.astype(np.float32), y_train.T.astype(np.float32), y_val.T.astype(np.float32), y_test.T.astype(np.float32), rank = 64, weight_decay=1e-3, linear = True)
        nmse, nr2, npearson = train_bilinear_model(A.astype(np.float32), B.astype(np.float32), y_train.T.astype(np.float32), y_val.T.astype(np.float32), y_test.T.astype(np.float32), rank = 64, weight_decay=1e-3, linear = False)
        print(f"Fold {fold}: Linear MSE: {lmse:.4f} | Linear R2: {lr2:.4f} | Linear Pearson: {lpearson:.4f} | Nonlinear MSE: {nmse:.4f} | Nonlinear R2: {nr2:.4f} | Nonlinear Pearson: {npearson:.4f}")
        df_performance.loc[fold] = [lmse, lr2, lpearson, nmse, nr2, npearson]
    
    df_performance.to_csv("fold_baseline_performance.csv")
