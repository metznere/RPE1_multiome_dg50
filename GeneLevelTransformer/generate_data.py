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
import sys
sys.path.append('/scratch/eli')
sys.path.append('/data1/normantm/eli/software')
from perturbseq import *
from sparsepca import *
import multiome as mo # can skip
from sklearn.feature_extraction.text import TfidfTransformer
import argparse

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

if __name__ == "__main__":

    atac = sc.read('/data1/normantm/multiome_collab/RPE1/RPE1_multiome_atac.h5ad')
    atac.obs.guide_target = atac.obs.guide_target.str.replace('NKX25', 'NKX2-5').str.replace('NKX31', 'NKX3-1')
    multiome_gex = sc.read('/data1/normantm/multiome_collab/RPE1/RPE1_multiome_gex.h5ad')
    multiome_gex.obs.guide_target = multiome_gex.obs.guide_target.str.replace('NKX25', 'NKX2-5').str.replace('NKX31', 'NKX3-1')
    with open('data/peakdict_dg50only.pkl', 'rb') as f:
        peak_dict = pickle.load(f)

    X = pd.read_parquet('/data1/normantm/eli/T7/202511_RPE1_dg50/models/genelevel_clustered/data/X_clustered.parquet')
    print(X.shape)

    tgts = [c.split("_")[0] for c in X.columns if c.endswith("_res_acc")]
    adata = atac[atac.obs.guide_target.isin(tgts + ['NTC'])]
    mtx = snap.pp.make_peak_matrix(adata, use_rep = X.index.tolist())
    mtx_pseudo = X.filter(like="_res_acc").T
    mtx_pseudo.index = mtx_pseudo.index.map(lambda i: i.split("_")[0])

    gex = multiome_gex[multiome_gex.obs.guide_target.isin(mtx_pseudo.index)]
    gex_pseudo = sc.get.aggregate(gex, by = 'guide_target', func = 'mean')
    gex_pseudo = pd.DataFrame(gex_pseudo.layers['mean'], index = gex_pseudo.obs_names, columns = gex_pseudo.var_names)

    dg50_gex = sc.read('/data1/normantm/eli/T7/202511_RPE1_dg50/analysis/intermediate_files/251204_dg50_gex_raw.h5ad')
    dg50_pseudo = sc.get.aggregate(dg50_gex, by = 'guide_target', func = 'mean')
    dg50_pseudo = pd.DataFrame(dg50_pseudo.layers['mean'], index = dg50_pseudo.obs_names, columns = dg50_pseudo.var_names)
    dg50_pseudo = dg50_pseudo.loc[gex_pseudo.index.str.replace("NKX2-5", "NKX25").str.replace("NKX3-1", "NKX31")]
    mtx_pseudo = mtx_pseudo.loc[gex_pseudo.index]

    corr_dict, dg50_corr_dict = {}, {}
    peak_dict_inmtx = {k: v for k, v in peak_dict.items() if k in gex_pseudo.columns and k in dg50_pseudo.columns}
    for k, v in tqdm(peak_dict_inmtx.items()):
        expr, dg50_expr = gex_pseudo[k], dg50_pseudo[k]
        corrs = [np.corrcoef(expr, mtx_pseudo[peak])[0,1] for peak in v]
        corr_dict[k] = corrs

    df_chromvar = pd.read_csv('/data1/normantm/eli/seq2gex/data/RPE1_chromvar_jaspar26.csv', index_col = 0)
    df_chromvar.index = df_chromvar.index.map(lambda i: i.replace("NKX2-5", "NKX25").replace("NKX3-1", "NKX31"))

    tgts = [i for i in dg50_pseudo.index if "_" not in i and i != 'NTC']
    assert len(tgts) == 50
    tgt_to_idx = {k: i for i, k in enumerate(tgts)}
    df_chromvar = df_chromvar.loc[tgts]
    chromvar_tensor = torch.tensor(df_chromvar.values, dtype=torch.float32)

    df_expr = dg50_pseudo.query("not index.str.contains('_')").T.copy()
    df_expr.columns = [c + "_expr" for c in df_expr.columns]

    genes = np.intersect1d(list(corr_dict.keys()), df_expr.index)
    gene_tensor = torch.empty((len(genes), 128, X.shape[1] + 1), dtype=torch.float32)
    for i, g in tqdm(enumerate(genes), total = len(genes)):
        peaks = X.loc[peak_dict[g]]
        peaks['corr_expr'] = corr_dict[g]
        gene_tensor[i, :, :] = torch.tensor(peaks.values, dtype=torch.float32)
    gene_to_idx = {k: i for i, k in enumerate(genes)}

    df_expr = df_expr.loc[genes]
    expr_tensor = torch.tensor(df_expr.values, dtype=torch.float32)

    save_dir = '/data1/normantm/eli/T7/202511_RPE1_dg50/models/genelevel_clustered/data'
    os.makedirs(save_dir, exist_ok=True)

    # tensors
    torch.save(gene_tensor, os.path.join(save_dir, 'gene_tensor.pt'))
    torch.save(chromvar_tensor, os.path.join(save_dir, 'chromvar_tensor.pt'))
    torch.save(expr_tensor, os.path.join(save_dir, 'expr_tensor.pt'))

    # lookup dicts
    with open(os.path.join(save_dir, 'gene_to_idx.pkl'), 'wb') as f:
        pickle.dump(gene_to_idx, f)
    with open(os.path.join(save_dir, 'tgt_to_idx.pkl'), 'wb') as f:
        pickle.dump(tgt_to_idx, f)
    with open(os.path.join(save_dir, 'corr_dict.pkl'), 'wb') as f:
        pickle.dump(corr_dict, f)

    # dataframes needed for FoldGenerator
    df_chromvar.to_csv(os.path.join(save_dir, 'df_chromvar.csv'))
    dg50_pseudo.to_parquet(os.path.join(save_dir, 'dg50_pseudo.parquet'))



