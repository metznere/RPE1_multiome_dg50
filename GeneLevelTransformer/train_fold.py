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
from lightning.pytorch import seed_everything
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
from modules import *
import multiome as mo
from sklearn.feature_extraction.text import TfidfTransformer
import argparse

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()

    seed_everything(42, workers = True)

    run_name = f"fold{args.fold}_ln"
    run_dir = os.path.join('/data1/normantm/eli/T7/202511_RPE1_dg50/models/genelevel_clustered/folds_ln', run_name)
    os.makedirs(run_dir, exist_ok=True)

    save_dir = '/data1/normantm/eli/T7/202511_RPE1_dg50/models/genelevel_clustered/data'
    X = pd.read_parquet(os.path.join(save_dir, 'X_clustered.parquet'))
    
    with open(os.path.join(save_dir, 'peakdict_dg50only.pkl'), 'rb') as f:
        peak_dict = pickle.load(f)

    gene_tensor = torch.load(os.path.join(save_dir, 'gene_tensor.pt'))
    chromvar_tensor = torch.load(os.path.join(save_dir, 'chromvar_tensor.pt'))
    expr_tensor = torch.load(os.path.join(save_dir, 'expr_tensor.pt'))

    with open(os.path.join(save_dir, 'gene_to_idx.pkl'), 'rb') as f:
        gene_to_idx = pickle.load(f)
    with open(os.path.join(save_dir, 'tgt_to_idx.pkl'), 'rb') as f:
        tgt_to_idx = pickle.load(f)
    with open(os.path.join(save_dir, 'corr_dict.pkl'), 'rb') as f:
        corr_dict = pickle.load(f)

    df_chromvar = pd.read_csv(os.path.join(save_dir, 'df_chromvar.csv'), index_col=0)
    dg50_pseudo = pd.read_parquet(os.path.join(save_dir, 'dg50_pseudo.parquet'))
    fg = FoldGenerator(X, peak_dict, corr_dict, resid_path = "/data1/normantm/eli/T7/202511_RPE1_dg50/analysis/intermediate_files/260506_tvals_expectation_fit.csv")
    train_data, val_data, test_data = fg.create_fold(args.fold)

    train_genes, val_genes, test_genes = train_data.gene.unique().tolist(), val_data.gene.unique().tolist(), test_data.gene.unique().tolist()
    tgts = [t for t in df_chromvar.index if t in dg50_pseudo.index and t != 'NTC']
    assert len(tgts) == 50

    train_tgts = [f"{t}_{t}" for t in tgts] * len(train_genes)
    train_singles = pd.DataFrame({'gene': train_genes * 50, 'guide_target': train_tgts, 'resid': 0.0})

    val_tgts = [f"{t}_{t}" for t in tgts] * len(val_genes)
    val_singles = pd.DataFrame({'gene': val_genes * 50, 'guide_target': val_tgts, 'resid': 0.0})

    test_tgts = [f"{t}_{t}" for t in tgts] * len(test_genes)
    test_singles = pd.DataFrame({'gene': test_genes * 50, 'guide_target': test_tgts, 'resid': 0.0})

    train_data = pd.concat([train_data, train_singles], axis = 0).drop_duplicates(['guide_target', 'gene'], keep = 'first')
    val_data = pd.concat([val_data, val_singles], axis = 0).drop_duplicates(['guide_target', 'gene'], keep = 'first')
    test_data = pd.concat([test_data, test_singles], axis = 0).drop_duplicates(['guide_target', 'gene'], keep = 'first')

    train_data = train_data.drop_duplicates(['guide_target', 'gene'], keep = 'first')
    val_data = val_data.drop_duplicates(['guide_target', 'gene'], keep = 'first')    
    test_data = test_data.drop_duplicates(['guide_target', 'gene'], keep = 'first')
    
    model = GeneLevelTransformer(gene_tensor, chromvar_tensor, expr_tensor, in_features = 464, d = 256, hidden_dim = 256, dropout = 0.1, lr = 1e-4, wd = 1e-6)
    loaders = PeakIndexDataModule(gene_to_idx, tgt_to_idx, train_data, val_data, test_data, batch_size=1024, num_workers=4)

    early_stop = EarlyStopping(
        monitor = 'val_corr',
        mode = 'max',
        patience = 20,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_corr', 
        mode='max',               
        save_top_k=1,
    )

    gpu_name = torch.cuda.get_device_name(0)
    bf16_supported_gpus = ["A100", "L40S", "H100"]
    precision = "bf16-mixed" if any(gpu in gpu_name for gpu in bf16_supported_gpus) else "16-mixed"
    trainer = Trainer(
        devices = 1,
        accelerator='auto',
        benchmark = False,
        deterministic = True,
        max_epochs = 20,
        default_root_dir='.',
        precision=precision,
        enable_checkpointing=True,
        callbacks=[early_stop, checkpoint_callback],
        logger = WandbLogger(project = 'GeneLevelClustered', name = run_name, save_dir = run_dir),
        log_every_n_steps = 100,
        gradient_clip_val = 1.0,
        enable_progress_bar=False,
        enable_model_summary=True
    )

    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, loaders)
    trainer.test(model, loaders)

    train_indices = [gene_to_idx[g] for g in train_data.gene.unique()]
    mean_gene_tensor = gene_tensor[train_indices].mean(dim=(0, 1)).unsqueeze(0).unsqueeze(0).expand_as(gene_tensor)
    mean_expr_tensor = expr_tensor[train_indices].mean(dim=0).expand_as(expr_tensor)
    mean_chromvar_tensor = chromvar_tensor.mean(dim=0).expand_as(chromvar_tensor)

    real_model = GeneLevelTransformer.load_from_checkpoint(checkpoint_callback.best_model_path, in_features = 464, d = 256, hidden_dim = 256, x=gene_tensor, chromvar_tensor=chromvar_tensor, expr_tensor=expr_tensor)
    real_model.eval()
    real_model.cuda()

    peak_ablated_model = GeneLevelTransformer.load_from_checkpoint(checkpoint_callback.best_model_path, in_features = 464, d = 256, hidden_dim = 256, x=mean_gene_tensor, chromvar_tensor=chromvar_tensor, expr_tensor=expr_tensor)
    peak_ablated_model.eval()
    peak_ablated_model.cuda()

    expr_ablated_model = GeneLevelTransformer.load_from_checkpoint(checkpoint_callback.best_model_path, in_features = 464, d = 256, hidden_dim = 256, x=gene_tensor, chromvar_tensor=chromvar_tensor, expr_tensor=mean_expr_tensor)
    expr_ablated_model.eval()
    expr_ablated_model.cuda()

    chromvar_ablated_model = GeneLevelTransformer.load_from_checkpoint(checkpoint_callback.best_model_path, in_features = 464, d = 256, hidden_dim = 256, x=gene_tensor, chromvar_tensor=mean_chromvar_tensor, expr_tensor=expr_tensor)
    chromvar_ablated_model.eval()
    chromvar_ablated_model.cuda()

    preds, mean_peak_preds, mean_expr_preds, mean_chromvar_preds = [], [], [], []
    with torch.no_grad():
        for idx, y in loaders.test_dataloader():
            
            idx = idx.to(real_model.device)
            x, expr, chromvar_a, chromvar_b = real_model._get_batch(idx)
            logits = real_model(x, expr, chromvar_a, chromvar_b)
            preds.append(logits.cpu())

            idx = idx.to(peak_ablated_model.device)
            x, expr, chromvar_a, chromvar_b = peak_ablated_model._get_batch(idx)
            logits = peak_ablated_model(x, expr, chromvar_a, chromvar_b)
            mean_peak_preds.append(logits.cpu())

            idx = idx.to(expr_ablated_model.device)
            x, expr, chromvar_a, chromvar_b = expr_ablated_model._get_batch(idx)
            logits = expr_ablated_model(x, expr, chromvar_a, chromvar_b)
            mean_expr_preds.append(logits.cpu())

            idx = idx.to(chromvar_ablated_model.device)
            x, expr, chromvar_a, chromvar_b = chromvar_ablated_model._get_batch(idx)
            logits = chromvar_ablated_model(x, expr, chromvar_a, chromvar_b)
            mean_chromvar_preds.append(logits.cpu())

    test_data['pred_resid'] = torch.cat(preds, dim=0).cpu().numpy()
    test_data['peak_ablated_pred_resid'] = torch.cat(mean_peak_preds, dim=0).cpu().numpy()
    test_data['expr_ablated_pred_resid'] = torch.cat(mean_expr_preds, dim=0).cpu().numpy()
    test_data['chromvar_ablated_pred_resid'] = torch.cat(mean_chromvar_preds, dim=0).cpu().numpy()
    test_data['fold'] = args.fold
    test_data.to_csv(os.path.join(run_dir, f"test_preds.csv"))