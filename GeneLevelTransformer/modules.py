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
import multiome as mo
from sklearn.feature_extraction.text import TfidfTransformer
import argparse

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PeakBlock(nn.Module): 

    def __init__(self, d, n_heads=2, dropout=0.1, need_weights = False):
        super().__init__()

        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=dropout)

        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, d)
        )

        self.dropout = nn.Dropout(dropout)
        self.need_weights = need_weights

    def forward(self, x):
        
        h = self.norm1(x)
        attn_out, weights = self.attn(h, h, h, need_weights = self.need_weights)
        x = x + self.dropout(attn_out)

        h = self.norm2(x)
        x = x + self.dropout(self.ff(h))

        if self.need_weights:
            self.attn_weights = weights

        return x

class ChromVARCrossAttentionHead(nn.Module):
    
    def __init__(self, d, n_heads = 2, hidden_dim=128, dropout=0.1, need_weights = False):
        super().__init__()

        self.need_weights = need_weights

        self.norm1 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=dropout)

        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, peak_tokens, label_tokens):

        label_tokens = label_tokens.unsqueeze(1)
        h = self.norm1(label_tokens)

        attn_out, attn_weights = self.cross_attn(h, peak_tokens, peak_tokens, need_weights=self.need_weights)
        label_tokens = label_tokens + self.dropout(attn_out)
        h = self.norm2(label_tokens)

        if self.need_weights:
            self.attn_weights = attn_weights
        
        return self.mlp(h).squeeze(-1).squeeze(-1)

class ChromVARInteractionBlock(nn.Module):
    
    def __init__(self, chromvar_dim=1019, hidden_dim=256, out_dim=128, noise_std=0.1, input_p=0.2, hidden_p=0.1):
        
        super().__init__()
        self.noise_std = noise_std
        
        self.input_dropout = nn.Dropout(input_p)
        self.encoder = nn.Linear(chromvar_dim, hidden_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(hidden_p),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, chromvar_a, chromvar_b):
        
        if self.training and self.noise_std > 0.0:
            chromvar_a = chromvar_a + torch.randn_like(chromvar_a) * self.noise_std
            chromvar_b = chromvar_b + torch.randn_like(chromvar_b) * self.noise_std
            
        chromvar_a = self.input_dropout(chromvar_a)
        chromvar_b = self.input_dropout(chromvar_b)
        
        h_a = self.encoder(chromvar_a)
        h_b = self.encoder(chromvar_b)
        
        sum_comp = h_a + h_b
        prod_comp = h_a * h_b
        
        combined = torch.cat([sum_comp, prod_comp], dim=-1)
        return self.decoder(combined)

class GeneLevelTransformer(LightningModule):

    def __init__(
        self,
        x,
        chromvar_tensor,
        expr_tensor,
        in_features=1079,
        hidden_dim=512,
        dropout=0.1,
        d=256,
        lr=1e-4,
        wd=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['x', 'chromvar_tensor', 'expr_tensor', 'y'])
        self.register_buffer("gene_tensor", x.contiguous(), persistent=False)
        self.register_buffer('chromvar_tensor', chromvar_tensor.contiguous(), persistent=False)
        self.register_buffer('expr_tensor', expr_tensor.contiguous(), persistent=False)

        self.peak_proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),  
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d)
        )
        self.pos = PositionalEncoding(d, max_len=128)
        self.peak_block = nn.Sequential(*[PeakBlock(d, dropout=dropout, n_heads=d // 64) for _ in range(2)])
        self.film_expr = nn.Sequential(
            nn.Linear(50, hidden_dim * 2), 
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, 2 * d)
        )
        
        self.head = ChromVARCrossAttentionHead(d=d, n_heads = d // 64, hidden_dim = 64, dropout=dropout)
        self.chromvar_block = ChromVARInteractionBlock(hidden_dim = hidden_dim * 2, out_dim=d, noise_std = 0.0, input_p = dropout, hidden_p = dropout)

        self.train_pearson = PearsonCorrCoef()
        self.val_pearson = PearsonCorrCoef()
        self.test_pearson = PearsonCorrCoef()
    
    def _get_batch(self, indices):

        gene_idx = indices[:, 0]
        a_idx = indices[:, 1]
        b_idx = indices[:, 2]

        x = self.gene_tensor[gene_idx]
        expr = self.expr_tensor[gene_idx]

        chromvar_a = self.chromvar_tensor[a_idx]
        chromvar_b = self.chromvar_tensor[b_idx]

        return x, expr, chromvar_a, chromvar_b

    def forward(self, x, expr, chromvar_a, chromvar_b):

        x = self.peak_proj(x)
        x = self.pos(x)
        x = self.peak_block(x)
        gamma, beta = self.film_expr(expr).chunk(2, dim=-1)
        x = x * gamma.unsqueeze(1) + beta.unsqueeze(1)

        label_token = self.chromvar_block(chromvar_a, chromvar_b)
        return self.head(x, label_token)

    def training_step(self, batch, batch_idx):

        indices, y = batch
        x, expr, chromvar_a, chromvar_b = self._get_batch(indices)
        logits = self(x, expr, chromvar_a, chromvar_b)

        loss = F.mse_loss(logits, y)
        self.train_pearson(logits.detach().float(), y.detach().float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_corr', self.train_pearson, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):

        indices, y = batch
        x, expr, chromvar_a, chromvar_b = self._get_batch(indices)
        logits = self(x, expr, chromvar_a, chromvar_b)
        
        loss = F.mse_loss(logits, y)
        self.val_pearson(logits.detach().float(), y.detach().float())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_corr', self.val_pearson, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):

        indices, y = batch
        x, expr, chromvar_a, chromvar_b = self._get_batch(indices)
        logits = self(x, expr, chromvar_a, chromvar_b)
        
        loss = F.mse_loss(logits, y)
        self.test_pearson(logits.detach().float(), y.detach().float())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_corr', self.test_pearson, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):

        encoder_params = (
            list(self.peak_proj.parameters()) +
            list(self.peak_block.parameters())
        )

        head_params = (
            list(self.film_expr.parameters()) +
            list(self.chromvar_block.parameters()) +
            list(self.head.parameters())
        )

        optimizer = torch.optim.AdamW([{'params': encoder_params, 'lr': self.hparams.lr * 0.1}, {'params': head_params, 'lr': self.hparams.lr}], weight_decay=self.hparams.wd)
        
        steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        scheduler = lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=[self.hparams.lr * 0.1, self.hparams.lr],
            epochs=self.trainer.max_epochs, 
            steps_per_epoch=steps_per_epoch,
            pct_start=0.05,
            div_factor=10, 
            final_div_factor=1000,
            three_phase=False,
        )
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": 'step'
            }
        }

class FoldGenerator:

    def __init__(self, X, peak_dict, corr_dict, gex_path = '/data1/normantm/eli/T7/202511_RPE1_dg50/analysis/intermediate_files/251204_dg50_gex_raw.h5ad', resid_path = "/data1/normantm/eli/seq2gex/data/260409_dg50_tval_resid.csv"):

        self.X = X
        self.peak_dict = peak_dict
        self.corr_dict = corr_dict

        self.gex = sc.read(gex_path)
        tgts = [t for t in self.gex.obs.guide_target.unique() if "_" not in t]
        self.regions = pd.read_csv("/data1/normantm/eli/seq2gex/data/genome/engreitz_tss500.bed", comment = '#', sep = '\t').query("not name.isin(@tgts) and name.isin(@self.gex.var_names) and name.isin(@self.corr_dict.keys())").copy()
        chrs = [c for c in self.regions.chr.unique() if 'Y' not in c and '_' not in c]

        universe = self.regions.name.unique()
        model_gex = self.gex[:, universe]
        meanpop = model_gex.to_df().join(model_gex.obs.guide_target).groupby('guide_target').mean()
        self.resid = pd.read_csv(resid_path, index_col = 0)
        index_var = 'guide_target' if 'guide_target' in self.resid.reset_index().columns else 'index'
        self.resid_melted = self.resid.reset_index().melt(id_vars = index_var, var_name = 'gene', value_name = 'resid')

        self.folds = {
            1: {'val': ['chr1', 'chr11'], 'test': ['chr2', 'chr12'], 'train': [c for c in chrs if c not in ['chr1', 'chr2', 'chr11', 'chr12']]},
            2: {'val': ['chr2', 'chr12'], 'test': ['chr3', 'chr13'], 'train': [c for c in chrs if c not in ['chr2', 'chr3', 'chr12', 'chr13']]},
            3: {'val': ['chr3', 'chr13'], 'test': ['chr4', 'chr14'], 'train': [c for c in chrs if c not in ['chr3', 'chr4', 'chr13', 'chr14']]},
            4: {'val': ['chr4', 'chr14'], 'test': ['chr5', 'chr15'], 'train': [c for c in chrs if c not in ['chr4', 'chr5', 'chr14', 'chr15']]},
            5: {'val': ['chr5', 'chr15'], 'test': ['chr6', 'chr16'], 'train': [c for c in chrs if c not in ['chr5', 'chr6', 'chr15', 'chr16']]},
            6: {'val': ['chr6', 'chr16'], 'test': ['chr7', 'chr17'], 'train': [c for c in chrs if c not in ['chr6', 'chr7', 'chr16', 'chr17']]},
            7: {'val': ['chr7', 'chr17'], 'test': ['chr8', 'chr18'], 'train': [c for c in chrs if c not in ['chr7', 'chr8', 'chr17', 'chr18']]},
            8: {'val': ['chr8', 'chr18'], 'test': ['chr9', 'chr19'], 'train': [c for c in chrs if c not in ['chr8', 'chr9', 'chr18', 'chr19']]},
            9: {'val': ['chr9', 'chr19'], 'test': ['chr10', 'chr20'], 'train': [c for c in chrs if c not in ['chr9', 'chr10', 'chr19', 'chr20']]},
            10: {'val': ['chr10', 'chr20'], 'test': ['chr11', 'chr21'], 'train': [c for c in chrs if c not in ['chr10', 'chr11', 'chr20', 'chr21']]},
            11: {'val': ['chr3', 'chr21'], 'test': ['chr22', 'chrX'], 'train': [c for c in chrs if c not in ['chr3', 'chr22', 'chr21', 'chrX']]},
            12: {'val': ['chr2', 'chr22'], 'test': ['chr1'], 'train': [c for c in chrs if c not in ['chr2', 'chr1', 'chr22']]}
        }

    def _get_genes_by_fold(self, fold):
        
        train_chrs, val_chrs, test_chrs = self.folds[fold]['train'], self.folds[fold]['val'], self.folds[fold]['test']
        train_genes = self.regions.query("chr.isin(@train_chrs)").name.unique()
        val_genes = self.regions.query("chr.isin(@val_chrs)").name.unique()
        test_genes = self.regions.query("chr.isin(@test_chrs)").name.unique()

        return train_genes, val_genes, test_genes
    
    def _split_data(self, target_genes):

        return self.resid_melted.query("gene.isin(@target_genes)").copy()
    
    def create_fold(self, fold, gene_level = False):

        train_genes, val_genes, test_genes = self._get_genes_by_fold(fold)

        train_data = self._split_data(train_genes)
        val_data = self._split_data(val_genes)
        test_data = self._split_data(test_genes)

        return train_data, val_data, test_data

def get_fold_heldout_strips(tgts, fold_idx):
    
    tgts = np.array(tgts)
    rng = np.random.RandomState(42)
    shuffled = tgts[rng.permutation(len(tgts))]
    
    folds = np.array_split(shuffled, 10)

    test_ids = set(folds[fold_idx])
    val_ids = set(folds[(fold_idx + 1) % 10])
    
    train_labels, val_labels, test_labels = [], [], []
    for a, b in itertools.combinations(sorted(tgts), 2):
        
        pair_str = f"{a}_{b}"
        if a in test_ids or b in test_ids:
            test_labels.append(pair_str)
        elif a in val_ids or b in val_ids:
            val_labels.append(pair_str)
        else:
            train_labels.append(pair_str)
            
    return train_labels, val_labels, test_labels

class PeakIndexDataset(Dataset):

    def __init__(self, data, gene_to_idx, tgt_to_idx):

        split = data["guide_target"].str.split("_", expand=True)

        self.indices = torch.stack([
            torch.tensor(
                data["gene"].map(gene_to_idx).values,
                dtype=torch.long
            ),
            torch.tensor(
                split[0].map(tgt_to_idx).values,
                dtype=torch.long
            ),
            torch.tensor(
                split[1].map(tgt_to_idx).values,
                dtype=torch.long
            ),
        ], dim=1)

        self.resid = torch.tensor(
            data["resid"].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.resid)

    def __getitem__(self, idx):

        return (
            self.indices[idx],
            self.resid[idx]
        )

class PeakIndexDataModule(LightningDataModule):

    def __init__(
        self,
        gene_to_idx,
        tgt_to_idx,
        train_data=None,
        val_data=None,
        test_data=None,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.gene_to_idx = gene_to_idx
        self.tgt_to_idx = tgt_to_idx

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        if self.train_data is not None:
            self.train_dataset = PeakIndexDataset(
                self.train_data,
                self.gene_to_idx,
                self.tgt_to_idx
            )

        if self.val_data is not None:
            self.val_dataset = PeakIndexDataset(
                self.val_data,
                self.gene_to_idx,
                self.tgt_to_idx
            )

        if self.test_data is not None:
            self.test_dataset = PeakIndexDataset(
                self.test_data,
                self.gene_to_idx,
                self.tgt_to_idx
            )

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        ) if hasattr(self, "train_dataset") else None

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        ) if hasattr(self, "val_dataset") else None

    def test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        ) if hasattr(self, "test_dataset") else None
    

class FasterPeakDataset(Dataset):

    def __init__(self, data, gene_tensor, gene_to_idx, chromvar_tensor, tgt_to_idx, expr_tensor):

        self.gene_tensor = gene_tensor
        self.gene_to_idx = gene_to_idx
        self.tgt_to_idx = tgt_to_idx
        self.chromvar_tensor = chromvar_tensor
        self.expr_tensor = expr_tensor

        split = data["guide_target"].str.split("_", expand=True)
        self.a_idx = torch.tensor(split[0].map(tgt_to_idx).values, dtype=torch.long)
        self.b_idx = torch.tensor(split[1].map(tgt_to_idx).values, dtype=torch.long)
        self.gene_idx = torch.tensor(data["gene"].map(gene_to_idx).values, dtype=torch.long)
        self.resid = torch.tensor(data["resid"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.resid)

    def __getitem__(self, idx):

        gene_idx = self.gene_idx[idx]
        a_idx = self.a_idx[idx]
        b_idx = self.b_idx[idx]

        x = self.gene_tensor[gene_idx]
        expr = self.expr_tensor[gene_idx]

        chromvar_a = self.chromvar_tensor[a_idx]
        chromvar_b = self.chromvar_tensor[b_idx]

        y = self.resid[idx]

        return x, expr, chromvar_a, chromvar_b, y

class FasterPeakDataModule(LightningDataModule):
    
    def __init__(
        self, 
        gene_tensor,
        gene_to_idx,
        chromvar_tensor,
        tgt_to_idx,
        expr_tensor,
        train_data=None, 
        val_data=None, 
        test_data=None, 
        batch_size=32, 
        num_workers=4, 
    ):
        super().__init__()
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.gene_tensor = gene_tensor
        self.gene_to_idx = gene_to_idx
        self.chromvar = chromvar_tensor
        self.tgt_to_idx = tgt_to_idx
        self.expr = expr_tensor

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
     
        if self.train_data is not None:
            self.train_dataset = FasterPeakDataset(self.train_data, self.gene_tensor, self.gene_to_idx, self.chromvar, self.tgt_to_idx, self.expr)
        if self.val_data is not None:
            self.val_dataset = FasterPeakDataset(self.val_data, self.gene_tensor, self.gene_to_idx, self.chromvar, self.tgt_to_idx, self.expr)
        if self.test_data is not None:
            self.test_dataset = FasterPeakDataset(self.test_data, self.gene_tensor, self.gene_to_idx, self.chromvar, self.tgt_to_idx, self.expr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True, pin_memory=True) if hasattr(self, 'train_dataset') else None

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True) if hasattr(self, 'val_dataset') else None

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True) if hasattr(self, 'test_dataset') else None
