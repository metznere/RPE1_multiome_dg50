import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import snapatac2 as snap
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import pyranges as pr
import os
import warnings
import requests
from importlib import reload
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import squareform
from umap.distances import hellinger

from tqdm import tqdm
tqdm.pandas()

import sys
sys.path.append('/scratch/eli')
sys.path.append('/data1/normantm/eli/software')
from perturbseq import *
from sparsepca import *
import multiome as mo

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

def bootstrap_programs(df, n_iter = 100, comps = 100, all_components = None):
    
    df = df.sample(n=len(df), replace = True)

    vectors = []
    for i in tqdm(range(n_iter)):
        sparsepca = NonNegativeSparsePCA(n_components=comps, alpha = 1, method = 'cd', max_iter = 1000, n_jobs = 16)
        sparsepca.fit_transform(df)
        vectors.append(sparsepca.components_)
        sys.stdout.flush()
    
    all_components = pd.concat([pd.DataFrame(df) for df in vectors])

    umap = UMAP(n_components = 5, metric = hellinger)
    umap_rep = umap.fit_transform(all_components)
    assert umap_rep.shape == (comps * n_iter, 5)
    
    hdb = HDBSCAN(min_cluster_size = int(0.8 * n_iter), min_samples = int(0.8 * n_iter))
    clusters = hdb.fit(umap_rep)
    n_clusters = len([i for i in np.unique(clusters.labels_).tolist() if i != '-1'])
    
    return all_components, n_clusters

gex = sc.read("/data1/normantm/eli/T7/202511_RPE1_dg50/analysis/intermediate_files/251204_dg50_gex_raw.h5ad")
degs = mo.get_differential_genes(gex, key = 'guide_target', control_key = "guide_target == 'NTC'", n_jobs = 16)

targets = set([t.split("_")[0] for t in gex.obs.target_a.unique() if not pd.isna(t)])
targets = [t.replace('NKX25', 'NKX2-5').replace('NKX31', 'NKX3-1') for t in targets]

meanpop = gex.to_df().join(gex.obs.guide_target).groupby('guide_target').mean().drop("NTC", axis = 0)
degs = pd.read_csv("/data1/normantm/eli/T7/202511_RPE1_dg50/analysis/intermediate_files/251204_degs.csv")
meanpop_fit = meanpop.filter(items = [g for g in degs.query("q < 0.01 and (z > 0.25 or z < -0.25)").gene.unique() if g not in targets], axis = 1)
gene_scores = pd.read_csv('/data1/normantm/multiome_collab/RPE1/amd_eurmeta_natgen_pops_scores.csv')
meanpop_fit = meanpop_fit.filter(items = gene_scores.HGNC.unique())

def zero_tgt(guide_target):
    
    a, b = guide_target.split("_") if "_" in guide_target else (guide_target, None)
    if a == 'NKX25':
        a = 'NKX2-5'
    if a == 'NKX31':
        a = 'NKX3-1'
    if b == 'NKX25':
        b = 'NKX2-5'
    if b == 'NKX31':
        b = 'NKX3-1'

    if a in meanpop_fit.columns:
        meanpop_fit.loc[meanpop_fit.index.str.contains(a.split("-")[0]), a] = 0.0
    if b in meanpop_fit.columns:
        meanpop_fit.loc[meanpop_fit.index.str.contains(b.split("-")[0]), b] = 0.0

for t in targets:
    zero_tgt(t)

bootstrap_comps, n_programs = bootstrap_programs(meanpop_fit, comps = 100)
print(n_programs)
