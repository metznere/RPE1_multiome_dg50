import numpy as np
import pandas as pd
import scanpy as sc
import snapatac2 as snap
import pickle
import sys
sys.path.append('/data1/normantm/eli/software')
import multiome as mo
from sklearn.metrics import r2_score
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import seaborn as sns

from captum.attr import InputXGradient, Saliency, IntegratedGradients
import warnings
import os
import glob

import pyarrow as pa
import pyarrow.parquet as pq

from modules import *

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42


def load_model(fold):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GeneLevelTransformer.load_from_checkpoint(
            fold_paths[fold],
            x=gene_tensor,
            expr_tensor=expr_tensor,
            chromvar_tensor=chromvar_tensor,
            in_features=464,
            hidden_dim=256
        )
        model.eval()
        model.cuda()
        return model


def gene_level_attribution(fold, model, universe_genes, pairs, writer, feature_cols, schema):
    """
    Streams long-format (gene, pair, features...) rows directly to `writer`
    instead of accumulating attribution in memory. With 5000 genes x 1225
    pairs x 460 features in float32, the full matrix is ~10.5GB, so nothing
    beyond a single gene's (n_pairs, n_features) block is ever held at once.

    Both the jacobian-based peak importance and the final IxG/IG attribution
    are computed independently per pair (no averaging across pairs):
      - top k peaks are selected per pair (argsort along the peak axis per
        row, not averaged across pairs first)
      - the final attribution is averaged only over each pair's own top k
        peaks; the pairs axis is never collapsed
    """
    fold_genes = df_genes.query("fold == @fold").index.tolist()
    attr_genes = [g for g in fold_genes if g in universe_genes]
    if not attr_genes:
        return 0

    n_pairs = len(pairs)
    n_written = 0

    fg = FoldGenerator(X, peak_dict, corr_dict, resid_path = "/data1/normantm/eli/T7/202511_RPE1_dg50/analysis/intermediate_files/260506_tvals_expectation_fit.csv")
    train_data, val_data, test_data = fg.create_fold(fold)
    train_indices = [gene_to_idx[g] for g in train_data.gene.unique()]
    mean_gene_tensor = gene_tensor[train_indices].mean(dim=0).unsqueeze(0).cuda()

    for gene in tqdm(attr_genes, desc=f"fold {fold}", leave=False):
        gene_idx = gene_to_idx[gene]
        a_idxs = [tgt_to_idx[tgt.split("_")[0]] for tgt in pairs]
        b_idxs = [tgt_to_idx[tgt.split("_")[1]] for tgt in pairs]

        indices = torch.stack([
            torch.full((n_pairs,), gene_idx, dtype=torch.long),
            torch.tensor(a_idxs, dtype=torch.long),
            torch.tensor(b_idxs, dtype=torch.long)
        ], dim=1)

        # --- jacobian importance, per pair (no averaging across pairs) ---
        x, expr, chromvar_a, chromvar_b = model._get_batch(indices)
        x = x.requires_grad_(True)

        activation_storage = {}
        def hook_fn(module, input, output):
            activation_storage["peak_tokens"] = output
            output.retain_grad()
        hook_handle = model.peak_proj.register_forward_hook(hook_fn)
        model.zero_grad()

        yhat = model(x, expr, chromvar_a, chromvar_b)
        yhat.sum().backward()
        hook_handle.remove()

        grads = activation_storage["peak_tokens"].grad  # (n_pairs, n_peaks, d)
        l2_norms = torch.linalg.vector_norm(grads, dim=-1)  # (n_pairs, n_peaks)
        topq_idx = torch.argsort(l2_norms, dim=1, descending=True)[:, :32]  # (n_pairs, 32)

        # --- IxG / IG ---
        x2, expr2, chromvar_a2, chromvar_b2 = model._get_batch(indices)
        x2.requires_grad_(True)

        def forward_x_only(x):
            return model(x, expr2, chromvar_a2, chromvar_b2)
        
        ixg = IntegratedGradients(forward_x_only, multiply_by_inputs = False)
        x_attr = ixg.attribute(x2, internal_batch_size=1225, n_steps=10, baselines = mean_gene_tensor)
        
        # gather each pair's own top peaks (indices differ per row, so use gather)
        idx_expanded = topq_idx.unsqueeze(-1).expand(-1, -1, x_attr.shape[-1])  # (n_pairs, 32, n_features)
        selected = torch.gather(x_attr, dim=1, index=idx_expanded)  # (n_pairs, 32, n_features)

        # average only over the top peaks; pairs axis stays intact
        gene_attr_per_pair = selected.mean(dim=1)  # (n_pairs, n_features)

        gene_attr_np = gene_attr_per_pair.detach().cpu().numpy().astype(np.float32)
        df_gene = pd.DataFrame(gene_attr_np, columns=feature_cols)
        df_gene.insert(0, 'pair', pairs)
        df_gene.insert(0, 'gene', gene)

        table = pa.Table.from_pandas(df_gene, schema=schema, preserve_index=False)
        writer.write_table(table)
        n_written += 1

    return n_written


if __name__ == "__main__":

    fold_dirs = [f.path for f in os.scandir('folds_ln') if f.name.startswith('fold')]
    test_dfs = [pd.read_csv(os.path.join(fold_dir, 'test_preds.csv'), index_col=0) for fold_dir in fold_dirs]
    test_preds = pd.concat(test_dfs, axis=0)
    test_preds['is_single'] = test_preds.guide_target.map(lambda g: g.split("_")[0] == g.split("_")[1])
    program_features = pd.read_csv("/data1/normantm/eli/T7/202511_RPE1_dg50/analysis/intermediate_files/260414_gex_raw_100_prgm_features.csv")

    df_genes = test_preds.query("not is_single").groupby('gene').apply(lambda g: np.corrcoef(g.resid, g.pred_resid)[0, 1]).to_frame('real_corr')
    df_genes = df_genes.join(test_preds.query("not is_single").groupby('gene').apply(lambda g: np.corrcoef(g.resid, g.peak_ablated_pred_resid)[0, 1]).to_frame('shuffled_corr'))
    df_genes['atac_uplift'] = df_genes.real_corr - df_genes.shuffled_corr
    df_genes['atac_uplift_pct'] = 100 * (df_genes.real_corr - df_genes.shuffled_corr) / df_genes.shuffled_corr
    df_genes_r2 = test_preds.query("not is_single").groupby('gene').apply(lambda g: r2_score(g.resid, g.pred_resid)).to_frame('real_r2')
    df_genes_r2 = df_genes_r2.join(test_preds.query("not is_single").groupby('gene').apply(lambda g: r2_score(g.resid, g.peak_ablated_pred_resid)).to_frame('shuffled_r2'))
    df_genes_r2['atac_uplift_r2'] = df_genes_r2.real_r2 - df_genes_r2.shuffled_r2
    df_genes = df_genes.join(df_genes_r2)
    df_genes['is_prgm'] = df_genes.index.isin(program_features.feature.unique())
    fold_dict = dict(zip(test_preds.drop_duplicates('gene', keep='first').gene, test_preds.drop_duplicates('gene', keep='first').fold))
    df_genes['fold'] = df_genes.index.map(fold_dict)

    fold_paths = {
        i: glob.glob(f"/data1/normantm/eli/T7/202511_RPE1_dg50/models/genelevel_clustered/folds_ln/fold{i}_ln/**/*.ckpt", recursive=True)[0]
        for i in range(1, 13)
    }

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
    fg = FoldGenerator(X, peak_dict, corr_dict, resid_path="/data1/normantm/eli/T7/202511_RPE1_dg50/analysis/intermediate_files/260506_tvals_expectation_fit.csv")
    train_data, val_data, test_data = fg.create_fold(1)

    universe = df_genes.query('real_corr >= 0.25 and atac_uplift_r2 >= 0 and is_prgm')
    pairs = test_data.guide_target.unique().tolist()

    feature_cols = X.columns.tolist() + ['corr_expr']
    out_path = "genelevel_attr_pairs_ig.parquet"

    # explicit schema so every fold's table matches exactly, no per-fold inference drift
    schema = pa.schema(
        [('gene', pa.string()), ('pair', pa.string())] +
        [(c, pa.float32()) for c in feature_cols]
    )

    writer = pq.ParquetWriter(out_path, schema)
    total_written = 0
    try:
        for fold in trange(1, 13):
            model = load_model(fold)
            total_written += gene_level_attribution(
                fold, model, universe.index.tolist(), pairs, writer, feature_cols, schema
            )
            del model
            torch.cuda.empty_cache()
    finally:
        writer.close()

    print(
        f"Done. Wrote {total_written} genes x {len(pairs)} pairs "
        f"({total_written * len(pairs)} rows) to {out_path} (long format: gene, pair columns)."
    )