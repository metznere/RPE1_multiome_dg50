import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import snapatac2 as snap
import os
from tqdm import tqdm
tqdm.pandas()
from perturbseq import *
import multiome as mo
from joblib import Parallel, delayed

# requires SnapATAC2 v2.6.0 because of a breaking change in TSS enrichment calculation that changes qc metrics in later versions

if __name__ == '__main__':

    for file in os.scandir("ID_calling/250102"):
        if file.name.endswith("_called_ids.csv"):
            df = pd.read_csv(file.path)[['CB','identity','UMI','total_reads','cluster','LB_identity']]

    expt = mo.MultiomeExperiment(
        experiment_dir = '/data1/normantm/RPE1_multiome_collab/cellranger_outs',
        id_dir = '/data1/normantm/RPE1_multiome_collab/ID_calling/250102',
        intermediate_file_dir='/data1/normantm/RPE1_multiome_collab/intermediate_files'
    )

    def correct_id(id):
        if 'NTC_' in id:
            return 'NTC'
        elif 'NKX2-5' in id:
            return id.replace('NKX2-5','NKX25')
        elif 'NKX3-1' in id:
            return id.replace('NKX3-1','NKX31')
        else:
            return id

    for s in expt.sample_names:
        expt.combined_qc(s, umi = 7, complexity=6.5, tsse = 9, plot_qc = False, verbose = True)
        df = expt.singlets_assigned[s]
        df['joint_id_ntc'] = df['identity'].map(correct_id)
        expt.singlets_assigned[s] = df

    expt.create_full_cellpop(low_expr_threshold=0.06, plot_first = False, plot_regressed=False)
    expt.create_full_atac(groupby = 'guide_target', plot = False, harmony = False)
    expt.peaks.var = mo.label_peaks(expt.peaks.var_names.to_frame().rename(columns={0:'peak_id'}).reset_index(drop=True))
    expt.all_singlets_assigned = pd.concat(expt.singlets_assigned.values())
    expt.all_singlets_assigned['gem_group'] = expt.all_singlets_assigned.index.map(lambda i: i.split("-")[1])
    expt.gex.obs['guide_target'] = expt.gex.obs.guide_target.map(lambda i: i.replace("NKX2-5", "NKX25").replace("NKX3-1", "NKX31"))
    expt.atac.obs['guide_target'] = expt.atac.obs.guide_target.map(lambda i: i.replace("NKX2-5", "NKX25").replace("NKX3-1", "NKX31"))
    expt.peaks.obs['guide_target'] = expt.peaks.obs.guide_target.map(lambda i: i.replace("NKX2-5", "NKX25").replace("NKX3-1", "NKX31"))
    expt.save("intermediate_files/RPE1_multiomeTFs.pkl")

    degs = mo.get_differential_genes(expt.gex, key = 'guide_target', control_key = "guide_target == 'NTC'", multi_method = 'fdr_bh')
    degs.to_csv("intermediate_files/RPE1_degs.csv", index = False)
    expt.degs = degs

    guides = expt.atac.obs.query("guide_target != 'NTC'").guide_target.unique()
    merged_peaks = snap.tl.merge_peaks(expt.atac.uns['macs3'], snap.genome.hg38).to_pandas()
    merged_peaks.columns = merged_peaks.columns.map(lambda i: i.replace("NKX2-5", "NKX25").replace("NKX3-1", "NKX31"))
    daps = Parallel(n_jobs=16, verbose = 10)(
        delayed(mo.get_differential_peaks)(expt.peaks, merged_peaks, g, groupby='guide_target') 
        for g in guides
    )
    daps_labeled = [mo.label_peaks(df.rename({"feature": "peak_id"}, axis = 1)) for df in daps]
    daps_combined = pd.concat(daps_labeled, axis = 0).query("q < 0.1")
    daps_combined.to_csv("intermediate_files/RPE1_daps.csv")
    expt.daps = daps_combined

    expt.gex.var['mito'] = expt.gex.var['mito'].astype(str)
    expt.gex.obs['single_cell'] = expt.gex.obs['single_cell'].astype(str)
    expt.save("intermediate_files/RPE1_multiomeTFs.pkl")

    expt.gex.write_h5ad("RPE1_multiome_gex.h5ad")
    expt.atac.write_h5ad("RPE1_multiome_atac.h5ad")
    expt.peaks.write_h5ad("RPE1_multiome_peaks.h5ad")
    merged_peaks.to_csv("peak_calls_by_tgt.csv")