import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scanpy as sc
import snapatac2 as snap
import anndata as ad
import pyBigWig as bw
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import requests
import sys
from plotnine import *
import pickle
from umap import UMAP
import pyranges as pr
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform, jaccard, pdist
import scipy.cluster.hierarchy as hc
from scipy.stats import mannwhitneyu, false_discovery_control, binomtest, brunnermunzel
from perturbseq import *

# this breaks if your anndata is >0.10.7, can't figure out why - check with ``import anndata; print(anndata.__version__)``

class MultiomeExperiment:

    """
    Initialize a multiome experiment. Assumes your cellranger-arc output is in the directory ``experiment_dir`` with one folder for each lane and folders named Lane1_xxx, Lane2_yyy ... Assumes your ID calls are in ``id_dir`` with csv files named {sample_name}_called_ids.csv where sample_name matches the folders detected in ``experiment_dir``.

    Args:
        experiment_dir (str): Path to the directory containing the cellranger-arc output.
        id_dir (str): Path to the directory containing the ID calls.
        intermediate_file_dir (str): Path to the directory where intermediate files will be stored. If None, defaults to ``{experiment_dir}/analysis/intermediate_files``.

    """

    def __init__(self, experiment_dir, id_dir, intermediate_file_dir = None):

        self.experiment_dir = experiment_dir
        self.id_dir = id_dir
        self.intermediate_file_dir = intermediate_file_dir if intermediate_file_dir is not None else os.path.join(self.experiment_dir, "analysis", "intermediate_files")
        self.sample_names = []
        for file in os.scandir(experiment_dir):
            if "Lane" in file.name:
                self.sample_names.append(file.name)
        print("Found samples: ", self.sample_names, flush = True)

        self.gex_qc = {}
        self.atac_qc = {}
        self.singlets_assigned = {}
        self.preprocessed_atac = {}
        self.cellpops = {}
        self.peak_mtxs = {}
        self.atac_mtxs = {}

        preprocessed_files_exist = all(
            os.path.exists(os.path.join(self.intermediate_file_dir, f"{sample}_preprocessed_atac.h5")) 
            for sample in self.sample_names
        )

        if preprocessed_files_exist:
            print("Loading preprocessed ATAC...", flush = True)
            self._load_preprocessed_atac()
        else:
            print("Preprocessing ATAC data...", flush = True)
            self._preprocess_atac()
            self._load_preprocessed_atac()

    @classmethod
    def load(cls, filename):
        
        """
        Load a MultiomeExperiment object from a pickle file.

        Args:
            filename (str): Path to the pickle file.
        """
        
        with open(filename, 'rb') as f:
            expt = pickle.load(f)
            return expt

    def save(self, filename):

        """
        Save a MultiomeExperiment object to a pickle file. If full population matrices have been created, removes individual lanes to save space.

        Args:
            filename (str): Path to the pickle file.
        """
        
        self.preprocessed_atac = {}
        if hasattr(self, 'gex') and hasattr(self, 'atac') and hasattr(self, 'peaks'):
            
            if self.gex and self.atac and self.peaks:
                
                if hasattr(self, 'cellpops'):
                    del self.cellpops
                if hasattr(self, 'atac_mtxs'):
                    del self.atac_mtxs
                if hasattr(self, 'peak_mtxs'):
                    del self.peak_mtxs

                print("Writing out singlets...", flush = True)
                cells = self.gex.obs.join(self.atac.obs[['n_fragment', 'tsse']])
                cells.to_csv(os.path.join(self.intermediate_file_dir, "singlets_assigned.csv"))
        
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def _preprocess_atac(self, outdir = None):

        outdir = outdir if outdir is not None else self.intermediate_file_dir
        for sample in self.sample_names:

            adata = snap.pp.import_data(
                f"{self.experiment_dir}/{sample}/outs/atac_fragments.tsv.gz",
                chrom_sizes = snap.genome.hg38,
                sorted_by_barcode = False
            )

            snap.metrics.tsse(adata, snap.genome.hg38)
            snap.pp.add_tile_matrix(adata)
            snap.pp.select_features(adata, n_features=250000)
            adata.write(os.path.join(outdir, f"{sample}_preprocessed_atac.h5"))
    
    def _load_preprocessed_atac(self):
        
        for sample in self.sample_names:
            self.preprocessed_atac[sample] = snap.read(os.path.join(self.intermediate_file_dir, f"{sample}_preprocessed_atac.h5"), backed = None)

    def _get_cellcycle_genes(self):
        
        dfS = requests.get("https://maayanlab.cloud/Harmonizome/api/1.0/gene_set/S+Phase/Reactome+Pathways+2024").json()
        dfG2M = requests.get("https://maayanlab.cloud/Harmonizome/api/1.0/gene_set/regulation+of+cell+cycle+G2%24slash%24M+phase+transition/GO+Biological+Process+Annotations+2023").json()
        dfM = requests.get("https://maayanlab.cloud/Harmonizome/api/1.0/gene_set/M+Phase/Reactome+Pathways+2024").json()

        S_genes = [e['gene']['symbol'] for e in dfS['associations']]
        G2M_genes = [e['gene']['symbol'] for e in dfG2M['associations']]
        G2M_genes.extend([e['gene']['symbol'] for e in dfM['associations']])
        G2M_genes = list(set(G2M_genes))

        return S_genes, G2M_genes

    def assignment_rate_by_lane(self, figsize = (7, 5.5), savefig = None, axhline = None):

        """
        Plot the ID assignment rate by lane. 

        Args:
            figsize (tuple): Figure size for output.
            savefig (str): Path to save the figure. If None, figure is displayed.
            axhline (float): Add a horizontal line to the plot at this value.
        """
        
        dfs = []
        guide_call_summary = [file for file in os.scandir(self.id_dir) if "guide_call_summary" in file.name][0].path
        for sample_id in self.sample_names:
            df = pd.read_csv(guide_call_summary, index_col = 0).query(f"sample == '{sample_id}'")
            df['calls'] = df.n_guides.map(lambda n: '3+' if n in ['3','4+'] else str(n))
            df.rename({'pct_cells': 'pct_nuclei', 'count': 'n_cells'}, axis = 1, inplace = True)
            df = df[['calls', 'n_cells', 'pct_nuclei']].groupby('calls').sum().reset_index()
            df['library'] = sample_id
            dfs.append(df)
        df = pd.concat(dfs, axis = 0).query("calls != '0'")
        df['calls'] = pd.Categorical(df.calls, categories = ['3+', '2', '1'], ordered = True)

        p = (
            ggplot(df)
                + geom_col(aes(x = 'library', y = 'pct_nuclei', fill = 'calls'))
                + theme_light()
                + theme(text=element_text(family="Helvetica"))
                + theme(axis_text_x=element_text(size=12),  # Set X-axis label font size
                        axis_text_y=element_text(size=12))
                + theme(figure_size=figsize)
                + theme(legend_position = (0.05,0.9))
                + ylim(0,100)
                + scale_fill_manual(sns.color_palette("rocket", 3).as_hex())
                + labs(x = '', y = 'Percent nuclei assigned', fill = 'IDs')
        )
        if axhline is not None:
            p += geom_hline(yintercept=axhline, linetype='dashed', color='grey')

        if savefig is not None:
            p.save(savefig, dpi = 300)
        p.draw(show=True)

        return df

    def id_density_by_lane(self, subplot_dims = (1,2), savefig = None, figsize = None):
        
        """
        Produce a density plot of ID total reads vs UMI by lane for assignment QC. 

        Args:
            figsize (tuple): Figure size for output.
            savefig (str): Path to save the figure. If None, figure is displayed.
            subplot_dims (float): Dimensions (rows, columns) for plotting multiple lanes within one figure.
        """
        
        dfs = []
        for sample_id in self.sample_names:
            
            raw_reads = pd.read_csv(os.path.join(self.id_dir, f"{sample_id}_filtered_barcode_umis.csv.gz"))
            if 'LB_identity' not in raw_reads.columns:
                raw_reads['LB_identity'] = "_"
            raw_reads['ID'] = raw_reads.apply(lambda df: df['identity'].split('_')[0] + "_" + df['LB_identity'].split('_')[0], axis=1)
            identities = raw_reads.groupby(['CB', 'ID'])[['ID']].count()
            identities.columns = ['UMI']
            identities['UMI'] = np.log2(identities.UMI)
            identities['total_reads'] = raw_reads.groupby(['CB', 'ID'])['count'].sum()
            identities.total_reads = np.log2(identities.total_reads)
            dfs.append(identities)
        
        figsize = (5 * subplot_dims[1], 5) if figsize is None else figsize
        fig, ax = plt.subplots(figsize = figsize, nrows = subplot_dims[0], ncols = subplot_dims[1], sharey = True, sharex = True)
        ax = np.array(ax).ravel()
        for i, df in enumerate(dfs):
            ax[i].hexbin(x = df.UMI, y = df.total_reads, cmap = 'mako', vmax = 100, gridsize = 50, mincnt = 1)
            ax[i].set_title(self.sample_names[i])
        fig.supxlabel("log2(ID UMI)", x = 0.55, y = 0.025)
        fig.supylabel("log2(ID total reads)")

        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig, dpi = 300)
        plt.show()

        return dfs

    def cut_qc_metrics_by_called(self, sample_name, figsize = (15,5), plot = False, common_norm = False, mode = 'gex', savefig = None):

        """
        Plot and store QC metrics for assigned and unassigned cells.

        Args:
            sample_name (str): Lane to calculate QC metrics for.
            figsize (tuple): Figure size for output.
            plot (bool): Plot metrics after calculation, or store only.
            common_norm (bool): Display assigned and unassigned histograms as counts (False) or each normalized to the same scale (True).
            mode (str): 'gex' or 'atac'.
            savefig (str): Path to save the figure. If None, figure is displayed.
        Returns: pd.DataFrame with QC metrics.
        """

        id_files = [file for file in os.scandir(self.id_dir) if "called_ids" in file.name]
        called_ids = pd.read_csv([f for f in id_files if sample_name in f.name][0])
        called_ids['joint_id'] = called_ids.apply(lambda df: df['identity'].split('_')[0] + "_" + df['LB_identity'].split('_')[0] if df['LB_identity'] != "_" else df['identity'].split("_")[0], axis=1)
        called_ids = called_ids.groupby("CB").joint_id.count().to_frame("n_id")
        called_ids.index = called_ids.index.map(lambda cb: cb + '-' + str(sample_name.split('Lane')[1].split('_')[0]))

        if mode == 'gex':

            adata_files = [file for file in os.scandir(self.experiment_dir) if sample_name in file.name]
            adata_path = adata_files[0].path + '/outs/filtered_feature_bc_matrix.h5'

            adata = sc.read_10x_h5(adata_path)
            adata.var_names_make_unique()
            adata.var["mito"] = adata.var_names.str.startswith("MT-")
            sc.pp.calculate_qc_metrics(adata, inplace=True, qc_vars = ["mito"])
            adata.obs.index = adata.obs.index.map(lambda cb: cb.split("-")[0] + '-' + str(sample_name.split('Lane')[1].split('_')[0]))
            merge = adata.obs.join(called_ids, how = 'left')
            merge["called"] = merge["n_id"].fillna(0).map(lambda n: n > 0)
            merge.drop_duplicates(inplace = True)

            if plot:
                fig, ax = plt.subplots(1, 3, figsize=figsize)
                sns.histplot(data = merge, x = 'log1p_total_counts', kde = True, common_norm = common_norm, hue = 'called', ax = ax[0])
                sns.histplot(data = merge, x = 'log1p_n_genes_by_counts', kde = True, common_norm = common_norm, hue = 'called', ax = ax[1])
                sns.histplot(data = merge, x = 'pct_counts_mito', kde = True, common_norm = common_norm, hue = 'called', ax = ax[2])
                ax[2].set_xlim([-2,50])
                fig.suptitle(f"{sample_name} GEX QC metrics by guide call")
                plt.tight_layout()
                if savefig is not None:
                    plt.savefig(savefig, dpi = 300)
                fig.show()

            self.gex_qc[sample_name] = merge

        elif mode == 'atac':
            
            self._load_preprocessed_atac()
            adata = self.preprocessed_atac[sample_name]
            adata.obs.index = adata.obs.index.map(lambda cb: cb.split("-")[0] + '-' + str(sample_name.split('Lane')[1].split('_')[0]))
            merge = adata.obs.join(called_ids, how = 'left')
            merge["called"] = merge["n_id"].fillna(0).map(lambda n: n > 0)
            merge.drop_duplicates(inplace = True)
            adata = adata[adata.obs_names.isin(merge.index),:].copy()
            adata.obs = merge

            fdist_assigned = snap.metrics.frag_size_distr(adata[adata.obs.called], inplace = False)
            fdist_unassigned = snap.metrics.frag_size_distr(adata[~adata.obs.called], inplace = False)

            if plot:
                fig, ax = plt.subplots(1, 3, figsize=figsize)
                sns.histplot(data = merge, x = 'n_fragment', kde = True, hue = 'called', ax = ax[0])
                ax[0].set_xlabel("Unique fragments")
                ax[0].set_xlim([0,6e4])
                sns.histplot(data = merge, x = 'tsse', kde = True, hue = 'called', ax = ax[1])
                ax[1].set_xlabel("TSS enrichment")
                ax[2].bar(np.arange(len(fdist_unassigned) - 1), fdist_unassigned[1:] / 1e5, alpha = 0.5, label = 'False', rasterized = True)
                ax[2].plot(np.arange(len(fdist_unassigned) - 1), fdist_unassigned[1:] / 1e5, alpha = 0.8)
                ax[2].bar(np.arange(len(fdist_assigned) - 1), fdist_assigned[1:] / 1e5, alpha = 0.5, label = 'True', rasterized = True)
                ax[2].plot(np.arange(len(fdist_assigned) - 1), fdist_assigned[1:] / 1e5, alpha = 0.8)
                ax[2].legend(title = 'called')
                ax[2].set_xlabel('Fragment size (bp)')
                ax[2].set_ylabel('Fragments x 10^6')
                fig.suptitle(f"{sample_name} ATAC QC metrics by guide call")
                plt.tight_layout()
                if savefig is not None:
                    plt.savefig(savefig, dpi = 300)
                fig.show()

            self.atac_qc[sample_name] = merge

        return merge


    def combined_qc(self, sample_name, umi = 6, complexity = 6, fragment = 1000, tsse = 4, verbose = True,  plot_qc = True, n_fragment_xlim = (0,6e4)):

        """
        QC filtering of cells by assignment and GEX and ATAC metrics.

        Args:
            sample_name (str): Lane to filter
            umi (float): Minimum UMI count for GEX (log1p).
            complexity (float): Minimum unique genes count for GEX (log1p).
            fragment (float): Minimum unique fragments for ATAC.
            tsse (float): Minimum TSS enrichment for ATAC.
            verbose (bool): print summary statistics.
            plot_qc (bool): plot QC metrics after calculation.
        Returns:
            singlets_assigned (pd.DataFrame): Cells passing QC with assignment info.
            qc (pd.DataFrame): All cells with GEX and ATAC QC metrics.
        """
        
        gex = self.cut_qc_metrics_by_called(sample_name, mode = 'gex', plot = plot_qc)
        atac = self.cut_qc_metrics_by_called(sample_name, mode = 'atac', plot = plot_qc)
        qc = gex.join(atac.drop(["n_id", "called"], axis = 1), how = 'inner')
        singlets = qc.query(f"n_id == 1 and log1p_total_counts > {umi} and log1p_n_genes_by_counts > {complexity} and n_fragment > {fragment} and tsse >= {tsse}", engine = 'python')
        
        if verbose:
            print(f"{sample_name}:", flush = True)
            print(f"{len(singlets)} ID singlets passing QC", flush = True)
            print(f"{np.expm1(singlets.log1p_total_counts).mean():.2f} GEX UMI", flush = True)
            print(f"{np.expm1(singlets.log1p_n_genes_by_counts).mean():.2f} unique genes", flush = True)
            print(f"{singlets.n_fragment.mean():.2f} unique ATAC fragments", flush = True)

        id_files = [file for file in os.scandir(self.id_dir) if "called_ids" in file.name]
        called_ids = pd.read_csv([f for f in id_files if sample_name in f.name][0])
        called_ids['joint_id'] = called_ids.apply(lambda df: df['identity'].split('_')[0] + "_" + df['LB_identity'].split('_')[0] if df['LB_identity'] != "_" else df['identity'].split("_")[0], axis=1)
        called_ids['joint_id_ntc'] = called_ids.joint_id.map(lambda i: "NTC" if i.startswith("NTC") else i)
        called_ids['guide_target'] = called_ids.joint_id_ntc.map(lambda i: i.split("_")[0])
        called_ids = called_ids[['CB', 'guide_target', 'identity', 'LB_identity', 'joint_id', 'joint_id_ntc', 'UMI']].set_index("CB")
        called_ids.index = called_ids.index.map(lambda cb: cb + '-' + str(sample_name.split('Lane')[1].split('_')[0]))
        singlets_assigned = singlets.join(called_ids, how = 'left')
        assert len(singlets) == len(singlets_assigned)
        
        self.singlets_assigned[sample_name] = singlets_assigned
        return singlets_assigned, qc
    
    def create_cellpop(self, sample_name, low_expr_threshold = 0.1, plot_first = False, plot_regressed = False, znorm = True, log1p = True, total = 1e6):

        """
        Create a gene expression matrix for a single lane. Updates self.cellpops in place.

        Args:
            sample_name (str): Lane to create cellpop for.
            low_expr_threshold (float): Threshold for filtering genes used by ``perturbseq.strip_low_expression()``, lower includes more genes.
            plot_first (bool): Plot UMAP before regressing out cell cycle.
            plot_regressed (bool): Plot UMAP after regressing out cell cycle.
            znorm (bool): Normalize to gem group control with ``perturbseq``.
            log1p (bool): Normalize UMI counts and log1p-transform without z-normalization.
            total (int): Target sum for normalization.
        Returns:
            cellpop_normalized_sc (AnnData): Cellpop before cell cycle regression.
            cellpop_regressed (AnnData): Cellpop after cell cycle regression.
        """

        # cell cycle genes for cellpops
        S_genes, G2M_genes = self._get_cellcycle_genes()
        
        # add id info
        adata = sc.read_10x_h5(f"{self.experiment_dir}/{sample_name}/outs/filtered_feature_bc_matrix.h5")
        adata.var_names_make_unique()
        adata.obs.index = adata.obs.index.map(lambda cb: cb.split("-")[0] + '-' + str(sample_name.split('Lane')[1].split('_')[0]))
        adata.obs = adata.obs.join(self.singlets_assigned[sample_name][['guide_target', 'identity', 'LB_identity', 'joint_id', 'joint_id_ntc','UMI']], how = 'left')
        adata.obs = adata.obs.rename(columns = {'UMI': 'ID_UMI'})
        singlets = adata[adata.obs['identity'].notna(),:].copy()
        singlets.obs['gem_group'] = singlets.obs.index[0].split("-")[1]
        singlets_no_mt = singlets[:,~singlets.var_names.str.startswith("MT-")].copy()
        singlets_no_mt.obs['guide_identity'] = singlets_no_mt.obs.identity.map(lambda i: i.split("_")[0])

        # qc metrics for plotting
        singlets_no_mt.obs = singlets_no_mt.obs.join(self.gex_qc[sample_name][['log1p_total_counts', 'log1p_n_genes_by_counts', 'pct_counts_mito']], how = 'left')

        # switch to perturbseq
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            singlets_no_mt.obs['single_cell'] = True
            cellpop = CellPopulation(
                        pd.DataFrame(singlets_no_mt.X.todense().A, index=singlets_no_mt.obs.index, columns=singlets_no_mt.var.index), 
                        singlets_no_mt.obs, 
                        singlets_no_mt.var, 
                        calculate_statistics=True
            )
            strip_low_expression(cellpop, threshold = low_expr_threshold)

            if znorm:
                cellpop.normalized_matrix = normalize_to_gemgroup_control(cellpop, control_cells = "guide_identity == 'NTC'")
            
            cellpop_normalized_sc = ad.AnnData(
                                        obs = cellpop.cells, 
                                        var = cellpop.genes.query('in_matrix'), 
                                        X = cellpop.where(normalized = znorm, dropna = True)
            )

            if not znorm:
                sc.pp.normalize_total(cellpop_normalized_sc, target_sum = total)
                if log1p:
                    sc.pp.log1p(cellpop_normalized_sc)
    
        mask = np.isnan(cellpop_normalized_sc.X).any(axis=0)
        cellpop_normalized_sc = cellpop_normalized_sc[:, ~mask].copy()

        sc.tl.pca(cellpop_normalized_sc)
        sc.pp.neighbors(cellpop_normalized_sc)
        sc.tl.umap(cellpop_normalized_sc)
        sc.tl.leiden(cellpop_normalized_sc)
        sc.tl.score_genes_cell_cycle(cellpop_normalized_sc, s_genes = S_genes, g2m_genes = G2M_genes)

        cellpop_normalized_sc.obs['perturbed'] = cellpop_normalized_sc.obs['guide_identity'].map(lambda guide: 0 if guide == 'NTC' else 1)
        if plot_first:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sc.pl.umap(cellpop_normalized_sc, color = ['log1p_total_counts', 'log1p_n_genes_by_counts', 'ID_UMI', 'leiden', 'pct_counts_mito', 'phase', 'perturbed', 'joint_id'], ncols = 4)
        
        cellpop_regressed = cellpop_normalized_sc.copy()
        print("Regressing out cell cycle...")
        
        sc.pp.regress_out(cellpop_regressed, keys = ['pct_counts_mito', 'S_score', 'G2M_score'])
        sc.tl.pca(cellpop_regressed)
        sc.pp.neighbors(cellpop_regressed)
        sc.tl.umap(cellpop_regressed)
        sc.tl.leiden(cellpop_regressed)
        sc.tl.score_genes_cell_cycle(cellpop_regressed, s_genes = S_genes, g2m_genes = G2M_genes)

        if plot_regressed:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sc.pl.umap(cellpop_regressed, color = ['log1p_total_counts', 'log1p_n_genes_by_counts', 'ID_UMI', 'leiden', 'pct_counts_mito', 'phase', 'perturbed', 'joint_id'], ncols = 4)

        if znorm:
            self.cellpops[sample_name] = cellpop_regressed
        
        return cellpop_normalized_sc, cellpop_regressed

    def create_atac_mtx(self, sample_name, groupby = 'joint_id_ntc', plot = True, comps = 11):
        
        """
        Create an ATAC count matrix for a single lane. Updates self.atac_mtxs in place. Creates peak matrix and updates self.peak_mtxs in place.
        
        Args:
            sample_name (str): Lane to create cellpop for.
            groupby (str): Cell grouping for peak calling.
            plot (bool): Whether to plot UMAP after spectral embedding.
            comps (int): Components for spectral embedding. First component is removed before clustering (highly correlated with sequencing depth).

        Returns:
            atac (AnnData): ATAC count matrix for lane.
        """
        
        if not self.preprocessed_atac:
            print("Loading atac h5s...")
            self._load_preprocessed_atac()
        
        atac = self.preprocessed_atac[sample_name]
        atac = atac[atac.obs.index.isin(self.singlets_assigned[sample_name].index),atac.var.selected].copy()
        atac.obs = atac.obs.join(self.singlets_assigned[sample_name][['guide_target', 'identity', 'LB_identity', 'joint_id', 'joint_id_ntc', 'UMI']], how = 'left')
        atac.obs = atac.obs.rename(columns = {'UMI': 'ID_UMI'})
        atac.var = label_peaks(atac.var.reset_index().rename({"index":"peak_id"}, axis = 1))
        snap.tl.macs3(atac, groupby = groupby, n_jobs = 1) # more than one job breaks
        peaks = snap.tl.merge_peaks(atac.uns['macs3'], snap.genome.hg38)
        mtx = snap.pp.make_peak_matrix(atac, use_rep = peaks["Peaks"], counting_strategy='paired-insertion')
        self.peak_mtxs[sample_name] = mtx

        # clustering
        snap.tl.spectral(atac, n_comps = comps)
        df = pd.DataFrame(atac.obsm['X_spectral'], index = atac.obs.index)
        df_first_comp_removed = df.iloc[:,1:]

        self.atac_mtxs[sample_name] = atac

        # umap
        if plot:
            umap_transformer = UMAP(metric = 'cosine', random_state=8)
            umap_rep = pd.DataFrame(umap_transformer.fit_transform(df_first_comp_removed), index = df_first_comp_removed.index)
            atac_umap = umap_rep.join(atac.obs[groupby])
            plot_umap(atac_umap, hue = groupby, figsize = (8,6))

        # self.atac_mtxs[sample_name] = atac
        return atac

    def create_full_cellpop(self, low_expr_threshold = 0.1, plot_first = False, plot_regressed = False, znorm = True, log1p = True, total = 1e6, inplace = True):
        
        """
        Create the gene expression matrix for the full experiment. Store in ``self.gex`` if ``inplace``.

        Args:
            low_expr_threshold (float): Threshold for filtering genes used by ``perturbseq.strip_low_expression()``, lower includes more genes.
            plot_first (bool): Plot UMAP before regressing out cell cycle.
            plot_regressed (bool): Plot UMAP after regressing out cell cycle.
            znorm (bool): Normalize to gem group control with ``perturbseq``.
            log1p (bool): Normalize UMI counts and log1p-transform without z-normalization.
            total (int): Target sum for normalization.
            inplace (bool): Whether to update ``self.gex`` with the resulting GEX matrix.
        Returns:
            cellpop_normalized_sc (AnnData): Cellpop before cell cycle regression.
            cellpop_regresed (AnnData): Cellpop after cell cycle regression.
        """

        S_genes, G2M_genes = self._get_cellcycle_genes()

        singlets = pd.concat([self.singlets_assigned[sample] for sample in self.sample_names], axis = 0)
        print(f"Reading in count matrices, {len(singlets)} total cells...", flush = True)
        adatas = []
        for sample_name in tqdm(self.sample_names):
            adata = sc.read_10x_h5(f"{self.experiment_dir}/{sample_name}/outs/filtered_feature_bc_matrix.h5")
            adata.var_names_make_unique()
            adata.obs.index = adata.obs.index.map(lambda cb: cb.split("-")[0] + '-' + str(sample_name.split('Lane')[1].split('_')[0]))
            adata.obs = adata.obs.join(self.singlets_assigned[sample_name][['guide_target', 'identity', 'LB_identity', 'joint_id', 'joint_id_ntc','UMI']], how = 'left')
            adata.obs = adata.obs.rename(columns = {'UMI': 'ID_UMI'})
            singlets = adata[adata.obs['identity'].notna(),:].copy()
            singlets.obs['gem_group'] = singlets.obs.index[0].split("-")[1]
            singlets.obs['guide_identity'] = singlets.obs.identity.map(lambda i: i.split("_")[0])
            adatas.append(singlets)
        
        adata = ad.concat(adatas, join = 'outer', fill_value = 0)
        adata.var["mito"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, inplace=True, qc_vars = ["mito"])

        singlets_no_mt = adata[:,~adata.var_names.str.startswith("MT-")].copy()

        # switch to perturbseq
        print("Normalizing...")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            singlets_no_mt.obs['single_cell'] = True
            cellpop = CellPopulation(
                        pd.DataFrame(singlets_no_mt.X.todense().A, index=singlets_no_mt.obs.index, columns=singlets_no_mt.var.index), 
                        singlets_no_mt.obs, 
                        singlets_no_mt.var, 
                        calculate_statistics=True
            )
            strip_low_expression(cellpop, threshold = low_expr_threshold)
            if znorm:
                cellpop.normalized_matrix = normalize_to_gemgroup_control(cellpop, control_cells = "guide_identity == 'NTC'")
            
            cellpop_normalized_sc = ad.AnnData(
                                        obs = cellpop.cells, 
                                        var = cellpop.genes.query('in_matrix'), 
                                        X = cellpop.where(normalized = znorm)
            )
            
            valid_mask = ~np.isnan(cellpop_normalized_sc.X).any(axis=0)
            cellpop_normalized_sc = cellpop_normalized_sc[:, valid_mask].copy()

            if not znorm:
                sc.pp.normalize_total(cellpop_normalized_sc, target_sum = total)
                if log1p:
                    sc.pp.log1p(cellpop_normalized_sc)

        sc.tl.pca(cellpop_normalized_sc)
        sc.pp.neighbors(cellpop_normalized_sc)
        sc.tl.umap(cellpop_normalized_sc)
        sc.tl.leiden(cellpop_normalized_sc)
        sc.tl.score_genes_cell_cycle(cellpop_normalized_sc, s_genes = S_genes, g2m_genes = G2M_genes)

        cellpop_normalized_sc.obs['perturbed'] = cellpop_normalized_sc.obs['guide_identity'].map(lambda guide: 0 if guide == 'NTC' else 1)
        if plot_first:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sc.pl.umap(cellpop_normalized_sc, color = ['log1p_total_counts', 'log1p_n_genes_by_counts', 'ID_UMI', 'leiden', 'pct_counts_mito', 'phase', 'perturbed', 'gem_group'], ncols = 4)
        
        cellpop_regressed = cellpop_normalized_sc.copy()
        print("Regressing out cell cycle...")
        
        sc.pp.regress_out(cellpop_regressed, keys = ['pct_counts_mito', 'S_score', 'G2M_score'])
        sc.tl.pca(cellpop_regressed)
        sc.pp.neighbors(cellpop_regressed)
        sc.tl.umap(cellpop_regressed)
        sc.tl.leiden(cellpop_regressed)
        sc.tl.score_genes_cell_cycle(cellpop_regressed, s_genes = S_genes, g2m_genes = G2M_genes)

        if plot_regressed:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sc.pl.umap(cellpop_regressed, color = ['log1p_total_counts', 'log1p_n_genes_by_counts', 'ID_UMI', 'leiden', 'pct_counts_mito', 'phase', 'perturbed', 'gem_group'], ncols = 4)
        
        if inplace:
            self.gex = cellpop_regressed
        
        return cellpop_normalized_sc, cellpop_regressed

    
    def create_full_atac(self, groupby = 'joint_id_ntc', plot = True, comps = 30, harmony = True, **harmony_kwargs):

        """
        Create the ATAC count matrix and peak matrix for the full experiment. Updates ``self.atac`` in place.
        
        Args:
            groupby (str): Cell grouping for peak calling (this should be a feature in obs, like guide_target or joint_id_ntc).
            plot (bool): Whether to plot UMAP after spectral embedding.
            comps (int): Components for spectral embedding. First component is removed before clustering (highly correlated with sequencing depth).
            harmony (bool): Whether to run harmony batch correction on the spectral embedding before clustering.
            harmony_kwargs (dict): Additional arguments for harmony.

        Returns:
            atac (AnnData): ATAC count matrix for experiment.
        """
        
        self._load_preprocessed_atac()
        adatas = []
        print("Concatenating fragments...", flush = True)
        for sample_name in tqdm(self.sample_names):
            atac = self.preprocessed_atac[sample_name]
            atac.obs.index = atac.obs.index.map(lambda cb: cb.split("-")[0] + '-' + str(sample_name.split('Lane')[1].split('_')[0]))
            atac = atac[atac.obs.index.isin(self.singlets_assigned[sample_name].index),:].copy()
            atac.obs = atac.obs.join(self.singlets_assigned[sample_name][['guide_target', 'identity', 'LB_identity', 'joint_id', 'joint_id_ntc', 'UMI']], how = 'left')
            atac.obs = atac.obs.rename(columns = {'UMI': 'ID_UMI'})
            atac.var = label_peaks(atac.var.reset_index().rename({"index":"peak_id"}, axis = 1))
            adatas.append(atac)
        
        adata = ad.concat(adatas, join = 'outer', fill_value = 0, uns_merge = 'same')
        adata.uns['reference_sequences'] = self.preprocessed_atac[self.sample_names[0]].uns['reference_sequences']
        adata.obs['gem_group'] = adata.obs.index.map(lambda i: str(i.split("-")[1]))

        print("Calling peaks...", flush = True)
        snap.tl.macs3(adata, groupby = groupby)
        merged_peaks = snap.tl.merge_peaks(adata.uns['macs3'], snap.genome.hg38)
        self.peaks = snap.pp.make_peak_matrix(adata, use_rep = merged_peaks["Peaks"], counting_strategy='paired-insertion')

        # clustering
        print("Clustering...", flush = True)
        snap.pp.select_features(adata, n_features = 250000)
        snap.tl.spectral(adata, n_comps = comps)
        if harmony:
            if len(harmony_kwargs) == 0:
                snap.pp.harmony(adata, batch = 'gem_group', max_iter_harmony = 100, sigma = 0.5, use_dims = range(1,adata.obsm['X_spectral'].shape[1]), epsilon_harmony = -np.inf, epsilon_cluster = -np.inf)
            else:
                snap.pp.harmony(adata, **harmony_kwargs)

        # umap
        if plot:
            df = pd.DataFrame(adata.obsm['X_spectral_harmony'] if harmony else adata.obsm['X_spectral'], index = adata.obs.index)
            df_first_comp_removed = df.iloc[:,1:]
            umap_transformer = UMAP(metric = 'cosine', random_state=8)
            umap_rep = pd.DataFrame(umap_transformer.fit_transform(df_first_comp_removed), index = df_first_comp_removed.index)
            atac_umap = umap_rep.join(adata.obs[groupby])
            plot_umap(atac_umap, hue = groupby)

        self.atac = adata
        return adata
    
    def plot_program_enrichment(self, program_id, groupby = 'joint_id_ntc', filter_features = None, filter_perturbations = None, plot = True, savefig = None, figsize = (14,4), gene_cmap = 'mako', peak_cmap = 'rocket', gene_vmin = 0, peak_vmin = 0, gene_vmax = 2, peak_vmax = 2):

        if not (hasattr(self, 'programs') and hasattr(self, 'gex') and hasattr(self, 'peaks')):
            raise ValueError("Experiment object has no programs!")

        genes = self.programs['feature_weights'].query(f"program_id == {program_id} and feature_type == 'gene'").feature
        df_genes = self.gex[:,genes].to_df().join(self.gex.obs[groupby]).groupby(groupby).mean().drop("NTC", axis = 0)
        df_genes.columns = df_genes.columns.map(lambda col: col + "_g")

        peaks = self.programs['feature_weights'].query(f"program_id == {program_id} and feature_type == 'peak'").sort_values("weight").drop_duplicates("feature_peak_associated", keep = 'first').feature # only keep a single peak from each gene for visualization, can be turned off for exploration
        allpeaks = self.programs['feature_weights'].query(f"program_id == {program_id} and feature_type == 'peak'").feature
        
        mtx = self.peaks.copy()
        df_ntc = mtx[mtx.obs.joint_id_ntc == 'NTC', mtx.var_names.isin(allpeaks)].to_df()
        df_mtx = mtx[mtx.obs.joint_id_ntc != 'NTC', mtx.var_names.isin(allpeaks)].to_df()
        ctrl_mean, ctrl_std = df_ntc.mean(axis = 0), df_ntc.std(axis = 0)
        df_mtx_normalized = df_mtx.sub(ctrl_mean).div(ctrl_std)
        normalized_peaks = df_mtx_normalized.join(mtx.obs[groupby], how = 'inner').groupby(groupby).mean()

        peak_dict = dict(zip(self.peaks.var.index, self.peaks.var.gene_name))
        df_peaks = normalized_peaks[peaks].copy()
        df_peaks.columns = df_peaks.columns.map(lambda s: peak_dict.get(s, np.nan)).map(lambda col: col + "_p")
        df_allpeaks = normalized_peaks[allpeaks].copy()

        score = np.dot(pd.concat([df_genes, df_allpeaks], axis = 1), self.programs['feature_weights'].query(f"program_id == {program_id}").weight)
        df_score = pd.DataFrame(score, index = df_allpeaks.index, columns = ['score'])

        df = pd.concat([df_genes, df_peaks, df_score], axis = 1).sort_values("score", ascending = False)
        if filter_features is not None:
            if filter_perturbations is not None:
                cols = [col for col in df.columns if any([f in col for f in filter_features]) or col == 'score']
                df = df[cols].loc[filter_perturbations]
            else:
                cols = [col for col in df.columns if any([f in col for f in filter_features]) or col == 'score']
                df = df[cols]
        else:
            if filter_perturbations is not None:
                df = df.loc[filter_perturbations]
        
        df_genes = df.copy()
        df_genes[df.filter(like = "_p").columns.tolist() + df.filter(like = "score").columns.tolist()] = np.nan
        df_genes.rename(columns = lambda col: col.split("_")[0] if "_" in col else col, inplace = True)

        df_peaks = df.copy()
        df_peaks[df.filter(like = '_g').columns.tolist() + df.filter(like = "score").columns.tolist()] = np.nan
        df_peaks.columns = df_peaks.columns.map(lambda col: col.split("_")[0] if "_" in col else col)

        df_score = df.copy()
        df_score.iloc[:,:-1] = np.nan
        df_score.columns = df_score.columns.map(lambda col: col.split("_")[0] if "_" in col else col)
        
        if plot:

            plt.figure(figsize = figsize)
            sns.heatmap(df_genes.drop("score", axis = 1), cmap = gene_cmap, cbar = False, vmin = gene_vmin, vmax = gene_vmax)
            sns.heatmap(df_peaks.drop("score", axis = 1), cmap = peak_cmap, cbar = False, vmin = peak_vmin, vmax = peak_vmax)
            # sns.heatmap(df_score, cmap = 'Spectral_r', cbar = False, center = 0)
            plt.tight_layout()
            if savefig is not None:
                plt.savefig(savefig)
            plt.show()

        return df_genes, df_peaks


class CoverageVisualizer:

    """
    Visualize ATAC coverage, expression, and autogluon feature importance for pseudobulk populations.

    Args:
        pop (AnnData): Single-cell gex matrix (like ``expt.gex``.)
        bw_dir (str): Directory with bigwig ATAC coverage files for each pseudobulk group.
        groupby (str): Grouping for pseudobulk populations, should be a feature in ``pop.obs`` and its values should match the filenames in ``bw_dir``.
        gtf (str): Path to gtf file for gene annotations.
        autogluon_outdir (str): Path to directory with autogluon feature importance csv files, if you want to visualize feature importance.

    """

    def __init__(self, pop, bw_dir, groupby, gtf = None, autogluon_outdir = None):

        if gtf is not None:
            self.gtf = gtf
        else:
            self.gtf = pr.read_gtf('/fscratch/eli/genomes/refdata-gex-GRCh38-2020-A/genes/genes.gtf')
        self.pop = pop
        self.bw_dir = bw_dir
        self.groupby = groupby
        self.autogluon_outdir = autogluon_outdir

        # np_version = float(np.__version__[:4])
        # if np_version > 1.24:
        #     print("Numpy version is greater than 1.24 which might break plotting functions!")
    
    def _plot_genomic_region(self, chrm, start, end, ax):
        
        """
        Adapted from https://github.com/snehamitra/SCARlink/blob/main/scarlink/src/plotExtra.py
        """
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_region = self.gtf.df.query(f"Chromosome == '{chrm}' and Start <= {end} and End >= {start} and gene_type == 'protein_coding'")

            transcripts = {}
            for i, r in df_region.iterrows():
                if r.Feature == 'transcript' and r[11].find('.') == -1 and r.transcript_type == 'protein_coding':
                    gene = r[11]
                    if r[6] == '+':
                        ax.add_patch(patches.Rectangle((r[3], 0.45), r[4] - r[3] + 1, 0.2, color = sns.color_palette("icefire",8)[0]))
                    else:
                        ax.add_patch(patches.Rectangle((r[3], -0.65), r[4] - r[3] + 1, 0.2, color = sns.color_palette("icefire",10)[-1]))
                    if gene not in transcripts:
                        transcripts[gene] = (r[3], r[4], r[6])
                    else:
                        transcripts[gene] = (min(r[3], transcripts[gene][0]), max(r[4], transcripts[gene][1]), r[6])
                elif r[2] == 'exon' and r[11].find('.') == -1 and r.transcript_type == 'protein_coding': 
    
                    gene = r[11]
                    if r[6] == '+':
                        ax.add_patch(patches.Rectangle((r[3], 0.1), r[4] - r[3] + 1, 0.9, color = sns.color_palette("icefire",8)[0]))
                    else:
                        ax.add_patch(patches.Rectangle((r[3], -1), r[4] - r[3] + 1, 0.9, color = sns.color_palette("icefire",8)[-1]))
                
            for t in transcripts:
                if transcripts[t][2] == '+':
                    if transcripts[t][0] > start:
                        ax.text(transcripts[t][0] + 10, 1.2, t, style = 'italic')
                    else:
                        ax.text(start + 10, 1.2, t, style = 'italic')
                else:
                    if transcripts[t][0] > start:
                        ax.text(transcripts[t][0] + 10, -3, t, style = 'italic')
                    else:
                        ax.text(start, -3, t, style = 'italic')

            ax.set_xlim((start, end))
            ax.set_ylim((-1.9, 1.9))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False) # new
            ax.spines['right'].set_visible(False) # new
            ax.spines['left'].set_visible(False) # new
            ax.spines['bottom'].set_visible(False) # new
    
    def plot_coverage(self, gene, locus = None, atac_window = 20, savefig = False, tick_window = 100, vlim = (-2,2), cdict = None, select_groups = None, figsize = (10,10), fill_between = 0):
            
        """
        Plot ATAC coverage across a locus.

        Args:
            gene (str): Gene name to visualize, if no locus specified the locus will be pulled from ``self.gtf`` and will include the whole gene body.
            locus (tuple): A tuple of (chr, start, end) like ("chr1", 10000, 20000) to visualize a specific locus.
            atac_window (int): Window for downsampling ATAC coverage, e.g., draw a point on the coverage plot every atac_window bp.
            savefig (str): Path to save figure, if False or None displays figure only.
            tick_window (int): Window for x-axis ticks in kb.
            vlim (tuple): axis limits for expression violin plot.
            cdict (dict): Color mapping for groups. Determines color of ATAC lineplot and expression violins. Like {"NTC": "#bfbfbf", "Guide1": "#357ba3"}.
            select_groups (list): List of groups to plot, if None plots all groups.
            figsize (tuple): Figure size.
            fill_between (float): Alpha for shading below ATAC line plot, if 0 only the line is plotted.
        """


        # define locus
        if locus is None:
            df_gene = self.gtf.df.query("Feature == 'gene' and gene_name == @gene")
            assert len(df_gene) == 1
            chrom = df_gene.Chromosome.item()
            start = int(np.round(df_gene.Start.item() / 500) * 500 - 1e4)
            end = int(np.round(df_gene.End.item() / 500) * 500 + 1e4)
        else:
            chrom, start, end = locus

        # get regulatory elements
        df_cre = pd.DataFrame({"Chromosome": chrom, "Start": np.arange(start, end, 500), "End": np.arange(start + 500, end + 500, 500)})
        df_cre['peak_id'] = df_cre.apply(lambda df: df.Chromosome + ":" + str(df.Start) + "-" + str(df.End), axis = 1)
        df_cre = label_peaks(df_cre, gtf = self.gtf)[['Start', 'reg']].sort_values("Start").set_index("Start")
        df_cre.index = df_cre.index.astype(int) / 1000
        
        # get bigwig coverage
        df_coverage = pd.DataFrame({'bp': np.arange(start, end)})
        for file in os.scandir(self.bw_dir):
            if select_groups is None:
                if file.name.endswith(".bw"):
                    coverage = bw.open(file.path)
                    df_coverage[file.name.split(".")[0]] = coverage.values(chrom, start, end)
            else:
                if file.name.endswith(".bw") and file.name.split(".")[0] in select_groups:
                    coverage = bw.open(file.path)
                    df_coverage[file.name.split(".")[0]] = coverage.values(chrom, start, end)
        df_coverage_rolling = df_coverage.iloc[::atac_window,:]
        ngroups = len(df_coverage_rolling.columns[1:])

        cdict = dict(zip(df_coverage_rolling.columns[1:], ['#bfbfbf' if col == 'NTC' else '#357ba3' for col in df_coverage_rolling.columns[1:]])) if cdict is None else cdict

        # get expression
        df_expr = self.pop.to_df().join(self.pop.obs[self.groupby]) if select_groups is None else self.pop.to_df().join(self.pop.obs[self.groupby]).query(f"{self.groupby} in @select_groups")
        try:
            sorted_expr = df_expr[[gene, self.groupby]].copy().groupby(self.groupby).median().sort_values(gene, ascending = False) if select_groups is None else df_expr[[gene, self.groupby]].copy().groupby(self.groupby).median().sort_values(gene, ascending = False).loc[select_groups,:]
        except:
            df_expr = pd.DataFrame({self.groupby: self.pop.obs[self.groupby], gene: 0}) if select_groups is None else pd.DataFrame({self.groupby: self.pop.obs[self.groupby], gene: 0}).query(f"{self.groupby} in @select_groups")
            sorted_expr = df_expr[[gene, self.groupby]].copy().groupby(self.groupby).median().sort_values(gene, ascending = False) if select_groups is None else df_expr[[gene, self.groupby]].copy().groupby(self.groupby).median().sort_values(gene, ascending = False).loc[select_groups,:]
        finally:
            order = sorted_expr.index.tolist()

        # get reg values
        reg_color_dict = {'enhP':'#86c769', 'enhD':'#558f8c', 'prom':'#f7e642', 'CTCF': '#444684', 'K4m3': '#c0df25', 'nan':'#ffffff'}
        cmap = ListedColormap(reg_color_dict.values())
        reg_color_dict = dict(zip(reg_color_dict.keys(), np.arange(len(reg_color_dict.keys()))))
        df_cre = df_cre.replace(reg_color_dict).T

        # make plot
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(ngroups + 2,2, height_ratios = [5 for i in range(ngroups)] + [1.5, 0.75], width_ratios=[6,1], wspace = 0)
        axes  = []
        for i, tgt in enumerate(order):
            ax = fig.add_subplot(gs[i,0])
            sns.lineplot(df_coverage_rolling, x = 'bp', y = tgt, ax = ax, linewidth = 2, color = cdict[tgt],)
            if fill_between:
                plt.fill_between(df_coverage_rolling['bp'], df_coverage_rolling[tgt], color = cdict[tgt], alpha = fill_between)
            ax.text(s = tgt, x = 0.01, y = 0.75, fontsize = 12, fontweight = 'bold', color = cdict[tgt], transform=ax.transAxes)
            ax.set_ylabel('')
            axes.append(ax)

        violin_colors = [cdict[tgt] for tgt in order]
        ax_v = fig.add_subplot(gs[0:ngroups,1])
        sns.violinplot(data = df_expr, y=self.groupby, x = gene, alpha = 0.8, orient = 'h', ax = ax_v, legend = False, order = order,cut=0, palette = violin_colors, scale = 'width', inner_kws = {'box_width': 6})
        for patch in ax_v.collections:
            patch.set_alpha(0.8)
        ax_v.axvline(0, color = 'grey', linestyle = '--')

        ax_genes = fig.add_subplot(gs[ngroups,0])
        self._plot_genomic_region(chrom, start, end, ax_genes)

        ax_cre = fig.add_subplot(gs[ngroups + 1,0])
        sns.heatmap(df_cre, ax = ax_cre, cmap = cmap, cbar = False)

        for ax in axes:
            ax.set_xlim(start, end)
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.set_ylim(0,df_coverage.drop("bp", axis = 1).max().max())
            ax.set_yticks([])
            sns.despine(ax = ax)

        ax_genes.axis('off')
        
        ax_v.get_yaxis().set_visible(False)
        ax_v.spines['left'].set_visible(False)
        ax_v.spines['top'].set_visible(True)
        ax_v.spines['bottom'].set_visible(False)
        ax_v.spines['right'].set_visible(False)
        ax_v.xaxis.set_ticks_position('top')
        ax_v.xaxis.set_label_position('top')
        ax_v.set_xlabel("Z-norm. expr.")
        ax_v.set_xlim(vlim)

        ticks = df_cre.columns[::tick_window]
        tick_positions = np.arange(0, df_cre.shape[1], tick_window)
        ax_cre.set_xticks(tick_positions)
        ax_cre.set_xticklabels([f'{int(x):,}' for x in ticks], rotation = 0)
        ax_cre.set_xlabel(f"Chromosome {chrom.split('r')[1]} (kb)")
        ax_cre.spines['bottom'].set_visible(True)

        if savefig:   
            plt.savefig(savefig, transparent = True)
        plt.show()


    def plot_autogluon_result(self, gene, locus = None, atac_window = 20, savefig = False, scale_feature_importance = True, feature_vmax = 2, tick_window = 200, vlim = (-2,2), cdict = None, select_groups = None, figsize = (10,10), fill_between = 0):

        """
        Plot ATAC coverage and Autogluon feature importance across a locus.

        Args:
            gene (str): Gene name to visualize, if no locus specified the locus will be pulled from ``self.gtf`` and will include the whole gene body.
            locus (tuple): A tuple of (chr, start, end) like ("chr1", 10000, 20000) to visualize a specific locus.
            atac_window (int): Window for downsampling ATAC coverage, e.g., draw a point on the coverage plot every atac_window bp.
            savefig (str): Path to save figure, if False or None displays figure only.
            scale_feature_importance (bool): Whether to z-normalize feature importance values.
            feature_vmax (float): Maximum value for feature importance color scale.
            tick_window (int): Window for x-axis ticks in kb.
            vlim (tuple): axis limits for expression violin plot.
            cdict (dict): Color mapping for groups. Determines color of ATAC lineplot and expression violins. Like {"NTC": "#bfbfbf", "Guide1": "#357ba3"}.
            select_groups (list): List of groups to plot, if None plots all groups.
            figsize (tuple): Figure size.
            fill_between (float): Alpha for shading below ATAC line plot, if 0 only the line is plotted.
        """

        if self.autogluon_outdir is None:
            raise ValueError("no autogluon outdir provided!")
        
        # get feature importance
        df = pd.read_csv(os.path.join(self.autogluon_outdir, f"{gene}_feature_importance.csv"), index_col = 0)
        
        df = df.sort_values(["importance", "Overlap"], ascending = False).reset_index()
        df['feature_id'] = df.apply(lambda df: df.iloc[0] if type(df.Chromosome) is float else f"{str(df.Chromosome)}:{str(int(df.Start))}-{str(int(df.End))}", axis = 1)
        df = df.drop('index', axis = 1).set_index('feature_id')
        df_filtered = df.reset_index().drop_duplicates('feature_id', keep = 'first')
        df_filtered = df_filtered.query("feature_id.str.startswith('chr')").sort_values("feature_id")
        df_filtered['adjusted_importance'] = df_filtered.apply(lambda df: df.importance if df.fdr < 0.1 and df.importance >=0 else 0, axis = 1)
        if scale_feature_importance:
            scaler = StandardScaler()
            df_filtered['adjusted_importance'] = scaler.fit_transform(df_filtered['adjusted_importance'].to_numpy().reshape(-1, 1))

        # define locus
        if locus is None:
            chrom = df_filtered.iloc[0,:].Chromosome
            start = int(df_filtered.iloc[0,:].Start)
            end = int(df_filtered.iloc[-1,:].End)
        else:
            chrom, start, end = locus

        # make relevant dfs for plotting
        df_heatmap = df_filtered[['Start', 'End', 'adjusted_importance']].query("Start >= @start and End <= @end").set_index("Start").drop("End", axis = 1).T
        df_cre = df_filtered[['Start', 'End', 'reg']].query("Start >= @start and End <= @end").set_index("Start").drop("End", axis = 1)
        df_cre.index = df_cre.index.astype(int) / 1000

        # get bigwig coverage
        df_coverage = pd.DataFrame({'bp': np.arange(start, end)})
        for file in os.scandir(self.bw_dir):
            if select_groups is None:
                if file.name.endswith(".bw"):
                    coverage = bw.open(file.path)
                    df_coverage[file.name.split(".")[0]] = coverage.values(chrom, start, end)
            else:
                if file.name.endswith(".bw") and file.name.split(".")[0] in select_groups:
                    coverage = bw.open(file.path)
                    df_coverage[file.name.split(".")[0]] = coverage.values(chrom, start, end)
        df_coverage_rolling = df_coverage.iloc[::atac_window,:]
        ngroups = len(df_coverage_rolling.columns[1:])

        cdict = dict(zip(df_coverage_rolling.columns[1:], ['#bfbfbf' if col == 'NTC' else '#357ba3' for col in df_coverage_rolling.columns[1:]])) if cdict is None else cdict

        # get expression
        df_expr = self.pop.to_df().join(self.pop.obs[self.groupby]) if select_groups is None else self.pop.to_df().join(self.pop.obs[self.groupby]).query(f"{self.groupby} in @select_groups")
        try:
            sorted_expr = df_expr[[gene, self.groupby]].copy().groupby(self.groupby).median().sort_values(gene, ascending = False) if select_groups is None else df_expr[[gene, self.groupby]].copy().groupby(self.groupby).median().sort_values(gene, ascending = False).loc[select_groups,:]
        except:
            df_expr = pd.DataFrame({self.groupby: self.pop.obs[self.groupby], gene: 0}) if select_groups is None else pd.DataFrame({self.groupby: self.pop.obs[self.groupby], gene: 0}).query(f"{self.groupby} in @select_groups")
            sorted_expr = df_expr[[gene, self.groupby]].copy().groupby(self.groupby).median().sort_values(gene, ascending = False) if select_groups is None else df_expr[[gene, self.groupby]].copy().groupby(self.groupby).median().sort_values(gene, ascending = False).loc[select_groups,:]
        finally:
            order = sorted_expr.index.tolist()

        reg_color_dict = {'enhP':'#86c769', 'enhD':'#558f8c', 'prom':'#f7e642', 'CTCF': '#444684', 'K4m3': '#c0df25', '-1':'#ffffff'}
        cmap = ListedColormap(reg_color_dict.values())
        reg_color_dict = dict(zip(reg_color_dict.keys(), np.arange(len(reg_color_dict.keys()))))
        df_cre = df_cre.replace(reg_color_dict).T

        # make plot
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(ngroups + 3,2, height_ratios = [5 for i in range(ngroups)] + [2, 1.5, 0.75], width_ratios=[6,1], wspace = 0)
        axes  = []
        for i, tgt in enumerate(order):
            ax = fig.add_subplot(gs[i,0])
            sns.lineplot(df_coverage_rolling, x = 'bp', y = tgt, ax = ax, linewidth = 2, color = cdict[tgt],)
            if fill_between:
                plt.fill_between(df_coverage_rolling['bp'], df_coverage_rolling[tgt], color = cdict[tgt], alpha = fill_between)
            ax.text(s = tgt, x = 0.01, y = 0.75, fontsize = 12, fontweight = 'bold', color = cdict[tgt], transform=ax.transAxes)
            ax.set_ylabel('')
            axes.append(ax)

        violin_colors = [cdict[tgt] for tgt in order]
        ax_v = fig.add_subplot(gs[0:ngroups,1])
        sns.violinplot(data = df_expr, y=self.groupby, x = gene, orient = 'h', saturation = 1, ax = ax_v, legend = False, order = order,cut=0, palette = violin_colors, scale = 'width', inner_kws = {'box_width': 6})
        for patch in ax_v.collections:
            patch.set_alpha(0.8)

        ax_v.axvline(0, color = 'grey', linestyle = '--')

        ax_features = fig.add_subplot(gs[ngroups,0])
        sns.heatmap(df_heatmap, ax = ax_features, cmap = 'PuRd', cbar = False, vmax = feature_vmax, vmin = 0)

        ax_genes = fig.add_subplot(gs[ngroups+1,0])
        self._plot_genomic_region(chrom, start, end, ax_genes)

        ax_cre = fig.add_subplot(gs[ngroups+2,0])
        sns.heatmap(df_cre, ax = ax_cre, cmap = cmap, cbar = False)

        for ax in axes:
            ax.set_xlim(start, end)
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.set_ylim(0,df_coverage.drop("bp", axis = 1).max().max())
            ax.set_yticks([])
            sns.despine(ax = ax)

        ax_features.axis('off')
        ax_genes.axis('off')
        
        ax_v.get_yaxis().set_visible(False)
        ax_v.spines['left'].set_visible(False)
        ax_v.spines['top'].set_visible(True)
        ax_v.spines['bottom'].set_visible(False)
        ax_v.spines['right'].set_visible(False)
        ax_v.xaxis.set_ticks_position('top')
        ax_v.xaxis.set_label_position('top')
        ax_v.set_xlabel("Z-norm. expr.")
        ax_v.set_xlim(vlim)

        ticks = df_cre.columns[::tick_window]
        tick_positions = np.arange(0, df_cre.shape[1], tick_window)
        ax_cre.set_xticks(tick_positions)
        ax_cre.set_xticklabels([f'{int(x):,}' for x in ticks], rotation = 0)
        ax_cre.set_xlabel(f"Chromosome {chrom.split('r')[1]} (kb)")
        ax_cre.spines['bottom'].set_visible(True)

        # plt.tight_layout()
        if savefig:
            plt.savefig(savefig, transparent = True, bbox_inches = 'tight')
        plt.show()


def label_peaks(peaks, gtf = None):

    """
    Label a dataframe of peaks as strings 'chr:start-end' with the nearest gene and CRE overlap if any.

    Args:
        peaks (pd.DataFrame): DataFrame with 'peak_id' column containing peak strings like 'chr1:1000-2000'.
        gtf (str): Path to gtf file for gene annotations. To label a lot of peaks, provide the gtf file as a pyranges object so it doesn't need to be opened every time.
    Returns:
        reg_mapped (pd.DataFrame): DataFrame mapping each peak to nearest gene and regulatory element.
    """
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        if 'peak_id' not in peaks.columns or not peaks.peak_id.str.contains(":").all() or not peaks.peak_id.str.startswith("chr").all() or not peaks.peak_id.str.contains("-").all():
            raise ValueError("expects a df with a string like 'chr1:1000-2000' in the 'peak_id' column")
        
        orig_ids = peaks['peak_id'].values.copy()
        peaks[['Chromosome', 'pos']] = peaks.peak_id.str.split(":", expand = True)
        peaks[['Start', 'End']] = peaks.pos.str.split("-", expand = True)
        peaks = pr.PyRanges(peaks.drop("pos", axis = 1))
        
        gtf = pr.read_gtf('/data1/normantm/eli/seq2gex/data/genome/genes.gtf')[["Chromosome", "Feature", "Start", "End", "gene_id", "gene_type", "gene_name"]] if gtf is None else gtf
        gtf = gtf[gtf.Feature == 'gene']
        df_nearest_gene = peaks.nearest(gtf)
        df_nearest_gene = pr.PyRanges(df_nearest_gene.df.drop(["Feature", "Start_b", "End_b", "Strand", 'gene_id', 'gene_type'], axis = 1))

        reg_elements = pd.read_csv("/data1/normantm/eli/seq2gex/data/genome/encodeCcreCombined.bed", sep = '\t', header = None).iloc[:,[0,1,2,12]]
        reg_elements.columns = ["Chromosome", "Start", "End", "reg"]
        reg_elements = reg_elements.query("reg != 'prom'")
        promoters = pd.read_csv("/data1/normantm/eli/seq2gex/data/genome/epdNewHuman006_extended_promoter_regions.bed", sep = '\t', header = None).iloc[:,:3]
        promoters.columns = ["Chromosome", "Start", "End"]
        promoters['reg'] = 'prom'
        reg_elements = pr.PyRanges(pd.concat([reg_elements, promoters], axis = 0))

        reg_mapped = df_nearest_gene.join(reg_elements, how = 'left', report_overlap=True).df.drop(["Start_b", "End_b"], axis = 1)
        reg_mapped = reg_mapped.sort_values(["Overlap"], ascending = False).drop_duplicates('peak_id', keep = 'first')

        reg_mapped['reg'] = reg_mapped.reg.map(lambda r: np.nan if r == '-1' else r).astype(str)
        reg_mapped['edited_peak_annotation'] = reg_mapped.reg.map(lambda a: 'prom' if 'prom' in a else 'enhP' if 'enhP' in a else 'enhD' if 'enhD' in a else 'CTCF' if 'CTCF' in a else 'K4m3' if 'K4m3' in a else np.nan)
        reg_mapped['edited_peak_annotation'] = reg_mapped.apply(lambda df: 'genic_unlabeled' if df.edited_peak_annotation is np.nan and df.Distance == 0 else 'intergenic_unlabeled' if df.edited_peak_annotation is np.nan else df.edited_peak_annotation, axis = 1)

    return reg_mapped.set_index("peak_id").reindex(orig_ids)


def normalize_atac(mtx, control_key = 'NTC'):

    df_ntc = mtx[mtx.obs.joint_id_ntc == control_key].to_df()

    ctrl_mean, ctrl_std = df_ntc.mean(axis = 0), df_ntc.std(axis = 0)
    df_mtx_normalized = mtx.to_df().sub(ctrl_mean).div(ctrl_std)

    return df_mtx_normalized


def get_differential_peaks(mtx, peaks, guide_target, control_key='NTC', mode='mwu', groupby='guide_identity'):

    """
    Calculate differential peaks from a peak matrix.

    Args:
        mtx (pd.DataFrame): Peak matrix, like expt.peaks.
        peaks (pd.DataFrame): Matrix containing logical values for which peaks are present in which groups (this is the output of snap.tl.merge_peaks())
        guide_target (str): group to test differential enrichment on (can parallelize by calling this function separately for each target).
        control_key (str): group to use as control.
        groupby (str): feature in mtx.obs containing cell groups delineating target and control, e.g., guide_target and control_key should be values of this feature.
        mode (str): One of 'mwu' (Mann-Whitney), 'snapatac2' (SnapATAC2 regression test, much more conservative and does not produce great results on Perturb-seq), 'binom' (binomial test, comparison only)
    Returns:
        diff_peaks (pd.DataFrame): DataFrame listing l2fc (log2 fold change) and Mann-Whitney q values for differential peaks.
    """

    selected_peaks = np.logical_or(peaks[guide_target].to_numpy(), peaks[control_key].to_numpy())

    if mode == 'snapatac2':

        ntc_cells = mtx.obs[groupby] == control_key
        target_cells = mtx.obs[groupby] == guide_target
        assert target_cells.sum() > 0

        diff_peaks = snap.tl.diff_test(mtx, cell_group1=target_cells, cell_group2=ntc_cells, features=selected_peaks).to_pandas()
        diff_peaks.columns = ['feature', 'l2fc', 'p', 'q']
        diff_peaks['logq'] = -np.log10(diff_peaks.q)
        diff_peaks['guide_target'] = guide_target
        diff_peaks['n_cells'] = target_cells.sum()
        diff_peaks['mode'] = 'snapatac2'

    elif mode == 'mwu':

        if not isinstance(peaks, pd.DataFrame):
            peaks = peaks.to_pandas()
        
        peak_names = peaks.query(f"{guide_target} or {control_key}").Peaks

        target_mtx = mtx[mtx.obs[groupby] == guide_target, selected_peaks].to_df()
        ctrl_mtx = mtx[mtx.obs[groupby] == control_key, selected_peaks].to_df()
        target_mtx_norm = target_mtx.div(mtx[mtx.obs[groupby] == guide_target].obs.n_fragment, axis=0).mul(1e6).to_numpy()
        ctrl_mtx_norm = ctrl_mtx.div(mtx[mtx.obs[groupby] == control_key].obs.n_fragment, axis=0).mul(1e6).to_numpy()

        ps = mannwhitneyu(target_mtx_norm, ctrl_mtx_norm, axis=0).pvalue
        lfcs = np.log2(np.divide(1e-3 + target_mtx_norm.mean(axis=0), 1e-3 + ctrl_mtx_norm.mean(axis=0)))
        adj_ps = false_discovery_control(ps)

        diff_peaks = pd.DataFrame({'peak_id': peak_names, 'l2fc': lfcs, 'p': ps, 'q': adj_ps, 'guide_target': guide_target, 'n_cells': target_mtx.shape[0], 'mode': 'mwu'})

    elif mode == 'brunnermunzel':

        if not isinstance(peaks, pd.DataFrame):
            peaks = peaks.to_pandas()
        
        peak_names = peaks.query(f"{guide_target} or {control_key}").Peaks

        target_mtx = mtx[mtx.obs[groupby] == guide_target, selected_peaks].to_df()
        ctrl_mtx = mtx[mtx.obs[groupby] == control_key, selected_peaks].to_df()
        target_mtx_norm = target_mtx.div(mtx[mtx.obs[groupby] == guide_target].obs.n_fragment, axis=0).mul(1e6).to_numpy()
        ctrl_mtx_norm = ctrl_mtx.div(mtx[mtx.obs[groupby] == control_key].obs.n_fragment, axis=0).mul(1e6).to_numpy()

        ps = brunnermunzel(target_mtx_norm, ctrl_mtx_norm, axis=0, distribution = 'normal').pvalue
        ps = np.where(np.isnan(ps), 1.0, ps)
        lfcs = np.log2(np.divide(1e-3 + target_mtx_norm.mean(axis=0), 1e-3 + ctrl_mtx_norm.mean(axis=0)))
        adj_ps = false_discovery_control(ps)

        diff_peaks = pd.DataFrame({'peak_id': peak_names, 'l2fc': lfcs, 'p': ps, 'q': adj_ps, 'guide_target': guide_target, 'n_cells': target_mtx.shape[0], 'mode': 'brunnermunzel'})

    elif mode == 'binom':

        peak_names = peaks.query(f"{guide_target} or NTC").Peaks
        target_mtx = mtx[mtx.obs[groupby] == guide_target, selected_peaks].X.toarray()
        ctrl_mtx = mtx[mtx.obs[groupby] == "NTC", selected_peaks].X.toarray()

        probs = (ctrl_mtx > 0).astype(int).mean(axis=0)
        ks = (target_mtx > 0).astype(int).sum(axis=0)
        n = target_mtx.shape[0]
        ps = [binomtest(k, p, n) for k, p in zip(ks, probs)]
        adj_ps = false_discovery_control(ps)
        print("only returning adjusted p values for comparison")
        return adj_ps

    else:
        raise ValueError("Invalid mode!")

    return diff_peaks


def get_differential_genes(pop, key = 'guide_identity', control_key = "guide_identity == 'NTC'", multi_method = 'fdr_bh', return_all = False, n_jobs = 16):

    """
    Calculate differential genes from a CellPopulation or AnnData. Wraps ``perturbseq.ks_de()``.

    Args:
        pop: cell population, like ``expt.gex``.
        control_key (str): group to use as control, written like a pandas query, e.g. ("guide_identity == 'NTC'").
        key (str): feature in pop.obs containing cell groups delineating target and control, e.g., control_key should be a value of this feature.
        multi_method (str): Method for multiple hypothesis correction, see perturbseq.ks_de().
    Returns:
        df (pd.DataFrame): DataFrame listing z-scores and KS q values for differential genes.
    """
    
    if not isinstance(pop, CellPopulation):
        pop = CellPopulation(pd.DataFrame(pop.X, index = pop.obs.index, columns = pop.var.index), pop.obs, pop.var, calculate_statistics=False)
        pop.genes['gene_name'] = pop.genes.index
        pop.genes['in_matrix'] = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ks, p, adjp = ks_de(
            pop, key = key,
            control_cells = control_key, 
            n_jobs = n_jobs,
            multi_method = multi_method
        )

        meanpop = pop.average(key)
        if "NTC" in adjp.columns:
            adjp = adjp.drop("NTC", axis = 1)
        dfs = []
        for col in adjp.columns:
            if not return_all:
                degs = adjp[[col]].query(f"{col} < 0.1").join(meanpop.matrix.T[[col]], how = 'inner', lsuffix = '_q', rsuffix = '_z').reset_index()
            else:
                degs = adjp[[col]].join(meanpop.matrix.T[[col]], how = 'inner', lsuffix = '_q', rsuffix = '_z').reset_index()
            degs['guide_target'] = col
            df = degs.melt(id_vars = ['index', 'guide_target'], var_name = 'gene', value_name = 'score')
            df['scoretype'] = df.gene.map(lambda s: s.split("_")[-1])
            df = df.drop("gene", axis = 1)
            df.columns = ['gene', 'guide_target', 'score', 'scoretype']
            dfs.append(df.pivot(index=['gene', 'guide_target'], columns='scoretype', values='score').reset_index().rename_axis(None, axis=1))
        df = pd.concat(dfs)

    return df


def cluster(adata, drop_ntc = True, groupby = 'guide_target', metric = 'correlation', use_rep = 'X_pca', starting_comp = 0, **cg_kwargs):

    """
    Perform hierarchical clustering on mean populations within an AnnData object.

    Args:
        adata (AnnData): object to cluster, like expt.gex or expt.atac.
        drop_ntc (bool): Whether to remove control cells before clustering.
        groupby (str): Feature in adata.obs to group cells by.
        metric (str): One of 'correlation' or 'cosine', the distance metric for the clustering.
        use_rep (str): Low-dimensional representation to use for clustering within adata.obsm, like 'X_pca' or 'X_spectral'.
        starting_comp (int): Index of the first component to use for clustering. Should be 0 if clustering gex or 1 if clustering atac on spectral.
        **cg_kwargs: Additional keyword arguments for sns.clustermap.
    Returns:
        cg (sns.clustermap): Clustermap object.
    """
    
    df = pd.DataFrame(adata.obsm[use_rep], index = adata.obs.index).iloc[:,starting_comp:].join(adata.obs[groupby]).groupby(groupby).mean()
    if drop_ntc:
        df = df.query(f"{groupby} != 'NTC'")

    if metric == 'correlation':
        distance = 1 - df.T.corr()
        condensed = squareform(distance)
    elif metric == 'cosine':
        condensed = pdist(df, metric='cosine')
        distance = pd.DataFrame(squareform(condensed), index=df.index, columns=df.index)
    else:
        raise NotImplementedError
    
    distance[distance < 1e-15] = 0
    link = hc.linkage(condensed, method = 'ward', optimal_ordering=True)
    cg = sns.clustermap(pd.DataFrame(cosine_similarity(df) if metric == 'cosine' else df.T.corr(), index = df.index), cmap = 'icefire', center = 0, row_linkage = link, col_linkage = link,  dendrogram_ratio=0.1, **cg_kwargs)
    cg.ax_heatmap.set_xticks([])
    cg.ax_row_dendrogram.set_visible(False)

    return cg

def show(df, max_rows = None):
    
    """
    Convenience function for displaying a pandas DataFrame in a Jupyter notebook without truncation.

    Args:
        df (pd.DataFrame): DataFrame to display.
        max_rows (int): Maximum number of rows to display, if None displays all rows.
    """

    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', None, 'display.width', None):
        display(df)


def plot_umap(df, hue = 'joint_id', show = True, figsize = (10,8), **kwargs):

    """
    Convenience function for plotting a nice-looking UMAP once you have the UMAP representation df.

    Args:
        df (pd.DataFrame): UMAP representation containing columns 0 and 1 for the UMAP coordinates, and optionally another column with groups for coloring.
        hue (str): Name of a column in df containing groups to color points by.
        show (bool): Whether to display the plot.
        figsize (tuple): Figure size.
        **kwargs: Other args for sns.scatterplot, like alpha, palette, etc.
    """

    plt.figure(figsize = figsize)
    with sns.axes_style("white"):
        ax = sns.scatterplot(data = df, x = 0, y = 1, edgecolor = 'none', alpha = 0.8, hue = hue, **kwargs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        if show:
            plt.show()

def plot_expr_violin(pop, gene, groupby = 'guide_identity', figsize = (10,4), select_groups = None, ylim = None, strip = True, **strip_kwargs):

    """
    Make a violin plot of single-cell expression for a given gene.

    Args:
        pop (AnnData): AnnData gex object.
        gene (str): Feature in pop to plot, should match a name in pop.var_names.
        groupby (str): Feature in pop.obs to group cells by, e.g., guide_identity.
        figsize (tuple): Figure size.
        select_groups (list): List of groups to plot, if None plots all groups.
        ylim (tuple): y-axis limits.
        strip (bool): Whether to overlay individual cells as points with sns.stripplot().
        **strip_kwargs: Additional keyword arguments for sns.stripplot.
    """
    
    if isinstance(pop, CellPopulation):
        df = pop.normalized_matrix.join(pop.cells[groupby])
    else:
        df = pop.to_df().join(pop.obs[groupby])

    if select_groups is not None:
        df = df.query(f"{groupby} in @select_groups")

    df[groupby] = df[groupby].astype(str)
    plt.figure(figsize=figsize)
    sns.violinplot(data = df, x = groupby, y = gene, color = 'lightgrey', cut = 0, inner_kws = dict(box_width=6, whis_width=2, zorder = 2))
    if strip:
        sns.stripplot(data = df, x = 'guide_identity', y = 'PDE1C', color = '#0390fc', alpha = 0.5, zorder = 1, **strip_kwargs)
    plt.xticks(rotation = 90)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.show()

def plot_dap_distribution(daps, groupby = 'guide_target', highlight_guides = None, color_mapping = None, percentiles = (25,75), savefig = False, figsize = (8,6)):

    """
    Make a dotplot of differential accessibility across an experiment, useful for quickly seeing which perturbations generally open vs. close chromatin. 
    
    Args:
        daps (pd.DataFrame): DataFrame of differential peaks, like expt.daps.
        groupby (str): Feature to group cells by, should be a column in daps.
        figsize (tuple): Figure size.
        highlight_guides (list): List of groups to plot, if None plots all groups.
        color_mapping (dict): Dictionary mapping groupby values to hex colors.
        percentiles (tuple): Percentiles to plot confidence interval.
        savefig (str): Path to save figure, if False or None displays figure only.
    """

    if 'n_cells' not in daps.columns:
        daps['n_cells'] = 20

    df_l2fcs = daps.groupby(groupby).l2fc.median().to_frame()
    df_l2fcs['ci_lower'] = daps.groupby(groupby).l2fc.apply(lambda x: np.percentile(x, percentiles[0]))
    df_l2fcs['ci_upper'] = daps.groupby("guide_target").l2fc.apply(lambda x: np.percentile(x, percentiles[1]))
    df_l2fcs = df_l2fcs.sort_values("l2fc", ascending=False).join(daps[[groupby, 'n_cells']].drop_duplicates().set_index(groupby)).query("n_cells >= 20")

    daps_plot = daps.query("n_cells >= 20").copy()
    daps_plot['in_ci'] = daps_plot.progress_apply(lambda df: df.l2fc < df_l2fcs.loc[df.guide_target, 'ci_upper'] and df.l2fc > df_l2fcs.loc[df.guide_target, 'ci_lower'], axis = 1)
    daps_plot = daps_plot.query("in_ci")

    if color_mapping is None:
        palette = sns.color_palette("husl", len(daps[groupby].unique()))
        color_mapping = dict(zip(pd.Series(daps[groupby].unique()).sort_values(), palette))

    if highlight_guides is not None:
        daps_plot['highlight'] = daps_plot.guide_target.map(lambda i: i.split('_')[0] in highlight_guides)
        df_l2fcs['highlight'] = df_l2fcs.index.map(lambda i: i.split('_')[0] in highlight_guides)
        daps_plot = daps_plot.query("highlight")
        df_l2fcs = df_l2fcs.query("highlight")
    
    sorted_groups = daps_plot.groupby(groupby)['l2fc'].median().sort_values().index
    daps_plot[groupby] = pd.Categorical(daps_plot[groupby], categories=sorted_groups, ordered=True)
    df_l2fcs[groupby] = pd.Categorical(df_l2fcs.index, categories=sorted_groups, ordered=True)

    plt.figure(figsize=figsize)
    sns.scatterplot(data = daps_plot, x = 'l2fc', y = groupby, color = '#dedede', alpha = 0.25, edgecolor = '#ffffff')
    sns.scatterplot(data = df_l2fcs, x = 'l2fc', y = groupby, hue = 'guide_target', palette = color_mapping, edgecolor = None, legend = None, s = 120)
    plt.axvline(x = 0, linestyle = '--', color = 'grey')
    plt.yticks([])
    for i, row in df_l2fcs.iterrows():
        plt.text(row['l2fc'] + 0.2, row.name, str(row.name.split("_")[0]), fontsize=8, ha='left', va='center', color=color_mapping[row[groupby]])
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylabel('')
    plt.xlabel('log2 fold change in differential peak accessibility vs. NTC')
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi = 500)
    plt.show()

class GexExperiment(MultiomeExperiment):

    def __init__(self, experiment_dir, id_dir, intermediate_file_dir = None):

        self.experiment_dir = experiment_dir
        self.id_dir = id_dir
        self.intermediate_file_dir = intermediate_file_dir if intermediate_file_dir is not None else os.path.join(self.experiment_dir, "analysis", "intermediate_files")
        self.sample_names = []
        for file in os.scandir(experiment_dir):
            if "Lane" in file.name:
                self.sample_names.append(file.name)
        print("Found samples: ", self.sample_names, flush = True)

        self.gex_qc = {}
        self.singlets_assigned = {}
        self.cellpops = {}

    def combined_qc(self, sample_name, umi = 6, complexity = 6, fragment = 1000, tsse = 4, verbose = True,  plot_qc = True):

        """
        QC filtering of cells by assignment and GEX and ATAC metrics.

        Args:
            sample_name (str): Lane to filter
            umi (float): Minimum UMI count for GEX (log1p).
            complexity (float): Minimum unique genes count for GEX (log1p).
            fragment (float): Minimum unique fragments for ATAC.
            tsse (float): Minimum TSS enrichment for ATAC.
            verbose (bool): print summary statistics.
            plot_qc (bool): plot QC metrics after calculation.
        Returns:
            singlets_assigned (pd.DataFrame): Cells passing QC with assignment info.
            qc (pd.DataFrame): All cells with GEX and ATAC QC metrics.
        """
        
        qc = self.cut_qc_metrics_by_called(sample_name, mode = 'gex', plot = plot_qc)
        singlets = qc.query(f"n_id == 1 and log1p_total_counts > {umi} and log1p_n_genes_by_counts > {complexity}")
        
        if verbose:
            print(f"{sample_name}:", flush = True)
            print(f"{len(singlets)} ID singlets passing QC", flush = True)
            print(f"{np.expm1(singlets.log1p_total_counts).mean():.2f} GEX UMI", flush = True)
            print(f"{np.expm1(singlets.log1p_n_genes_by_counts).mean():.2f} unique genes", flush = True)

        id_files = [file for file in os.scandir(self.id_dir) if "called_ids" in file.name]
        called_ids = pd.read_csv([f for f in id_files if sample_name in f.name][0])
        called_ids['joint_id'] = called_ids.apply(lambda df: df['identity'].split('_')[0] + "_" + df['LB_identity'].split('_')[0] if df['LB_identity'] != "_" else df['identity'].split("_")[0], axis=1)
        called_ids['joint_id_ntc'] = called_ids.joint_id.map(lambda i: "NTC" if i.startswith("NTC") else i)
        called_ids['guide_target'] = called_ids.joint_id_ntc.map(lambda i: i.split("_")[0])
        called_ids = called_ids[['CB', 'guide_target', 'identity', 'LB_identity', 'joint_id', 'joint_id_ntc', 'UMI']].set_index("CB")
        called_ids.index = called_ids.index.map(lambda cb: cb + '-' + str(sample_name.split('Lane')[1].split('_')[0]))
        singlets_assigned = singlets.join(called_ids, how = 'left')
        assert len(singlets) == len(singlets_assigned)
        
        self.singlets_assigned[sample_name] = singlets_assigned
        return singlets_assigned, qc