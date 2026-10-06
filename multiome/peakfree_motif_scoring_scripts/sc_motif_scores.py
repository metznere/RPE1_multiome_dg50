import os
import numpy as np
import pandas as pd
import subprocess
import argparse
from joblib import Parallel, delayed
from tqdm import tqdm
import sys

def get_motif_coverage(bcs, tgt, motif):
    
    covr = []
    for bc in bcs:
        cmd = f"bigWigAverageOverBed {bc} /data1/normantm/eli/scratch/motif_beds/{motif}.bed /dev/stdout"
        out = subprocess.run(cmd, shell = True, capture_output = True)
        covr.append(np.mean([float(result.split('\t')[-1]) for result in out.stdout.decode('utf-8').split('\n')[1:-1]]))
    
    df = pd.DataFrame(covr, index = [i.split('/')[-1].split("_")[0] for i in bcs])
    return df

def main(args):

    targets = args.target if 'all' not in args.target else [f.name for f in os.scandir("/data1/normantm/eli/scratch/sc_bws") if f.is_dir()]
    assert all([os.path.exists(f"/data1/normantm/eli/scratch/motif_beds/{motif}.bed") for motif in args.motif])
    
    tgt_dfs = []
    for tgt in targets:
        
        print(f"Processing {tgt}...", flush = True)
        dfs = []
        for motif in tqdm(args.motif):
            
            bcs = [f.path for f in os.scandir(f"/data1/normantm/eli/scratch/sc_bws/{tgt}") if f.name.endswith(".bw")]
            bc_lists = [i.tolist() for i in np.array_split(bcs, args.p)]
            results = Parallel(n_jobs = args.p)(delayed(get_motif_coverage)(bc, tgt, motif) for bc in bc_lists)
            df_results = pd.concat(results, axis = 0)
            dfs.append(df_results)
        
        df_results = pd.concat(dfs, axis = 1)
        df_results.columns = args.motif
        df_results['guide_target'] = tgt
        tgt_dfs.append(df_results)

    df_final = pd.concat(tgt_dfs, axis = 0)
    df_final.to_csv(args.o)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--motif", type=str, required = True, nargs = "+")
    parser.add_argument("--target", type = str, nargs = "+", required = True)
    parser.add_argument("-p", type = int, default = 24, help = "Number of threads to use for multiprocessing")
    parser.add_argument("-o", type = str, default = "motif_scores.csv", help = "Output file path")
    args = parser.parse_args()
    main(args)