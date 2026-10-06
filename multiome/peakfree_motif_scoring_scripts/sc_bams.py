import subprocess
import os
from tqdm import tqdm
import sys
sys.path.append("/data1/normantm/eli/software/")
import multiome as mo

expt = mo.MultiomeExperiment.load("/data1/normantm/eli/202412_expt/250102_multiomeTFs_guide_tgt_peaks.pkl")
bams = [b for b in os.scandir('/data1/normantm/rasmusa1/multiome_TFs/temp') if b.name.endswith(".bam") and not "PRDM16" in b.name and not "NTC" in b.name]

for b in tqdm(bams):
    target = b.name.split(".")[0]
    if target in expt.all_singlets_assigned["guide_target"].unique():
        print(f"Processing {target}...", flush = True)
        if not os.path.exists(f"/data1/normantm/eli/scratch/sc_bws/{target}"):
            os.makedirs(f"/data1/normantm/eli/scratch/sc_bws/{target}", exist_ok = True)
            cmd = f"samtools split -d CB -@ 24 -M -1 -f '/data1/normantm/eli/scratch/sc_bws/{target}/%!_.%.' {b.path}"
            subprocess.run(cmd, shell = True, check = True)
            bcs = expt.all_singlets_assigned.query(f"guide_target == '{target}'").index
            for file in os.scandir(f"/data1/normantm/eli/scratch/sc_bws/{target}"):
                bc = file.name.split("_")[0]
                if not bc in bcs:
                    os.remove(file.path)
        else:
            continue