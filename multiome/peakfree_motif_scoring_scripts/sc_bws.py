import subprocess
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--target", type = str, nargs = "+")
args = parser.parse_args()

targets = [f.name for f in os.scandir("/data1/normantm/eli/scratch/sc_bws") if f.is_dir()] if not args.target else list(args.target)
for tgt in tqdm(targets):
    bcs = [f.path for f in os.scandir(f"/data1/normantm/eli/scratch/sc_bws/{tgt}") if f.name.endswith(".bam")]
    print(f"Processing {tgt}...", flush = True)
    for bc in tqdm(bcs):
        if not os.path.exists(bc + ".bai"):
            idx_cmd = f"samtools index {bc} -@ 32"
            subprocess.run(idx_cmd, shell = True, check = True)
            bw_name = bc.replace(".bam", ".bw")
            bw_cmd = f"bamCoverage -b {bc} -o {bw_name} --binSize 10 --Offset 4 -5 --normalizeUsing CPM  --extendReads 0 --ignoreDuplicates --minMappingQuality 30 --centerReads -p 24"
            subprocess.run(bw_cmd, shell = True, check = True, capture_output=True)
    print(f"Done with {tgt}", flush = True)