import subprocess
import sqlite3
import pandas as pd
from joblib import Parallel, delayed
import sys


chromsizes = pd.read_csv("/data1/normantm/eli/seq2gex/data/genome/chromsizes.bed", sep="\t", header=None)
chromsizes.columns = ['chr', 'start', 'end']
chromsizes = chromsizes.set_index('chr')

def insert_chrom_motifs(chrom, chunksize = 50000):

    conn = sqlite3.connect(f'/data1/normantm/eli/scratch/chr_motifs/{chrom}_motifs.db')
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS motifs (
                chr TEXT,
                start INTEGER,
                end INTEGER,
                id TEXT,
                score INTEGER,
                name TEXT
            )
    """)
    conn.commit()
    
    for start in range(0, chromsizes.loc[chrom, 'end'], chunksize):
        
        end = min(start + chunksize, chromsizes.loc[chrom, 'end'])
        cmd = f"bigBedToBed http://hgdownload.soe.ucsc.edu/gbdb/hg38/jaspar/JASPAR2024.bb -chrom={chrom} -start={start} -end={end} stdout"
        out = subprocess.run(cmd, shell=True, capture_output=True)
        
        entries = out.stdout.decode('utf-8').strip().split('\n')
        data = []
        
        for line in entries:
            fields = line.split('\t')
            if len(fields) < 7:
                continue
            data.append((chrom, int(fields[1]), int(fields[2]), fields[3], int(fields[4]), fields[6]))
        
        cursor.executemany(
            "INSERT INTO motifs (chr, start, end, id, score, name) VALUES (?, ?, ?, ?, ?, ?)", data
        )
        
        conn.commit()
    
    conn.close()

Parallel(n_jobs = 23)(delayed(insert_chrom_motifs)(chrom) for chrom in chromsizes.index)