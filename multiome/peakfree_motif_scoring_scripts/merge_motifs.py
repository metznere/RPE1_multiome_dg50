import sqlite3
from tqdm import tqdm
import os

merged_conn = sqlite3.connect('/data1/normantm/eli/scratch/chr_motifs/motifs.db')
merged_cursor = merged_conn.cursor()
merged_cursor.execute("""
    CREATE TABLE IF NOT EXISTS merged (
            chr TEXT,
            start INTEGER,
            end INTEGER,
            id TEXT,
            score INTEGER,
            name TEXT
        )
""")

for file_name in tqdm(os.listdir("/data1/normantm/eli/scratch/chr_motifs/")):
    if file_name.endswith(".db"):
        try:
            conn = sqlite3.connect(f'/data1/normantm/eli/scratch/chr_motifs/{file_name}')
            cursor = conn.cursor()
            
            cursor.execute("""
                ATTACH DATABASE '/data1/normantm/eli/scratch/chr_motifs/motifs.db' AS merged
            """)
            
            cursor.execute("""
                INSERT INTO merged SELECT * FROM motifs
            """)
            
            conn.commit()
            conn.close()
        except:
            print(file_name)
            conn.close()
            continue