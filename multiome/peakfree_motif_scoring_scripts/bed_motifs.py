import sqlite3
import os
from joblib import Parallel, delayed

DB_PATH = "/data1/normantm/eli/scratch/chr_motifs/motifs.db"
OUTPUT_DIR = "/data1/normantm/eli/scratch/motif_beds"
N_JOBS = 48

def create_index():
    """Create an index on the 'name' column for faster lookups."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON merged(name);")
        conn.commit()

def get_unique_names():
    """Retrieve unique motif names."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT name FROM merged;")
        return [row[0] for row in cursor.fetchall()]

def process_name(name):
    """Query motifs with score >= 300 and save as a BED file."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT chr, start, end, name FROM merged WHERE name = ? AND score >= 300;", (name,))
        results = cursor.fetchall()

    if results:
        bed_file = os.path.join(OUTPUT_DIR, f"{name}.bed")
        with open(bed_file, "w") as f:
            for idx, row in enumerate(results, 1):
                modified_name = f"{row[3]}_{idx}"
                f.write("\t".join([str(row[0]), str(row[1]), str(row[2]), modified_name]) + "\n")

def main():
    # create_index()
    unique_names = get_unique_names()
    Parallel(n_jobs=N_JOBS)(delayed(process_name)(name) for name in unique_names)

if __name__ == "__main__":
    main()
