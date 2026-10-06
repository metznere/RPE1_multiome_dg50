#!/bin/bash
#SBATCH --job-name=sc_motif_scores_array      # Job name
#SBATCH --cpus-per-task=48           # Number of CPUs per task
#SBATCH --mem-per-cpu=16G            # Memory per CPU
#SBATCH --time=2-00:00:00            # Time limit (1 day)
#SBATCH --output=logs/job_%A_%a.out  # Output log (%A = job ID, %a = array index)
#SBATCH --error=logs/job_%A_%a.err   # Error log
#SBATCH --array=1-14                # Job array range (10 jobs)

# Define the commands as an array
commands=(
"python /data1/normantm/eli/202412_expt/sc_motif_scores.py --target all -p 48 -o /data1/normantm/eli/202412_expt/ont_sc_motif_scores_0.csv --motif Alx4 Arid3a ATOH7 BCL11A Bcl11B BNC2 CEBPA"
"python /data1/normantm/eli/202412_expt/sc_motif_scores.py --target all -p 48 -o /data1/normantm/eli/202412_expt/ont_sc_motif_scores_1.csv --motif CEBPB Crx Dlx2 Dlx3 EGR2 EGR3 ELF1"
"python /data1/normantm/eli/202412_expt/sc_motif_scores.py --target all -p 48 -o /data1/normantm/eli/202412_expt/ont_sc_motif_scores_2.csv --motif EOMES EPAS1 FOS FOXA1 FOXA3 FOXC1 FOXC2"
"python /data1/normantm/eli/202412_expt/sc_motif_scores.py --target all -p 48 -o /data1/normantm/eli/202412_expt/ont_sc_motif_scores_3.csv --motif Foxf1 FOXF2 Foxl2 Foxq1 GATA2 GATA6 Gli1"
"python /data1/normantm/eli/202412_expt/sc_motif_scores.py --target all -p 48 -o /data1/normantm/eli/202412_expt/ont_sc_motif_scores_4.csv --motif GLIS3 GSX2 HES1 HES5 HES6 HES7 Hic1"
"python /data1/normantm/eli/202412_expt/sc_motif_scores.py --target all -p 48 -o /data1/normantm/eli/202412_expt/ont_sc_motif_scores_5.csv --motif HIF1A Hmx3 HOXB6 HOXB9 HOXC8 Irf1 IRF4"
"python /data1/normantm/eli/202412_expt/sc_motif_scores.py --target all -p 48 -o /data1/normantm/eli/202412_expt/ont_sc_motif_scores_6.csv --motif JDP2 JUN KLF15 KLF16 KLF2 KLF4 KLF5"
"python /data1/normantm/eli/202412_expt/sc_motif_scores.py --target all -p 48 -o /data1/normantm/eli/202412_expt/ont_sc_motif_scores_7.csv --motif LBX1 LHX2 Lhx4 LHX6 Mecom MEIS2 MYC"
"python /data1/normantm/eli/202412_expt/sc_motif_scores.py --target all -p 48 -o /data1/normantm/eli/202412_expt/ont_sc_motif_scores_8.csv --motif NEUROD1 Neurod2 NEUROG2 Nfatc1 NKX2-5 Nkx3-1 Nr2e1"
"python /data1/normantm/eli/202412_expt/sc_motif_scores.py --target all -p 48 -o /data1/normantm/eli/202412_expt/ont_sc_motif_scores_9.csv --motif Nr5A2 OLIG2 ONECUT1 ONECUT2 ONECUT3 OSR1 OSR2"
"python /data1/normantm/eli/202412_expt/sc_motif_scores.py --target all -p 48 -o /data1/normantm/eli/202412_expt/ont_sc_motif_scores_10.csv --motif OTX2 OVOL2 PAX2 PAX6 POU3F1 POU4F1 POU4F2"
"python /data1/normantm/eli/202412_expt/sc_motif_scores.py --target all -p 48 -o /data1/normantm/eli/202412_expt/ont_sc_motif_scores_11.csv --motif PPARG PRDM1 PROP1 PROX1 RAX RUNX3 RXRG"
"python /data1/normantm/eli/202412_expt/sc_motif_scores.py --target all -p 48 -o /data1/normantm/eli/202412_expt/ont_sc_motif_scores_12.csv --motif Six3 Sox11 SOX13 SOX18 SOX2 SOX9 TBX21"
"python /data1/normantm/eli/202412_expt/sc_motif_scores.py --target all -p 48 -o /data1/normantm/eli/202412_expt/ont_sc_motif_scores_13.csv --motif TEAD3 TFAP2A TGIF1 THRB TP73 VSX1 Zfx"
)

# Get the command corresponding to the array index
cmd="${commands[$SLURM_ARRAY_TASK_ID - 1]}"  # Slurm arrays start at 1

# Load environment
source ~/.bashrc       # Ensure Mamba is available
mamba activate seq2gex  # Activate the seq2gex environment

# Run the command
$cmd