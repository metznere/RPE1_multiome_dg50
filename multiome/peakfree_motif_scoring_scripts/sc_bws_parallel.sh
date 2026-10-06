#!/bin/bash
#SBATCH --job-name=sc_bws_array      # Job name
#SBATCH --cpus-per-task=32           # Number of CPUs per task
#SBATCH --mem-per-cpu=24G            # Memory per CPU
#SBATCH --time=1-00:00:00            # Time limit (1 day)
#SBATCH --output=logs/job_%A_%a.out  # Output log (%A = job ID, %a = array index)
#SBATCH --error=logs/job_%A_%a.err   # Error log
#SBATCH --array=1-10                 # Job array range (10 jobs)

# Define the commands as an array
commands=(
  "python /data1/normantm/eli/202412_expt/sc_bws.py --target HOXB6 RUNX3 BCL11A ZFX OLIG2 TLX1 RXRG FOXQ1 NR2E1 GLI1"
  "python /data1/normantm/eli/202412_expt/sc_bws.py --target ONECUT1 VSX1 DLX3 NR4A3 ATOH7 ONECUT3 ZNF813 TEAD3 HIF1A EGR2"
  "python /data1/normantm/eli/202412_expt/sc_bws.py --target ONECUT2 DLX2 HMX3 TGIF1 POU4F1 CIITA GATA2 FOXC1 FOS NKX2-5"
  "python /data1/normantm/eli/202412_expt/sc_bws.py --target FOSB NEUROD1 LBX1 TOX OSR1 SIX3 KLF4 FOXL2 PRDM1 CRX"
  "python /data1/normantm/eli/202412_expt/sc_bws.py --target ARID1A LHX4 OSR2 KLF16 SOX9 FOXN4 PAX6 POU4F2 SOX13 EPAS1"
  "python /data1/normantm/eli/202412_expt/sc_bws.py --target GLIS3 NEUROG2 NR5A2 MEIS2 HES1 CREBBP ZNF827 JUN SOX18 FOXC2"
  "python /data1/normantm/eli/202412_expt/sc_bws.py --target NKX3-1 FOXA1 PROP1 HES6 FOXF1 JDP2 KLF15 SOX11 NFATC1 HELZ2"
  "python /data1/normantm/eli/202412_expt/sc_bws.py --target MYC OTX2 PPARG PROX1 HOXC8 SIM2 MECOM PPARGC1A PAX2 TP73"
  "python /data1/normantm/eli/202412_expt/sc_bws.py --target ARID3A TFAP2A OVOL2 TBX21 MYRF HES5 LHX2 ELF1 FOXA3 BNC2"
  "python /data1/normantm/eli/202412_expt/sc_bws.py --target FOXF2 POU3F1 ZEB2 SOX2"
)

# Get the command corresponding to the array index
cmd="${commands[$SLURM_ARRAY_TASK_ID - 1]}"  # Slurm arrays start at 1

# Load environment
source ~/.bashrc       # Ensure Mamba is available
mamba activate seq2gex  # Activate the seq2gex environment

# Run the command
$cmd