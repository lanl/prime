#!/bin/bash
#SBATCH --job-name=ESM_MLM
#SBATCH --output=logs/version_%j/slurm_out/%j.out	     # Redirect standard out to slurm_outs
#SBATCH --error=logs/version_%j/slurm_out/%j.err	     # Redirect standard err to slurm_outs
#SBATCH --partition=gpu                                          # GPU partition
#SBATCH --account=w25_artimis_g                                  # Chicoma account
#SBATCH --qos=standard                                           # 16 hour limit
#SBATCH --nodes=1                                                # Number of nodes
#SBATCH --ntasks-per-node=1                                      # Number of processes per node (match GPU count)
#SBATCH --exclusive                                              # Use entire node exclusively

# Load environment
module load cudatoolkit
source ../../../../venvs/spike/bin/activate

# Run
srun python lightning-esm.py

