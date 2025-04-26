#!/bin/bash
#SBATCH --job-name=ESM_MLM
#SBATCH --output=logs/version_%j/slurm_out/%j.out	     # Redirect standard out to slurm_outs
#SBATCH --error=logs/version_%j/slurm_out/%j.err	     # Redirect standard err to slurm_outs
#SBATCH --partition=gpu                                  # GPU partition
#SBATCH --account=w25_artimis_g                          # Chicoma account
#SBATCH --qos=standard                                   # Standard QOS
#SBATCH --time=4:00:00                                  # 16 hour max limit
#SBATCH --nodes=2                                        # Number of nodes
#SBATCH --ntasks-per-node=4                              # Number of processes per node (match GPU count)
#SBATCH --exclude=nid001448                              # Exclude node
#SBATCH --exclusive                                      # Use entire node exclusively

# Load environment
module load cudatoolkit
source ../../../../venvs/spike/bin/activate

# Run
srun python lightning-esm.py

