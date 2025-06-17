#!/bin/bash
#SBATCH --job-name=DEBUG
#SBATCH --output=logs/esm_mlm_fcn_be/version_%j/slurm_out/%j.out	 # Redirect standard out to slurm_outs
#SBATCH --error=logs/esm_mlm_fcn_be/version_%j/slurm_out/%j.err	     # Redirect standard err to slurm_outs
#SBATCH --partition gpu                                          # GPU partition
#SBATCH --reservation=gpu_debug                                  # Debug
#SBATCH --time=2:00:00                                           # 2 hour max limit
#SBATCH --nodes=2                                                # Number of nodes
#SBATCH --ntasks-per-node=4                                      # Number of processes per node (match GPU count)
#SBATCH --exclude=nid001432                                      # Exclude node
#SBATCH --exclusive                                              # Use entire node exclusively

# Load environment
module load cudatoolkit
source /lustre/scratch4/turquoise/exempt/artimis/biosecurity/venvs/spike/bin/activate

# Run
srun python lightning-esm_mlm_fcn_be.py \
--num_epochs 20 \
--lr 1e-4 \
--from_esm_mlm ../ESM_MLM/logs/version_21768307/ckpt/best_model-epoch=73.val_loss=0.0022.val_accuracy=99.6612.ckpt

