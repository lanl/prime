#!/bin/bash
#SBATCH --job-name=DEBUG
#SBATCH --output=logs/version_%j/slurm_out/%j.out	 # Redirect standard out to slurm_outs
#SBATCH --error=logs/version_%j/slurm_out/%j.err	 # Redirect standard err to slurm_outs
#SBATCH --partition=gpu                                          # GPU partition
#SBATCH --reservation=gpu_debug                                  # Debug
#SBATCH --time=2:00:00                                           # 2 hour max limit
#SBATCH --nodes=1                                                # Number of nodes
#SBATCH --ntasks-per-node=1                                      # Number of processes per node (match GPU count)
#SBATCH --exclude=nid001432                                      # Exclude node
#SBATCH --exclusive                                              # Use entire node exclusively

# Load environment
module load cudatoolkit
source /lustre/scratch4/turquoise/exempt/artimis/biosecurity/venvs/spike/bin/activate

# Run
echo "Running ESM-RBD model (non finetuned)"
python supp_fig-mlm_dms_comp.py \
--model_version esm
echo "====================================="

# Run
echo "Running ESM-RBD model (finetuned)"
python supp_fig-mlm_dms_comp.py \
--model_version esm \
--use_finetuned
echo "====================================="

# Run
echo "Running BERT-RBD model (non finetuned)"
python supp_fig-mlm_dms_comp.py \
--model_version bert
echo "====================================="

# Run
echo "Running BERT-RBD model (finetuned)"
python supp_fig-mlm_dms_comp.py \
--model_version bert \
--use_finetuned
echo "====================================="