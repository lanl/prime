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
echo "Running script A"
python fig3-esm_mlm_blstm_be-r2_values.py \
--script_letter A \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm_be/version_22049885/ckpt/best_model-epoch=976.val_be_rmse=1.2556.val_binding_rmse=1.0826.val_expression_rmse=0.6360.val_mlm_accuracy=6.4038.ckpt

# Run
echo "Running script B"
python fig3-esm_mlm_blstm_be-r2_values.py \
--script_letter B \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm_be/version_22049886/ckpt/best_model-epoch=982.val_be_rmse=1.7264.val_binding_rmse=1.5023.val_expression_rmse=0.8507.val_mlm_accuracy=6.5486.ckpt

# Run
echo "Running script C"
python fig3-esm_mlm_blstm_be-r2_values.py \
--script_letter C \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm_be/version_22049887/ckpt/best_model-epoch=128.val_be_rmse=1.1948.val_binding_rmse=1.0391.val_expression_rmse=0.5899.val_mlm_accuracy=98.4479.ckpt

# Run
echo "Running script D"
python fig3-esm_mlm_blstm_be-r2_values.py \
--script_letter D \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm_be/version_22049888/ckpt/best_model-epoch=832.val_be_rmse=1.1668.val_binding_rmse=1.0120.val_expression_rmse=0.5807.val_mlm_accuracy=98.4382.ckpt

# Run
echo "Running script E"
python fig3-esm_mlm_blstm_be-r2_values.py \
--script_letter E \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm_be/version_22047976/ckpt/best_model-epoch=997.val_be_rmse=1.2001.val_binding_rmse=1.0395.val_expression_rmse=0.5997.val_mlm_accuracy=96.5463.ckpt

# Run
echo "Running script F"
python fig3-esm_mlm_blstm_be-r2_values.py \
--script_letter F \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm_be/version_22047977/ckpt/best_model-epoch=990.val_be_rmse=1.4415.val_binding_rmse=1.2326.val_expression_rmse=0.7473.val_mlm_accuracy=96.4694.ckpt

# Run
echo "Running script G"
python fig3-esm_mlm_blstm_be-r2_values.py \
--script_letter G \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm_be/version_22047978/ckpt/best_model-epoch=128.val_be_rmse=1.1925.val_binding_rmse=1.0403.val_expression_rmse=0.5829.val_mlm_accuracy=98.4476.ckpt

# Run
echo "Running script H"
python fig3-esm_mlm_blstm_be-r2_values.py \
--script_letter H \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm_be/version_22049889/ckpt/best_model-epoch=768.val_be_rmse=1.1623.val_binding_rmse=1.0108.val_expression_rmse=0.5737.val_mlm_accuracy=98.4437.ckpt