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
python fig3-esm_blstm_be-r2_values.py \
--script_letter A \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm_be/version_21774745/ckpt/best_model-epoch=801.val_be_rmse=0.6693.val_binding_rmse=0.5647.val_expression_rmse=0.3593.ckpt

# Run
echo "Running script B"
python fig3-esm_blstm_be-r2_values.py \
--script_letter B \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm_be/version_21774746/ckpt/best_model-epoch=992.val_be_rmse=0.9518.val_binding_rmse=0.7958.val_expression_rmse=0.5221.ckpt

# Run
echo "Running script C"
python fig3-esm_blstm_be-r2_values.py \
--script_letter C \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm_be/version_21774747/ckpt/best_model-epoch=886.val_be_rmse=0.6215.val_binding_rmse=0.5274.val_expression_rmse=0.3288.ckpt

# Run
echo "Running script D"
python fig3-esm_blstm_be-r2_values.py \
--script_letter D \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm_be/version_21774748/ckpt/best_model-epoch=152.val_be_rmse=0.6489.val_binding_rmse=0.5516.val_expression_rmse=0.3418.ckpt

# Run
echo "Running script E"
python fig3-esm_blstm_be-r2_values.py \
--script_letter E \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm_be/version_21787558/ckpt/best_model-epoch=966.val_be_rmse=0.6141.val_binding_rmse=0.5114.val_expression_rmse=0.3400.ckpt

# Run
echo "Running script F"
python fig3-esm_blstm_be-r2_values.py \
--script_letter F \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm_be/version_21829912/ckpt/best_model-epoch=989.val_be_rmse=0.7919.val_binding_rmse=0.6464.val_expression_rmse=0.4576.ckpt

# Run
echo "Running script G"
python fig3-esm_blstm_be-r2_values.py \
--script_letter G \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm_be/version_21787560/ckpt/best_model-epoch=652.val_be_rmse=0.6119.val_binding_rmse=0.5175.val_expression_rmse=0.3265.ckpt

# Run
echo "Running script H"
python fig3-esm_blstm_be-r2_values.py \
--script_letter H \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm_be/version_21787561/ckpt/best_model-epoch=192.val_be_rmse=0.6396.val_binding_rmse=0.5441.val_expression_rmse=0.3362.ckpt