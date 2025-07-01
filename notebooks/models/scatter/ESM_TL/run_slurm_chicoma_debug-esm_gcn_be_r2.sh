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
python fig3-esm_gcn_be-r2_values.py \
--script_letter A \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn_be/version_21774753/ckpt/best_model-epoch=435.val_be_rmse=1.5045.val_binding_rmse=1.2355.val_expression_rmse=0.8584.ckpt

# Run
echo "Running script B"
python fig3-esm_gcn_be-r2_values.py \
--script_letter B \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn_be/version_21774754/ckpt/best_model-epoch=998.val_be_rmse=1.5405.val_binding_rmse=1.2908.val_expression_rmse=0.8407.ckpt

# Run
echo "Running script C"
python fig3-esm_gcn_be-r2_values.py \
--script_letter C \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn_be/version_21774755/ckpt/best_model-epoch=863.val_be_rmse=1.1654.val_binding_rmse=0.8605.val_expression_rmse=0.7859.ckpt

# Run
echo "Running script D"
python fig3-esm_gcn_be-r2_values.py \
--script_letter D \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn_be/version_21774757/ckpt/best_model-epoch=192.val_be_rmse=1.1871.val_binding_rmse=0.8927.val_expression_rmse=0.7824.ckpt

# Run
echo "Running script E"
python fig3-esm_gcn_be-r2_values.py \
--script_letter E \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn_be/version_21787566/ckpt/best_model-epoch=865.val_be_rmse=1.5299.val_binding_rmse=1.2291.val_expression_rmse=0.9110.ckpt

# Run
echo "Running script F"
python fig3-esm_gcn_be-r2_values.py \
--script_letter F \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn_be/version_21831725/ckpt/best_model-epoch=997.val_be_rmse=1.5967.val_binding_rmse=1.3446.val_expression_rmse=0.8612.ckpt

# Run
echo "Running script G"
python fig3-esm_gcn_be-r2_values.py \
--script_letter G \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn_be/version_21831726/ckpt/best_model-epoch=901.val_be_rmse=1.1647.val_binding_rmse=0.8628.val_expression_rmse=0.7824.ckpt

# Run
echo "Running script H"
python fig3-esm_gcn_be-r2_values.py \
--script_letter H \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn_be/version_21831729/ckpt/best_model-epoch=204.val_be_rmse=1.1927.val_binding_rmse=0.8931.val_expression_rmse=0.7905.ckpt