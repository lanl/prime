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
python fig3-esm_fcn_be-r2_values.py \
--script_letter A \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn_be/version_21774749/ckpt/best_model-epoch=982.val_be_rmse=1.4034.val_binding_rmse=1.2437.val_expression_rmse=0.6503.ckpt

# Run
echo "Running script B"
python fig3-esm_fcn_be-r2_values.py \
--script_letter B \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn_be/version_21831730/ckpt/best_model-epoch=996.val_be_rmse=1.5575.val_binding_rmse=1.3645.val_expression_rmse=0.7511.ckpt

# Run
echo "Running script C"
python fig3-esm_fcn_be-r2_values.py \
--script_letter C \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn_be/version_21831731/ckpt/best_model-epoch=981.val_be_rmse=0.6067.val_binding_rmse=0.5162.val_expression_rmse=0.3187.ckpt

# Run
echo "Running script D"
python fig3-esm_fcn_be-r2_values.py \
--script_letter D \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn_be/version_21831732/ckpt/best_model-epoch=186.val_be_rmse=0.6420.val_binding_rmse=0.5454.val_expression_rmse=0.3386.ckpt

# Run
echo "Running script E"
python fig3-esm_fcn_be-r2_values.py \
--script_letter E \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn_be/version_21831733/ckpt/best_model-epoch=868.val_be_rmse=1.5088.val_binding_rmse=1.3199.val_expression_rmse=0.7310.ckpt

# Run
echo "Running script F"
python fig3-esm_fcn_be-r2_values.py \
--script_letter F \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn_be/version_21831734/ckpt/best_model-epoch=999.val_be_rmse=1.6711.val_binding_rmse=1.4585.val_expression_rmse=0.8157.ckpt

# Run
echo "Running script G"
python fig3-esm_fcn_be-r2_values.py \
--script_letter G \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn_be/version_21787564/ckpt/best_model-epoch=632.val_be_rmse=0.6015.val_binding_rmse=0.5077.val_expression_rmse=0.3225.ckpt

# Run
echo "Running script H"
python fig3-esm_fcn_be-r2_values.py \
--script_letter H \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn_be/version_21787565/ckpt/best_model-epoch=224.val_be_rmse=0.6114.val_binding_rmse=0.5158.val_expression_rmse=0.3283.ckpt