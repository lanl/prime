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
echo "Running script A binding"
python fig3-esm_blstm-r2_values.py \
--script_letter A \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21837252/ckpt/best_model-epoch=822.val_rmse=0.5685.ckpt

# Run
echo "Running script A expression"
python fig3-esm_blstm-r2_values.py \
--script_letter A \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21774790/ckpt/best_model-epoch=687.val_rmse=0.3497.ckpt

# Run
echo "Running script B binding"
python fig3-esm_blstm-r2_values.py \
--script_letter B \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21774792/ckpt/best_model-epoch=994.val_rmse=0.8058.ckpt

# Run
echo "Running script B expression"
python fig3-esm_blstm-r2_values.py \
--script_letter B \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21774791/ckpt/best_model-epoch=999.val_rmse=0.4444.ckpt

# Run
echo "Running script C binding"
python fig3-esm_blstm-r2_values.py \
--script_letter C \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21774793/ckpt/best_model-epoch=456.val_rmse=0.5341.ckpt

# Run
echo "Running script C expression"
python fig3-esm_blstm-r2_values.py \
--script_letter C \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21774794/ckpt/best_model-epoch=462.val_rmse=0.3253.ckpt

# Run
echo "Running script D binding"
python fig3-esm_blstm-r2_values.py \
--script_letter D \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21774795/ckpt/best_model-epoch=992.val_rmse=0.5626.ckpt

# Run
echo "Running script D expression"
python fig3-esm_blstm-r2_values.py \
--script_letter D \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21774796/ckpt/best_model-epoch=131.val_rmse=0.3238.ckpt

# Run
echo "Running script E binding"
python fig3-esm_blstm-r2_values.py \
--script_letter E \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21837253/ckpt/best_model-epoch=857.val_rmse=0.5224.ckpt

# Run
echo "Running script E expression"
python fig3-esm_blstm-r2_values.py \
--script_letter E \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21837254/ckpt/best_model-epoch=897.val_rmse=0.3135.ckpt

# Run
echo "Running script F binding"
python fig3-esm_blstm-r2_values.py \
--script_letter F \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21837255/ckpt/best_model-epoch=989.val_rmse=0.6631.ckpt

# Run
echo "Running script F expression"
python fig3-esm_blstm-r2_values.py \
--script_letter F \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21837256/ckpt/best_model-epoch=992.val_rmse=0.4131.ckpt

# Run
echo "Running script G binding"
python fig3-esm_blstm-r2_values.py \
--script_letter G \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21837257/ckpt/best_model-epoch=301.val_rmse=0.5281.ckpt

# Run
echo "Running script G expression"
python fig3-esm_blstm-r2_values.py \
--script_letter G \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21837258/ckpt/best_model-epoch=852.val_rmse=0.3181.ckpt

# Run
echo "Running script H binding"
python fig3-esm_blstm-r2_values.py \
--script_letter H \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21837259/ckpt/best_model-epoch=974.val_rmse=0.5531.ckpt

# Run
echo "Running script H expression"
python fig3-esm_blstm-r2_values.py \
--script_letter H \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_blstm/version_21837260/ckpt/best_model-epoch=150.val_rmse=0.3337.ckpt