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
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter A \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046353/ckpt/best_model-epoch=949.val_rmse=1.0786.val_mlm_accuracy=6.6747.ckpt

# Run
echo "Running script A expression"
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter A \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046354/ckpt/best_model-epoch=946.val_rmse=0.6161.val_mlm_accuracy=6.5574.ckpt

# Run
echo "Running script B binding"
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter B \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046355/ckpt/best_model-epoch=982.val_rmse=1.5060.val_mlm_accuracy=6.5486.ckpt

# Run
echo "Running script B expression"
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter B \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046356/ckpt/best_model-epoch=982.val_rmse=0.8141.val_mlm_accuracy=6.5486.ckpt

# Run
echo "Running script C binding"
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter C \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046357/ckpt/best_model-epoch=101.val_rmse=1.0409.val_mlm_accuracy=98.4291.ckpt

# Run
echo "Running script C expression"
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter C \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046358/ckpt/best_model-epoch=126.val_rmse=0.5767.val_mlm_accuracy=98.4378.ckpt

# Run
echo "Running script D binding"
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter D \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046359/ckpt/best_model-epoch=768.val_rmse=1.0156.val_mlm_accuracy=98.4441.ckpt

# Run
echo "Running script D expression"
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter D \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046360/ckpt/best_model-epoch=716.val_rmse=0.5696.val_mlm_accuracy=98.3989.ckpt

# Run
echo "Running script E binding"
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter E \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046361/ckpt/best_model-epoch=832.val_rmse=1.0397.val_mlm_accuracy=96.4367.ckpt

# Run
echo "Running script E expression"
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter E \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046362/ckpt/best_model-epoch=992.val_rmse=0.5788.val_mlm_accuracy=96.5487.ckpt

# Run
echo "Running script F binding"
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter F \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046363/ckpt/best_model-epoch=963.val_rmse=1.2520.val_mlm_accuracy=96.5333.ckpt

# Run
echo "Running script F expression"
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter F \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046364/ckpt/best_model-epoch=944.val_rmse=0.7033.val_mlm_accuracy=96.5634.ckpt

# Run
echo "Running script G binding"
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter G \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046365/ckpt/best_model-epoch=92.val_rmse=1.0491.val_mlm_accuracy=98.4069.ckpt

# Run
echo "Running script G expression"
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter G \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046366/ckpt/best_model-epoch=129.val_rmse=0.5765.val_mlm_accuracy=98.4204.ckpt

# Run
echo "Running script H binding"
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter H \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046367/ckpt/best_model-epoch=832.val_rmse=1.0181.val_mlm_accuracy=98.4384.ckpt

# Run
echo "Running script H expression"
python fig3-esm_mlm_blstm-r2_values.py \
--script_letter H \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_blstm/version_22046368/ckpt/best_model-epoch=946.val_rmse=0.5673.val_mlm_accuracy=98.4149.ckpt