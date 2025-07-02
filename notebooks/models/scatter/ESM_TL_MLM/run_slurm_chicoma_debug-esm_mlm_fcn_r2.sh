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
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter A \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044640/ckpt/best_model-epoch=937.val_rmse=1.8579.val_mlm_accuracy=6.6861.ckpt

# Run
echo "Running script A expression"
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter A \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044641/ckpt/best_model-epoch=915.val_rmse=0.9715.val_mlm_accuracy=6.6005.ckpt

# Run
echo "Running script B binding"
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter B \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044643/ckpt/best_model-epoch=981.val_rmse=1.8660.val_mlm_accuracy=6.5781.ckpt

# Run
echo "Running script B expression"
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter B \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044644/ckpt/best_model-epoch=973.val_rmse=0.9752.val_mlm_accuracy=6.4766.ckpt

# Run
echo "Running script C binding"
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter C \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044659/ckpt/best_model-epoch=131.val_rmse=1.0458.val_mlm_accuracy=98.4169.ckpt

# Run
echo "Running script C expression"
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter C \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044510/ckpt/best_model-epoch=126.val_rmse=0.5746.val_mlm_accuracy=98.4378.ckpt

# Run
echo "Running script D binding"
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter D \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044511/ckpt/best_model-epoch=832.val_rmse=1.0238.val_mlm_accuracy=98.4385.ckpt

# Run
echo "Running script D expression"
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter D \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044513/ckpt/best_model-epoch=993.val_rmse=0.5705.val_mlm_accuracy=98.4113.ckpt

# Run
echo "Running script E binding"
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter E \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044514/ckpt/best_model-epoch=922.val_rmse=1.7742.val_mlm_accuracy=96.5379.ckpt

# Run
echo "Running script E expression"
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter E \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044515/ckpt/best_model-epoch=993.val_rmse=0.9384.val_mlm_accuracy=96.4938.ckpt

# Run
echo "Running script F binding"
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter F \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044516/ckpt/best_model-epoch=834.val_rmse=1.7923.val_mlm_accuracy=96.5780.ckpt

# Run
echo "Running script F expression"
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter F \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044517/ckpt/best_model-epoch=993.val_rmse=0.9448.val_mlm_accuracy=96.4938.ckpt

# Run
echo "Running script G binding"
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter G \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044519/ckpt/best_model-epoch=98.val_rmse=1.0404.val_mlm_accuracy=98.4031.ckpt

# Run
echo "Running script G expression"
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter G \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044520/ckpt/best_model-epoch=111.val_rmse=0.5756.val_mlm_accuracy=98.4205.ckpt

# Run
echo "Running script H binding"
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter H \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044521/ckpt/best_model-epoch=768.val_rmse=1.0196.val_mlm_accuracy=98.4443.ckpt

# Run
echo "Running script H expression"
python fig3-esm_mlm_fcn-r2_values.py \
--script_letter H \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn/version_22044522/ckpt/best_model-epoch=946.val_rmse=0.5684.val_mlm_accuracy=98.4146.ckpt