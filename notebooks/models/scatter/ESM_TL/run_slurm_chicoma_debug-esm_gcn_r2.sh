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
python fig3-esm_gcn-r2_values.py \
--script_letter A \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21774823/ckpt/best_model-epoch=306.val_rmse=1.1718.ckpt

# Run
echo "Running script A expression"
python fig3-esm_gcn-r2_values.py \
--script_letter A \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21774824/ckpt/best_model-epoch=499.val_rmse=0.6067.ckpt

# Run
echo "Running script B binding"
python fig3-esm_gcn-r2_values.py \
--script_letter B \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21774825/ckpt/best_model-epoch=973.val_rmse=1.1707.ckpt

# Run
echo "Running script B expression"
python fig3-esm_gcn-r2_values.py \
--script_letter B \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21774826/ckpt/best_model-epoch=998.val_rmse=0.6365.ckpt

# Run
echo "Running script C binding"
python fig3-esm_gcn-r2_values.py \
--script_letter C \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21774828/ckpt/best_model-epoch=419.val_rmse=0.5397.ckpt

# Run
echo "Running script C expression"
python fig3-esm_gcn-r2_values.py \
--script_letter C \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21774827/ckpt/best_model-epoch=798.val_rmse=0.3265.ckpt

# Run
echo "Running script D binding"
python fig3-esm_gcn-r2_values.py \
--script_letter D \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21774829/ckpt/best_model-epoch=998.val_rmse=0.5913.ckpt

# Run
echo "Running script D expression"
python fig3-esm_gcn-r2_values.py \
--script_letter D \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21774830/ckpt/best_model-epoch=182.val_rmse=0.3519.ckpt

# Run
echo "Running script E binding"
python fig3-esm_gcn-r2_values.py \
--script_letter E \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21837274/ckpt/best_model-epoch=590.val_rmse=1.1662.ckpt

# Run
echo "Running script E expression"
python fig3-esm_gcn-r2_values.py \
--script_letter E \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21837275/ckpt/best_model-epoch=814.val_rmse=0.6624.ckpt

# Run
echo "Running script F binding"
python fig3-esm_gcn-r2_values.py \
--script_letter F \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21837276/ckpt/best_model-epoch=987.val_rmse=1.2181.ckpt

# Run
echo "Running script F expression"
python fig3-esm_gcn-r2_values.py \
--script_letter F \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21837277/ckpt/best_model-epoch=998.val_rmse=0.7113.ckpt

# Run
echo "Running script G binding"
python fig3-esm_gcn-r2_values.py \
--script_letter G \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21837278/ckpt/best_model-epoch=391.val_rmse=0.5513.ckpt

# Run
echo "Running script G expression"
python fig3-esm_gcn-r2_values.py \
--script_letter G \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21837279/ckpt/best_model-epoch=833.val_rmse=0.3391.ckpt

# Run
echo "Running script H binding"
python fig3-esm_gcn-r2_values.py \
--script_letter H \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21837281/ckpt/best_model-epoch=979.val_rmse=0.6041.ckpt

# Run
echo "Running script H expression"
python fig3-esm_gcn-r2_values.py \
--script_letter H \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_gcn/version_21837282/ckpt/best_model-epoch=109.val_rmse=0.3669.ckpt