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
python fig3-esm_fcn-r2_values.py \
--script_letter A \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21837261/ckpt/best_model-epoch=883.val_rmse=1.2680.ckpt

# Run
echo "Running script A expression"
python fig3-esm_fcn-r2_values.py \
--script_letter A \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21837262/ckpt/best_model-epoch=968.val_rmse=0.6283.ckpt

# Run
echo "Running script B binding"
python fig3-esm_fcn-r2_values.py \
--script_letter B \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21837264/ckpt/best_model-epoch=996.val_rmse=1.3841.ckpt

# Run
echo "Running script B expression"
python fig3-esm_fcn-r2_values.py \
--script_letter B \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21837265/ckpt/best_model-epoch=992.val_rmse=0.6986.ckpt

# Run
echo "Running script C binding"
python fig3-esm_fcn-r2_values.py \
--script_letter C \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21774809/ckpt/best_model-epoch=236.val_rmse=0.5097.ckpt

# Run
echo "Running script C expression"
python fig3-esm_fcn-r2_values.py \
--script_letter C \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21774811/ckpt/best_model-epoch=920.val_rmse=0.3277.ckpt

# Run
echo "Running script D binding"
python fig3-esm_fcn-r2_values.py \
--script_letter D \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21774812/ckpt/best_model-epoch=988.val_rmse=0.5644.ckpt

# Run
echo "Running script D expression"
python fig3-esm_fcn-r2_values.py \
--script_letter D \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21774813/ckpt/best_model-epoch=144.val_rmse=0.3293.ckpt

# Run
echo "Running script E binding"
python fig3-esm_fcn-r2_values.py \
--script_letter E \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21837266/ckpt/best_model-epoch=781.val_rmse=1.3500.ckpt

# Run
echo "Running script E expression"
python fig3-esm_fcn-r2_values.py \
--script_letter E \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21837267/ckpt/best_model-epoch=956.val_rmse=0.7070.ckpt

# Run
echo "Running script F binding"
python fig3-esm_fcn-r2_values.py \
--script_letter F \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21837268/ckpt/best_model-epoch=983.val_rmse=1.4483.ckpt

# Run
echo "Running script F expression"
python fig3-esm_fcn-r2_values.py \
--script_letter F \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21837269/ckpt/best_model-epoch=986.val_rmse=0.7621.ckpt

# Run
echo "Running script G binding"
python fig3-esm_fcn-r2_values.py \
--script_letter G \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21837270/ckpt/best_model-epoch=288.val_rmse=0.5081.ckpt

# Run
echo "Running script G expression"
python fig3-esm_fcn-r2_values.py \
--script_letter G \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21837271/ckpt/best_model-epoch=786.val_rmse=0.3218.ckpt

# Run
echo "Running script H binding"
python fig3-esm_fcn-r2_values.py \
--script_letter H \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21837272/ckpt/best_model-epoch=124.val_rmse=0.5370.ckpt

# Run
echo "Running script H expression"
python fig3-esm_fcn-r2_values.py \
--script_letter H \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_fcn/version_21837273/ckpt/best_model-epoch=155.val_rmse=0.3163.ckpt