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
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter A \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22046371/ckpt/best_model-epoch=997.val_rmse=1.8306.val_mlm_accuracy=6.5476.ckpt

# Run
echo "Running script A expression"
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter A \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22046372/ckpt/best_model-epoch=915.val_rmse=0.9643.val_mlm_accuracy=6.6005.ckpt

# Run
echo "Running script B binding"
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter B \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22046373/ckpt/best_model-epoch=982.val_rmse=1.8394.val_mlm_accuracy=6.5486.ckpt

# Run
echo "Running script B expression"
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter B \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22046374/ckpt/best_model-epoch=982.val_rmse=0.9701.val_mlm_accuracy=6.5486.ckpt

# Run
echo "Running script C binding"
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter C \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22046375/ckpt/best_model-epoch=116.val_rmse=1.0375.val_mlm_accuracy=98.4353.ckpt

# Run
echo "Running script C expression"
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter C \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22046376/ckpt/best_model-epoch=111.val_rmse=0.5741.val_mlm_accuracy=98.4198.ckpt

# Run
echo "Running script D binding"
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter D \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22046377/ckpt/best_model-epoch=768.val_rmse=1.0184.val_mlm_accuracy=98.4435.ckpt

# Run
echo "Running script D expression"
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter D \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22046378/ckpt/best_model-epoch=921.val_rmse=0.5687.val_mlm_accuracy=98.4612.ckpt

# Run
echo "Running script E binding"
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter E \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22046379/ckpt/best_model-epoch=976.val_rmse=1.6761.val_mlm_accuracy=96.4777.ckpt

# Run
echo "Running script E expression"
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter E \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22095956/ckpt/best_model-epoch=890.val_rmse=0.9059.val_mlm_accuracy=96.4996.ckpt

# Run
echo "Running script F binding"
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter F \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22049876/ckpt/best_model-epoch=940.val_rmse=1.6947.val_mlm_accuracy=96.5183.ckpt

# Run
echo "Running script F expression"
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter F \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22095957/ckpt/best_model-epoch=960.val_rmse=0.9145.val_mlm_accuracy=96.5722.ckpt

# Run
echo "Running script G binding"
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter G \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22049878/ckpt/best_model-epoch=128.val_rmse=1.0366.val_mlm_accuracy=98.4478.ckpt

# Run
echo "Running script G expression"
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter G \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22049879/ckpt/best_model-epoch=137.val_rmse=0.5740.val_mlm_accuracy=98.4353.ckpt

# Run
echo "Running script H binding"
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter H \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22049880/ckpt/best_model-epoch=768.val_rmse=1.0199.val_mlm_accuracy=98.4435.ckpt

# Run
echo "Running script H expression"
python fig3-esm_mlm_gcn-r2_values.py \
--script_letter H \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn/version_22049882/ckpt/best_model-epoch=921.val_rmse=0.5673.val_mlm_accuracy=98.4613.ckpt