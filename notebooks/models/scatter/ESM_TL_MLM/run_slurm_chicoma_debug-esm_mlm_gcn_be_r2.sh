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
python fig3-esm_mlm_gcn_be-r2_values.py \
--script_letter A \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn_be/version_22046318/ckpt/best_model-epoch=997.val_be_rmse=2.1113.val_binding_rmse=1.8536.val_expression_rmse=1.0108.val_mlm_accuracy=6.5476.ckpt

# Run
echo "Running script B"
python fig3-esm_mlm_gcn_be-r2_values.py \
--script_letter B \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn_be/version_22049866/ckpt/best_model-epoch=982.val_be_rmse=2.1176.val_binding_rmse=1.8641.val_expression_rmse=1.0048.val_mlm_accuracy=6.5486.ckpt

# Run
echo "Running script C"
python fig3-esm_mlm_gcn_be-r2_values.py \
--script_letter C \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn_be/version_22049867/ckpt/best_model-epoch=129.val_be_rmse=1.4698.val_binding_rmse=1.1991.val_expression_rmse=0.8500.val_mlm_accuracy=98.4204.ckpt

# Run
echo "Running script D"
python fig3-esm_mlm_gcn_be-r2_values.py \
--script_letter D \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn_be/version_22047986/ckpt/best_model-epoch=962.val_be_rmse=1.4611.val_binding_rmse=1.1968.val_expression_rmse=0.8381.val_mlm_accuracy=98.4289.ckpt

# Run
echo "Running script E"
python fig3-esm_mlm_gcn_be-r2_values.py \
--script_letter E \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn_be/version_22049868/ckpt/best_model-epoch=940.val_be_rmse=1.9764.val_binding_rmse=1.7230.val_expression_rmse=0.9682.val_mlm_accuracy=96.5183.ckpt

# Run
echo "Running script F"
python fig3-esm_mlm_gcn_be-r2_values.py \
--script_letter F \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn_be/version_22049869/ckpt/best_model-epoch=940.val_be_rmse=1.9964.val_binding_rmse=1.7494.val_expression_rmse=0.9619.val_mlm_accuracy=96.5183.ckpt

# Run
echo "Running script G"
python fig3-esm_mlm_gcn_be-r2_values.py \
--script_letter G \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn_be/version_22049870/ckpt/best_model-epoch=152.val_be_rmse=1.4694.val_binding_rmse=1.1966.val_expression_rmse=0.8528.val_mlm_accuracy=98.4140.ckpt

# Run
echo "Running script H"
python fig3-esm_mlm_gcn_be-r2_values.py \
--script_letter H \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_gcn_be/version_22049871/ckpt/best_model-epoch=832.val_be_rmse=1.4587.val_binding_rmse=1.1771.val_expression_rmse=0.8615.val_mlm_accuracy=98.4379.ckpt