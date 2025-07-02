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
python fig3-esm_mlm_fcn_be-r2_values.py \
--script_letter A \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn_be/version_22045123/ckpt/best_model-epoch=898.val_be_rmse=2.0976.val_binding_rmse=1.8578.val_expression_rmse=0.9739.val_mlm_accuracy=6.5062.ckpt

# Run
echo "Running script B"
python fig3-esm_mlm_fcn_be-r2_values.py \
--script_letter B \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn_be/version_22045124/ckpt/best_model-epoch=981.val_be_rmse=2.1078.val_binding_rmse=1.8672.val_expression_rmse=0.9780.val_mlm_accuracy=6.5781.ckpt

# Run
echo "Running script C"
python fig3-esm_mlm_fcn_be-r2_values.py \
--script_letter C \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn_be/version_22047980/ckpt/best_model-epoch=145.val_be_rmse=1.1965.val_binding_rmse=1.0452.val_expression_rmse=0.5825.val_mlm_accuracy=98.3934.ckpt

# Run
echo "Running script D"
python fig3-esm_mlm_fcn_be-r2_values.py \
--script_letter D \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn_be/version_22049883/ckpt/best_model-epoch=832.val_be_rmse=1.1675.val_binding_rmse=1.0121.val_expression_rmse=0.5820.val_mlm_accuracy=98.4372.ckpt

# Run
echo "Running script E"
python fig3-esm_mlm_fcn_be-r2_values.py \
--script_letter E \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn_be/version_22045128/ckpt/best_model-epoch=951.val_be_rmse=2.0094.val_binding_rmse=1.7757.val_expression_rmse=0.9406.val_mlm_accuracy=96.5235.ckpt

# Run
echo "Running script F"
python fig3-esm_mlm_fcn_be-r2_values.py \
--script_letter F \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn_be/version_22045129/ckpt/best_model-epoch=931.val_be_rmse=2.0332.val_binding_rmse=1.7969.val_expression_rmse=0.9514.val_mlm_accuracy=96.5775.ckpt

# Run
echo "Running script G"
python fig3-esm_mlm_fcn_be-r2_values.py \
--script_letter G \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn_be/version_22047982/ckpt/best_model-epoch=98.val_be_rmse=1.1846.val_binding_rmse=1.0296.val_expression_rmse=0.5858.val_mlm_accuracy=98.4021.ckpt

# Run
echo "Running script H"
python fig3-esm_mlm_fcn_be-r2_values.py \
--script_letter H \
--from_saved ../../../../src/pnlp/ESM_TL/logs/esm_mlm_fcn_be/version_22049884/ckpt/best_model-epoch=832.val_be_rmse=1.1660.val_binding_rmse=1.0121.val_expression_rmse=0.5790.val_mlm_accuracy=98.4390.ckpt