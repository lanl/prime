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
python fig3-bert_gcn_be-r2_values.py \
--script_letter A \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn_be/version_22090044/ckpt/best_model-epoch=517.val_be_rmse=2.1687.val_binding_rmse=1.9108.val_expression_rmse=1.0256.val_mlm_accuracy=8.1420.ckpt

# Run
echo "Running script B"
python fig3-bert_gcn_be-r2_values.py \
--script_letter B \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn_be/version_22090045/ckpt/best_model-epoch=771.val_be_rmse=2.1706.val_binding_rmse=1.9109.val_expression_rmse=1.0296.val_mlm_accuracy=6.7883.ckpt

# Run
echo "Running script C"
python fig3-bert_gcn_be-r2_values.py \
--script_letter C \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn_be/version_22090046/ckpt/best_model-epoch=987.val_be_rmse=1.5114.val_binding_rmse=1.2653.val_expression_rmse=0.8268.val_mlm_accuracy=98.4159.ckpt

# Run
echo "Running script D"
python fig3-bert_gcn_be-r2_values.py \
--script_letter D \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn_be/version_22090047/ckpt/best_model-epoch=412.val_be_rmse=2.1371.val_binding_rmse=1.9033.val_expression_rmse=0.9719.val_mlm_accuracy=98.4044.ckpt

# Run
echo "Running script E"
python fig3-bert_gcn_be-r2_values.py \
--script_letter E \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn_be/version_22090048/ckpt/best_model-epoch=145.val_be_rmse=2.2444.val_binding_rmse=2.0053.val_expression_rmse=1.0079.val_mlm_accuracy=6.3570.ckpt

# Run
echo "Running script F"
python fig3-bert_gcn_be-r2_values.py \
--script_letter F \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn_be/version_22090049/ckpt/best_model-epoch=564.val_be_rmse=2.3587.val_binding_rmse=2.0987.val_expression_rmse=1.0765.val_mlm_accuracy=6.4073.ckpt

# Run
echo "Running script G"
python fig3-bert_gcn_be-r2_values.py \
--script_letter G \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn_be/version_22090050/ckpt/best_model-epoch=957.val_be_rmse=1.5049.val_binding_rmse=1.2599.val_expression_rmse=0.8231.val_mlm_accuracy=98.4744.ckpt

# Run
echo "Running script H"
python fig3-bert_gcn_be-r2_values.py \
--script_letter H \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn_be/version_22090051/ckpt/best_model-epoch=00.val_be_rmse=2.2882.val_binding_rmse=1.9226.val_expression_rmse=1.2407.val_mlm_accuracy=6.5949.ckpt