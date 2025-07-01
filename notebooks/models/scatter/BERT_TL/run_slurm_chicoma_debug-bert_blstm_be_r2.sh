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
python fig3-bert_blstm_be-r2_values.py \
--script_letter A \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm_be/version_22089905/ckpt/best_model-epoch=972.val_be_rmse=1.3513.val_binding_rmse=1.1619.val_expression_rmse=0.6899.val_mlm_accuracy=6.8979.ckpt

# Run
echo "Running script B"
python fig3-bert_blstm_be-r2_values.py \
--script_letter B \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm_be/version_22089906/ckpt/best_model-epoch=362.val_be_rmse=2.0812.val_binding_rmse=1.8380.val_expression_rmse=0.9763.val_mlm_accuracy=7.9926.ckpt

# Run
echo "Running script C"
python fig3-bert_blstm_be-r2_values.py \
--script_letter C \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm_be/version_22089907/ckpt/best_model-epoch=924.val_be_rmse=1.1504.val_binding_rmse=0.9998.val_expression_rmse=0.5692.val_mlm_accuracy=98.3951.ckpt

# Run
echo "Running script D"
python fig3-bert_blstm_be-r2_values.py \
--script_letter D \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm_be/version_22089908/ckpt/best_model-epoch=651.val_be_rmse=2.0821.val_binding_rmse=1.8393.val_expression_rmse=0.9758.val_mlm_accuracy=98.4246.ckpt

# Run
echo "Running script E"
python fig3-bert_blstm_be-r2_values.py \
--script_letter E \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm_be/version_22089909/ckpt/best_model-epoch=989.val_be_rmse=1.3239.val_binding_rmse=1.1577.val_expression_rmse=0.6421.val_mlm_accuracy=5.5673.ckpt

# Run
echo "Running script F"
python fig3-bert_blstm_be-r2_values.py \
--script_letter F \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm_be/version_22089911/ckpt/best_model-epoch=157.val_be_rmse=2.1492.val_binding_rmse=1.9040.val_expression_rmse=0.9969.val_mlm_accuracy=5.3354.ckpt

# Run
echo "Running script G"
python fig3-bert_blstm_be-r2_values.py \
--script_letter G \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm_be/version_22089912/ckpt/best_model-epoch=924.val_be_rmse=1.1493.val_binding_rmse=0.9991.val_expression_rmse=0.5680.val_mlm_accuracy=98.3951.ckpt

# Run
echo "Running script H"
python fig3-bert_blstm_be-r2_values.py \
--script_letter H \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm_be/version_22089913/ckpt/best_model-epoch=631.val_be_rmse=1.9249.val_binding_rmse=1.6767.val_expression_rmse=0.9456.val_mlm_accuracy=98.3792.ckpt