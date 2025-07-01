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
python fig3-bert_blstm-r2_values.py \
--script_letter A \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089236/ckpt/best_model-epoch=924.val_rmse=1.0460.val_mlm_accuracy=8.9331.ckpt

# Run
echo "Running script A expression"
python fig3-bert_blstm-r2_values.py \
--script_letter A \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089238/ckpt/best_model-epoch=924.val_rmse=0.5817.val_mlm_accuracy=8.9331.ckpt

# Run
echo "Running script B binding"
python fig3-bert_blstm-r2_values.py \
--script_letter B \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089239/ckpt/best_model-epoch=213.val_rmse=1.8273.val_mlm_accuracy=7.0561.ckpt

# Run
echo "Running script B expression"
python fig3-bert_blstm-r2_values.py \
--script_letter B \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089240/ckpt/best_model-epoch=265.val_rmse=0.9697.val_mlm_accuracy=7.7214.ckpt

# Run
echo "Running script C binding"
python fig3-bert_blstm-r2_values.py \
--script_letter C \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089241/ckpt/best_model-epoch=995.val_rmse=1.0091.val_mlm_accuracy=98.3702.ckpt

# Run
echo "Running script C expression"
python fig3-bert_blstm-r2_values.py \
--script_letter C \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089242/ckpt/best_model-epoch=921.val_rmse=0.5675.val_mlm_accuracy=98.4564.ckpt

# Run
echo "Running script D binding"
python fig3-bert_blstm-r2_values.py \
--script_letter D \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089243/ckpt/best_model-epoch=923.val_rmse=1.8343.val_mlm_accuracy=98.3852.ckpt

# Run
echo "Running script D expression"
python fig3-bert_blstm-r2_values.py \
--script_letter D \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089244/ckpt/best_model-epoch=486.val_rmse=0.9261.val_mlm_accuracy=98.4341.ckpt

# Run
echo "Running script E binding"
python fig3-bert_blstm-r2_values.py \
--script_letter E \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089245/ckpt/best_model-epoch=924.val_rmse=1.1322.val_mlm_accuracy=5.4756.ckpt

# Run
echo "Running script E expression"
python fig3-bert_blstm-r2_values.py \
--script_letter E \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089246/ckpt/best_model-epoch=985.val_rmse=0.6960.val_mlm_accuracy=6.0321.ckpt

# Run
echo "Running script F binding"
python fig3-bert_blstm-r2_values.py \
--script_letter F \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089248/ckpt/best_model-epoch=974.val_rmse=1.8113.val_mlm_accuracy=6.0130.ckpt

# Run
echo "Running script F expression"
python fig3-bert_blstm-r2_values.py \
--script_letter F \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089249/ckpt/best_model-epoch=896.val_rmse=0.9577.val_mlm_accuracy=6.9859.ckpt

# Run
echo "Running script G binding"
python fig3-bert_blstm-r2_values.py \
--script_letter G \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089250/ckpt/best_model-epoch=587.val_rmse=1.0118.val_mlm_accuracy=98.4265.ckpt

# Run
echo "Running script G expression"
python fig3-bert_blstm-r2_values.py \
--script_letter G \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089251/ckpt/best_model-epoch=924.val_rmse=0.5604.val_mlm_accuracy=98.3951.ckpt

# Run
echo "Running script H binding"
python fig3-bert_blstm-r2_values.py \
--script_letter H \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089252/ckpt/best_model-epoch=923.val_rmse=1.6253.val_mlm_accuracy=98.3852.ckpt

# Run
echo "Running script H expression"
python fig3-bert_blstm-r2_values.py \
--script_letter H \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_blstm/version_22089916/ckpt/best_model-epoch=950.val_rmse=0.8793.val_mlm_accuracy=98.4108.ckpt