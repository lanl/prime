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
python fig3-bert_gcn-r2_values.py \
--script_letter A \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090052/ckpt/best_model-epoch=517.val_rmse=1.8944.val_mlm_accuracy=8.1420.ckpt

# Run
echo "Running script A expression"
python fig3-bert_gcn-r2_values.py \
--script_letter A \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090053/ckpt/best_model-epoch=771.val_rmse=0.9928.val_mlm_accuracy=6.7883.ckpt

# Run
echo "Running script B binding"
python fig3-bert_gcn-r2_values.py \
--script_letter B \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090054/ckpt/best_model-epoch=830.val_rmse=1.8958.val_mlm_accuracy=8.3308.ckpt

# Run
echo "Running script B expression"
python fig3-bert_gcn-r2_values.py \
--script_letter B \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090055/ckpt/best_model-epoch=644.val_rmse=0.9936.val_mlm_accuracy=8.6166.ckpt

# Run
echo "Running script C binding"
python fig3-bert_gcn-r2_values.py \
--script_letter C \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090056/ckpt/best_model-epoch=924.val_rmse=1.0810.val_mlm_accuracy=98.3951.ckpt

# Run
echo "Running script C expression"
python fig3-bert_gcn-r2_values.py \
--script_letter C \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090057/ckpt/best_model-epoch=995.val_rmse=0.6182.val_mlm_accuracy=98.3702.ckpt

# Run
echo "Running script D binding"
python fig3-bert_gcn-r2_values.py \
--script_letter D \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090058/ckpt/best_model-epoch=555.val_rmse=1.8098.val_mlm_accuracy=98.3993.ckpt

# Run
echo "Running script D expression"
python fig3-bert_gcn-r2_values.py \
--script_letter D \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090059/ckpt/best_model-epoch=88.val_rmse=0.9955.val_mlm_accuracy=79.4655.ckpt

# Run
echo "Running script E binding"
python fig3-bert_gcn-r2_values.py \
--script_letter E \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090060/ckpt/best_model-epoch=105.val_rmse=1.9165.val_mlm_accuracy=5.5775.ckpt

# Run
echo "Running script E expression"
python fig3-bert_gcn-r2_values.py \
--script_letter E \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090064/ckpt/best_model-epoch=789.val_rmse=1.0573.val_mlm_accuracy=6.5915.ckpt

# Run
echo "Running script F binding"
python fig3-bert_gcn-r2_values.py \
--script_letter F \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090065/ckpt/best_model-epoch=511.val_rmse=2.0026.val_mlm_accuracy=5.1636.ckpt

# Run
echo "Running script F expression"
python fig3-bert_gcn-r2_values.py \
--script_letter F \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090066/ckpt/best_model-epoch=675.val_rmse=1.1870.val_mlm_accuracy=5.0557.ckpt

# Run
echo "Running script G binding"
python fig3-bert_gcn-r2_values.py \
--script_letter G \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090067/ckpt/best_model-epoch=817.val_rmse=1.0784.val_mlm_accuracy=98.3734.ckpt

# Run
echo "Running script G expression"
python fig3-bert_gcn-r2_values.py \
--script_letter G \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090068/ckpt/best_model-epoch=994.val_rmse=0.6080.val_mlm_accuracy=98.4524.ckpt

# Run
echo "Running script H binding"
python fig3-bert_gcn-r2_values.py \
--script_letter H \
--binding_or_expression binding \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090069/ckpt/best_model-epoch=982.val_rmse=1.7050.val_mlm_accuracy=98.3992.ckpt

# Run
echo "Running script H expression"
python fig3-bert_gcn-r2_values.py \
--script_letter H \
--binding_or_expression expression \
--from_saved ../../../../src/pnlp/BERT_TL/logs/bert_gcn/version_22090070/ckpt/best_model-epoch=992.val_rmse=0.9157.val_mlm_accuracy=98.3743.ckpt