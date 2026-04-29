#!/bin/bash

# Job 1
EMB_FILE1=ESM2_8M_base_cls
LOG_FILE1=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE1}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE1 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE1}_out.log 2> ${LOG_FILE1}_err.log

echo "Job complete (${EMB_FILE1})."

# Job 2
EMB_FILE2=ESM2_8M_finetuned_cls
LOG_FILE2=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE2}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE2 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE2}_out.log 2> ${LOG_FILE2}_err.log

echo "Job complete (${EMB_FILE2})."

# Job 3 
EMB_FILE3=ESM2_8M_base_mean
LOG_FILE3=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE3}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE3 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE3}_out.log 2> ${LOG_FILE3}_err.log

echo "Job complete (${EMB_FILE3})."

# Job 4
EMB_FILE1=ESM2_8M_finetuned_mean
LOG_FILE1=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE1}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE1 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE1}_out.log 2> ${LOG_FILE1}_err.log

echo "Job complete (${EMB_FILE1})."


# Job 5
EMB_FILE2=ESM2_150M_base_cls
LOG_FILE2=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE2}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE2 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE2}_out.log 2> ${LOG_FILE2}_err.log

echo "Job complete (${EMB_FILE2})."

# Job 6
EMB_FILE3=ESM2_150M_finetuned_cls
LOG_FILE3=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE3}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE3 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE3}_out.log 2> ${LOG_FILE3}_err.log

echo "Job complete (${EMB_FILE3})."

# Job 7
EMB_FILE1=ESM2_150M_base_mean
LOG_FILE1=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE1}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE1 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE1}_out.log 2> ${LOG_FILE1}_err.log

echo "Job complete (${EMB_FILE1})."

# Job 8
EMB_FILE2=ESM2_150M_finetuned_mean
LOG_FILE2=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE2}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE2 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE2}_out.log 2> ${LOG_FILE2}_err.log

echo "Job complete (${EMB_FILE2})."


# Job 9 
EMB_FILE3=ESM2_650M_base_cls
LOG_FILE3=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE3}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE3 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE3}_out.log 2> ${LOG_FILE3}_err.log

echo "Job complete (${EMB_FILE3})."

# Job 10
EMB_FILE1=ESM2_650M_finetuned_cls
LOG_FILE1=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE1}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE1 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE1}_out.log 2> ${LOG_FILE1}_err.log

echo "Job complete (${EMB_FILE1})."

# Job 11
EMB_FILE2=ESM2_650M_base_mean
LOG_FILE2=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE2}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE2 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE2}_out.log 2> ${LOG_FILE2}_err.log

echo "Job complete (${EMB_FILE2})."

# Job 12 
EMB_FILE3=ESM2_650M_finetuned_mean
LOG_FILE3=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE3}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE3 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE3}_out.log 2> ${LOG_FILE3}_err.log

echo "Job complete (${EMB_FILE3})."


# Job 13
EMB_FILE1=ESMC_300M_base_cls
LOG_FILE1=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE1}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE1 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE1}_out.log 2> ${LOG_FILE1}_err.log

echo "Job complete (${EMB_FILE1})."

# Job 14
EMB_FILE2=ESMC_300M_finetuned_cls
LOG_FILE2=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE2}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE2 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE2}_out.log 2> ${LOG_FILE2}_err.log

echo "Job complete (${EMB_FILE2})."

# Job 15 
EMB_FILE3=ESMC_300M_base_mean
LOG_FILE3=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE3}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE3 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE3}_out.log 2> ${LOG_FILE3}_err.log

echo "Job complete (${EMB_FILE3})."

# Job 16
EMB_FILE1=ESMC_300M_finetuned_mean
LOG_FILE1=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE1}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE1 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE1}_out.log 2> ${LOG_FILE1}_err.log

echo "Job complete (${EMB_FILE1})."


# Job 17
EMB_FILE3=ESMC_600M_base_cls
LOG_FILE3=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE3}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE3 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE3}_out.log 2> ${LOG_FILE3}_err.log

echo "Job complete (${EMB_FILE3})."

# Job 18
EMB_FILE4=ESMC_600M_finetuned_cls
LOG_FILE4=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE4}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE4 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE4}_out.log 2> ${LOG_FILE4}_err.log

echo "Job complete (${EMB_FILE4})."

# Job 19
EMB_FILE1=ESMC_600M_base_mean
LOG_FILE1=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE1}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE1 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE1}_out.log 2> ${LOG_FILE1}_err.log

echo "Job complete (${EMB_FILE1})."

# Job 20
EMB_FILE2=ESMC_600M_finetuned_mean
LOG_FILE2=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE2}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE2 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE2}_out.log 2> ${LOG_FILE2}_err.log

echo "Job complete (${EMB_FILE2})."


# Job 21
EMB_FILE1=ONE-HOT_mean
LOG_FILE1=logs/slurm_out/comp/blackwell/splitter_comp_${EMB_FILE1}

CUDA_VISIBLE_DEVICES=0 python split_comparison.py \
    --emb_file $EMB_FILE1 \
    --lr 1e-5 \
    --num_epochs 1000 \
    > ${LOG_FILE1}_out.log 2> ${LOG_FILE1}_err.log

echo "Job complete (${EMB_FILE1})."
