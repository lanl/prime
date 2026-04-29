#!/bin/bash

# Job 1
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esm2_model esm2_8m \
    --pooling cls 

# Job 2
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esm2_model esm2_8m \
    --pooling cls \
    --finetuned_ckpt ../../ESM_MLM/logs/version_23018343-esm2_8M_15epochs/ckpt/best_model-epoch=14.val_loss=0.0027.val_accuracy=99.5800.ckpt

# Job 3
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esm2_model esm2_8m \
    --pooling mean 

# Job 4
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esm2_model esm2_8m \
    --pooling mean \
    --finetuned_ckpt ../../ESM_MLM/logs/version_23018343-esm2_8M_15epochs/ckpt/best_model-epoch=14.val_loss=0.0027.val_accuracy=99.5800.ckpt


# Job 5
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esm2_model esm2_150m \
    --pooling cls 

# Job 6
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esm2_model esm2_150m \
    --pooling cls \
    --finetuned_ckpt ../../ESM_MLM/logs/version_23019522-esm2_150M_15epochs/ckpt/best_model-epoch=14.val_loss=0.0024.val_accuracy=99.6278.ckpt

# Job 7
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esm2_model esm2_150m \
    --pooling mean

# Job 8
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esm2_model esm2_150m \
    --pooling mean \
    --finetuned_ckpt ../../ESM_MLM/logs/version_23019522-esm2_150M_15epochs/ckpt/best_model-epoch=14.val_loss=0.0024.val_accuracy=99.6278.ckpt


# Job 9
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esm2_model esm2_650m \
    --pooling cls 

# Job 10
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esm2_model esm2_650m \
    --pooling cls \
    --finetuned_ckpt ../../ESM_MLM/logs/version_23021200-esm2_650M_15epochs/ckpt/best_model-epoch=14.val_loss=0.0022.val_accuracy=99.6470.ckpt

# Job 11
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esm2_model esm2_650m \
    --pooling mean

# Job 12
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esm2_model esm2_650m \
    --pooling mean \
    --finetuned_ckpt ../../ESM_MLM/logs/version_23021200-esm2_650M_15epochs/ckpt/best_model-epoch=14.val_loss=0.0022.val_accuracy=99.6470.ckpt


# Job 13
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esmc_model esmc_300m \
    --pooling cls 

# Job 14
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esmc_model esmc_300m \
    --pooling cls \
    --finetuned_ckpt ../../ESM_MLM/logs/version_23018365-esmc_300M_15epochs/ckpt/best_model-epoch=11.val_loss=0.0151.val_accuracy=99.6495.ckpt

# Job 15
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esmc_model esmc_300m \
    --pooling mean

# Job 16
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esmc_model esmc_300m \
    --pooling mean \
    --finetuned_ckpt ../../ESM_MLM/logs/version_23018365-esmc_300M_15epochs/ckpt/best_model-epoch=11.val_loss=0.0151.val_accuracy=99.6495.ckpt


# Job 17
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esmc_model esmc_600m \
    --pooling cls 

# Job 18
 CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esmc_model esmc_600m \
    --pooling cls \
    --finetuned_ckpt ../../ESM_MLM/logs/version_22887805-esmc_600M_15epochs/ckpt/best_model-epoch=12.val_loss=0.0149.val_accuracy=99.6531.ckpt

# Job 19
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esmc_model esmc_600m \
    --pooling mean 

# Job 20
CUDA_VISIBLE_DEVICES=0 python esm_embedder.py \
    --esmc_model esmc_600m \
    --pooling mean \
    --finetuned_ckpt ../../ESM_MLM/logs/version_22887805-esmc_600M_15epochs/ckpt/best_model-epoch=12.val_loss=0.0149.val_accuracy=99.6531.ckpt


# Job 21
CUDA_VISIBLE_DEVICES=0 python one_hot_embedder.py \
    --pooling mean 


