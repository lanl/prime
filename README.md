### O# (O4987)
# PRIME: **P**rotein **R**epresentation **I**nference for **M**utation **E**valuation
Understand the sequence to function, or genotype-phenotype, relationship of proteins by utilizing a language model-based approach. In particular, focusing on tailoring protein language models to predict protein mutation phenotypes, such as binding affinity or level of expression. Establish a framework for pathogen biosurveillance.

## Contents
### Documentation
- [Data Processing](https://github.com/lanl/prime/tree/main/notebooks/data_processing)
- [Clustering](https://github.com/lanl/prime/tree/main/notebooks/clustering)
    - Analysis using TSNE + HDBSCAN: `clustering/betacov`, `clustering/rbd`
    - NEW: analysis using `cuML` GPU for UMAP + HDBSCAN: `clustering/blackwell/betacov`, `clustering/blackwell/rbd`
        - Used NVIDIA RTX PRO 6000 (Blackwell edition) GPUs, requirements for package installs in `requirements/blackwell_requirements.txt`
        - TSNE + HDBSCAN using `cuML` also in here, but manuscript results do not reflect these results. This was more of an exploration in case we needed it.
- [Model Comparison & Development](https://github.com/lanl/prime/tree/main/notebooks/models)
- [Phylogenetic Analysis](https://github.com/lanl/prime/tree/main/notebooks/phylogenetic_analysis)

### Random vs Position-Stratified Split
- NEW, located [here](https://github.com/lanl/prime/tree/main/random_vs_stratified_split)
- Used NVIDIA RTX PRO 6000 (Blackwell edition) GPUs, requirements for package installs in `requirements/blackwell_requirements.txt`

### Models
Non masked language models do not use the EsmForMaskedLM head.
Note: We used the 8M parameter ESM2 model in our training. This is the smallest ESM2 model. This version of the model was utilized because we started on our laptops. We continued to use this in our HPC environments due to time limit and resource restrictions. 
- [ESM MLM](https://github.com/lanl/prime/tree/main/src/pnlp/ESM_MLM) (ESM-RBD; masked language model)
    - Model weights for 8M model [available](/panfs/biopan03/prime_ml/prime/src/pnlp/ESM_MLM/model_weights)
- [BERT MLM](https://github.com/lanl/prime/tree/main/src/pnlp/BERT_MLM) (masked language model)
- [ESM TL](https://github.com/lanl/prime/tree/main/src/pnlp/ESM_TL) (transfer learning models; non masked language models and masked language models)
    - Non Masked Language Models
        - ESM BLSTM BE (multi task)
        - ESM BLSTM (single task)
        - ESM FCN BE (multi task)
        - ESM FCN (single task)
        - ESM GCN BE (multi task)
        - ESM GCN BE (single task)
    - Masked Language Models
        - ESM BLSTM BE (multi task)
        - ESM BLSTM (single task)
        - ESM FCN BE (multi task)
        - ESM FCN (single task)
        - ESM GCN BE (multi task)
        - ESM GCN BE (single task)
    - Random vs Position Stratified Split Evaluation
- [BERT TL](https://github.com/lanl/prime/tree/main/src/pnlp/BERT_TL) (transfer learning models; masked language models)
    - BERT BLSTM BE (multi task)
    - BERT BLSTM (single task)
    - BERT GCN BE (multi task)
    - BERT GCN BE (single task)

## Installation
1) Git clone the repo.
2) Set up the environment.
    - We have two environments we use. This is because we gained access to NVIDIA RTX PRO 6000 (Blackwell edition) GPUs for revisions. This requires a different version of many of the software packages we use. As a result, this section has changed. 
    - Blackwell env
        1) Create a conda env: `conda create -n blackwell_prime python=3.13.12 -y`
        2) Install the packages: `pip install -r requirements/blackwell_prime_requirements.txt`
            - You can also run these commands with the flag `--no-cache-dir` if your folder where pip sends downloads to cache is full.
        3) From the top of the `prime` directory run this command to install the rest of the dependencies: `pip install -e .`
    - Original env
        1) Create a conda env: `conda create -n prime python=3.11.5 -y`
        2) Install the packages: `pip install -r requirements/prime_requirements.txt`
            - You can also run these commands with the flag `--no-cache-dir` if your folder where pip sends downloads to cache is full.
        3) From the top of the `prime` directory run this command to install the rest of the dependencies: `pip install -e .`

Other requirements:
- NVIDIA GPU is recommended 

## Usage
### ESM MLM or BERT MLM
To run the BERT MLM models, all you need to do is make sure your environment is active, and then run the command `python lightning-bert.py`. By default, if you are not in a SLURM environment, it is set up to use a single GPU on a single node. You can adjust the number of epochs within the script using the variable `max_epochs`. 

For the ESM MLM models (now including ESM-C and other ESM-2 versions), you need to make sure the environment is active, and then run the command `python lightning-esm.py` or `python lightning-esmc.py`. By default, the `--esm2_model` or `--esmc_model` flag is set to the smallest available model; otherwise, you can choose which model you want. Note that you will likely need to adjust batch size if the model is large to fit on your GPU.

It is recommended to run in an environment with multiple GPUs, preferably in a SLURM environment, to take advantage of using Pytorch Lightning. If you would like to use SLURM with multiple GPUs, here is an example bash script using ESM MLM:
```bash
#!/bin/bash
#SBATCH --job-name=ESM_MLM
#SBATCH --output=logs/version_%j/slurm_out/%j.out	     # Redirect standard out to slurm_outs
#SBATCH --error=logs/version_%j/slurm_out/%j.err	     # Redirect standard err to slurm_outs
#SBATCH --partition=gpu                                  # GPU partition
#SBATCH --time=4:00:00                                   # Max time limit
#SBATCH --nodes=2                                        # Number of nodes
#SBATCH --ntasks-per-node=4                              # Number of processes per node (match GPU count)
#SBATCH --exclusive                                      # Use entire node exclusively

# Load environment
conda activate prime

# Run
srun python lightning-esm.py --esm2_model esm2_8m
```
This SLURM script utilizes 8 total GPUs, 4 on each node. When using SLURM, `srun` is necessary in order to detect all of the devices properly.

### ESM TL or BERT TL
To run any of the ESM TL or BERT TL models, there are flags you can set from the command line.
- For ESM TL
    - `--binding_or_expression`: Set 'binding' or 'expression' as target; type=str, default="binding"
        - This flag does not exist for multi-task models (i.e., ESM BLSTM BE, ESM FCN BE, ESM GCN BE), only for the single task.
    - `--lr`: Set learning rate; type=float, default=1e-5
    - `--num_epochs`: Number of epochs; type=int, default=100
    - `--from_checkpoint`: Path to existing checkpoint to resume training from; type=str, default=None
    - `--from_esm_mlm`: Path to pretrained ESM_MLM checkpoint; type=str, default=None
    - `--freeze_esm`: Whether to freeze ESM model weights. Abscence of flag sets to False

- For BERT TL
    - `--binding_or_expression`: Set 'binding' or 'expression' as target; type=str, default="binding"
        - This flag does not exist for multi-task models (i.e., BERT BLSTM BE, BERT GCN BE), only for the single task.
    - `--lr`: Set learning rate; type=float, default=1e-5
    - `--num_epochs`: Number of epochs; type=int, default=100
    - `--from_checkpoint`: Path to existing checkpoint to resume training from; type=str, default=None
    - `--from_bert_mlm`: Path to pretrained BERT_MLM checkpoint; type=str, default=None
    - `--freeze_bert`: Whether to freeze BERT model weights. Abscence of flag sets to False.

Here is an example SLURM bash script for running ESM FCN BE, where we run for 20 epochs at a learning rate of 1e-4 after loading in the pretrained ESM MLM weights:
```bash
#!/bin/bash
#SBATCH --job-name=ESM_FCN_BE
#SBATCH --output=logs/esm_mlm_fcn_be/version_%j/slurm_out/%j.out    # Redirect standard out to slurm_outs
#SBATCH --error=logs/esm_mlm_fcn_be/version_%j/slurm_out/%j.err     # Redirect standard err to slurm_outs
#SBATCH --partition=gpu                                             # GPU partition
#SBATCH --time=4:00:00                                              # Max time limit
#SBATCH --nodes=2                                                   # Number of nodes
#SBATCH --ntasks-per-node=4                                         # Number of processes per node (match GPU count)
#SBATCH --exclusive                                                 # Use entire node exclusively

# Load environment
conda activate prime

# Run
srun python lightning-esm_mlm_fcn_be.py \
--num_epochs 20 \
--lr 1e-4 \
--from_esm_mlm best_model-epoch=73.val_loss=0.0022.val_accuracy=99.6612.ckpt
```
You could also run this from the command line without SLURM as well, without using the `srun` part of the command. Again, I would recommend using SLURM to take advantage of Pytorch Lightning. All of this code was written and ran using a SLURM environment.

## Citation
Gibson, K., Li, PE., Li, V. et al. PRIME: An evaluation framework for protein representation inference and generalization in viral mutation space. BMC Genomics (2026). https://doi.org/10.1186/s12864-026-12976-5

```
@article{gibson_prime_2026,
	title = {{PRIME}: {An} evaluation framework for protein representation inference and generalization in viral mutation space},
	issn = {1471-2164},
	url = {https://doi.org/10.1186/s12864-026-12976-5},
	doi = {10.1186/s12864-026-12976-5},
	abstract = {Protein language models (PLMs) have revolutionized protein fitness prediction, yet their application to rapidly evolving viral pathogens is often confounded by extreme sequence homology. This homology leads to “data leakage” in standard random validation splits, yielding inflated performance metrics that fail to translate into real-world biosurveillance utility.},
	journal = {BMC Genomics},
	author = {Gibson, Kaetlyn and Li, Po-E and Li, Valerie and Dix, Martha and Hung, Li-Wei and Stelle, George Widgery and Babinski, Michal and Chain, Patrick and Hu, Bin},
	month = may,
	year = {2026},
}
```
