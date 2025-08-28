# PRIME: **P**rotein **R**epresentation **I**nference for **M**utation **E**valuation
Understand the sequence to function, or genotype-phenotype, relationship of proteins by utilizing a language model-based approach. In particular, focusing on tailoring protein language models to predict protein mutation phenotypes, such as binding affinity or level of expression. Establish a framework for pathogen biosurveillance.

## Contents
### Documentation
- [Data Processing](https://github.com/kae-gi/Spike_NLP-Lightning/tree/main/notebooks/data_processing)
- [Clustering](https://github.com/kae-gi/Spike_NLP-Lightning/tree/main/notebooks/clustering)
- [Model Comparison & Development](https://github.com/kae-gi/Spike_NLP-Lightning/tree/main/notebooks/models)
- [Phylogenetic Analysis](https://github.com/kae-gi/Spike_NLP-Lightning/tree/main/notebooks/phylogenetic_analysis)

### Models
Non masked language models do not use the EsmForMaskedLM head.
- [ESM MLM](https://github.com/kae-gi/Spike_NLP-Lightning/tree/main/src/pnlp/ESM_MLM) (ESM-RBD; masked language model)
- [BERT MLM](https://github.com/kae-gi/Spike_NLP-Lightning/tree/main/src/pnlp/BERT_MLM) (masked language model)
- [ESM TL](https://github.com/kae-gi/Spike_NLP-Lightning/tree/main/src/pnlp/ESM_TL) (transfer learning models; non masked language models and masked language models)
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
- [BERT TL](https://github.com/kae-gi/Spike_NLP-Lightning/tree/main/src/pnlp/BERT_TL) (transfer learning models; masked language models)
    - BERT BLSTM BE (multi task)
    - BERT BLSTM (single task)
    - BERT GCN BE (multi task)
    - BERT GCN BE (single task)

## Installation
1) Git clone the repo.
2) Set up the environment.
    1) Create or locate your `venvs` folder.
    2) Create your environment. Our environment is set to use Python 3.11.5.
        - `python3.11 -m venv ./venvs/spike`, adjust to where your `venvs` folder is located.
    3) Activate your environment.
        - `source ./venvs/spike/bin/activate`, adjust to where your `venvs` folder is located.
    4) From the top of the `Spike_NLP-Lightning` directory run this command to install the dependencies: 
        - `pip install -e .`
        - You may also need to run `pip install -r requirements/torchreq.txt`, but torch should be installed through the other requirements.
        - You can also run these commands with the flag `--no-cache-dir` if your folder where pip sends downloads to cache is full.

Other requirements:
- NVIDIA GPU is recommended

## Usage
### ESM MLM or BERT MLM
To run the ESM MLM and BERT MLM models, all you need to do is make sure your environment is active, and then run the command `python lightning-esm.py` or `python lightning-bert.py`. By default, if you are not in a SLURM environment, it is set up to use a single GPU on a single node. You can adjust the number of epochs within the script using the variable `max_epochs`.

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
source venvs/spike/bin/activate

# Run
srun python lightning-esm.py
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
source venvs/spike/bin/activate

# Run
srun python lightning-esm_mlm_fcn_be.py \
--num_epochs 20 \
--lr 1e-4 \
--from_esm_mlm best_model-epoch=73.val_loss=0.0022.val_accuracy=99.6612.ckpt
```
You could also run this from the command line without SLURM as well, without using the `srun` part of the command. Again, I would recommend using SLURM to take advantage of Pytorch Lightning. All of this code was written and ran using a SLURM environment.

## License
[MIT](https://github.com/kae-gi/Spike_NLP-Lightning/blob/main/LICENSE.md)

## Citation
WIP