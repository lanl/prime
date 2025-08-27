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


## License
[MIT](https://github.com/kae-gi/Spike_NLP-Lightning/blob/main/LICENSE.md)

## Citation
WIP