#!/usr/bin/env python
"""
ESM Model embedders.
"""
import os
import torch
import argparse
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from transformers import EsmTokenizer, EsmModel
from huggingface_hub.constants import HF_HUB_CACHE

from pnlp.ESM_TL.random_vs_stratified_split.esmc_util import load_from_cache

class Esm2Embedder(torch.nn.Module):
    def __init__(self, esm_version="facebook/esm2_t6_8M_UR50D", cache_dir=None, pooling="mean", max_length=1022, finetuned_ckpt=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EsmModel.from_pretrained(esm_version, cache_dir=cache_dir).to(self.device)
        self.tokenizer = EsmTokenizer.from_pretrained(esm_version, cache_dir=cache_dir)
        self.pooling = pooling
        self.max_length = max_length

        print(f"Loaded {esm_version} on {self.device}.")

        if finetuned_ckpt is not None:
            self._load_finetuned_ckpt(finetuned_ckpt)

    def _load_finetuned_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("model.esm."):
                k = k[len("model.esm."):]

            if k.startswith("lm_head"):
                continue

            new_state[k] = v

        missing, unexpected = self.model.load_state_dict(new_state, strict=False)

        allowed_missing = {"pooler.dense.weight", "pooler.dense.bias"}
        real_missing = [k for k in missing if k not in allowed_missing]

        if real_missing:
            raise RuntimeError(f"Unexpected missing keys: {real_missing}")
        
        print("Fine-tuned MLM weights loaded successfully.")
    
    def tokenize(self, seq):
        tokenized_seq = self.tokenizer(seq, return_tensors="pt", truncation=True, max_length=self.max_length+2) # CLS + EOS
        return {k: v.to(self.device) for k, v in tokenized_seq.items()}

    def forward(self, seq):
        # Tokenize sequence
        tokenized_seq = self.tokenize(seq)

        # Get embedding
        self.model.eval()
        with torch.inference_mode():    # Disable gradient calculations, disable autocasting  
            esm_last_hidden_state = self.model(**tokenized_seq).last_hidden_state # shape: (batch_size, sequence_length, embedding_dim)
            
            if self.pooling == "mean":
                embedding = esm_last_hidden_state[:, 1:-1, :].mean(dim=1).squeeze() # Mean pool, exclude CLS + EOS; shape: [embedding_dim]
            elif self.pooling == "cls":
                embedding = esm_last_hidden_state[:, 0, :].squeeze() # CLS token embedding; shape: [embedding_dim]
            return embedding.detach().to("cpu")
        
class EsmCEmbedder(torch.nn.Module):
    def __init__(self, esm_version="esmc_300m_local", cache_dir=None, pooling="mean", max_length=2046, finetuned_ckpt=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_from_cache(esm_version, cache_dir=cache_dir).to(self.device, dtype=torch.float32)
        self.tokenizer = self.model.tokenizer
        self.pooling = pooling
        self.max_length = max_length

        print(f"Loaded {esm_version} on {self.device}")

        if finetuned_ckpt is not None:
            self._load_finetuned_ckpt(finetuned_ckpt)

    def _load_finetuned_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                k = k[len("model."):]

            new_state[k] = v

        self.model.load_state_dict(new_state, strict=True)
        
        print("Fine-tuned MLM weights loaded successfully.")
    
    def tokenize(self, seq):
        tokenized_seq = self.tokenizer(seq, return_tensors="pt", truncation=True, max_length=self.max_length+2) # CLS + EOS
        return {k: v.to(self.device) for k, v in tokenized_seq.items()}

    def forward(self, seq):
        # Tokenize sequence
        tokenized_seq = self.tokenize(seq)

        # Get embedding
        self.model.eval()
        with torch.inference_mode():    # Disable gradient calculations, disable autocasting  
            esm_last_hidden_state = self.model.forward(sequence_tokens=tokenized_seq["input_ids"]).embeddings # shape: (batch_size, sequence_length, embedding_dim)
            
            if self.pooling == "mean":
                embedding = esm_last_hidden_state[:, 1:-1, :].mean(dim=1).squeeze() # Mean pool, exclude CLS + EOS; shape: [embedding_dim]
            elif self.pooling == "cls":
                embedding = esm_last_hidden_state[:, 0, :].squeeze() # CLS token embedding; shape: [embedding_dim]
            return embedding.detach().to("cpu")

def load_fasta_sequences(fasta_file):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        protein_id = record.id.split("|")[1] if "train" in fasta_file else record.id  # Extract UniProt ID
        sequences[protein_id] = str(record.seq)

    print(f"Number of sequences in {fasta_file}:", len(sequences))
    return sequences

if __name__ == "__main__":
    # Embedder setup
    ESMC_MODELS = {
        "esmc_300m": "esmc_300m_local", 
        "esmc_600m": "esmc_600m_local",
    }
    ESM2_MODELS = {
        "esm2_8m": "facebook/esm2_t6_8M_UR50D",
        "esm2_150m": "facebook/esm2_t30_150M_UR50D",
        "esm2_650m": "facebook/esm2_t33_650M_UR50D",
    }

    parser = argparse.ArgumentParser(description="Run ESM Embedder")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--esmc_model", type=str, choices=list(ESMC_MODELS.keys()), 
                             help="ESMC Model version to use. Available: %(choices)s")
    model_group.add_argument("--esm2_model", type=str, choices=list(ESM2_MODELS.keys()), 
                             help="ESM2 model version to use. Available: %(choices)s")
    parser.add_argument("--cache_dir", type=str, default="../../../../.cache", # HF_HUB_CACHE
                        help="Cache directory for model weights (default: %(default)s).")
    parser.add_argument("--pooling", type=str, default="mean", choices=list(["mean", "cls"]), 
                        help="Pooling method for the embeddings (default: %(default)s). Available: %(choices)s")
    parser.add_argument("--finetuned_ckpt", type=str, default=None,
                        help="Path to a Lightning .ckpt or other weights file containing a fine-tuned ESM MLM model. "
                        "If provided, weights will be loaded into the ESM model before embedding (default: %(default)s).")
    args = parser.parse_args()

    if args.esmc_model:
        model_version = args.esmc_model
        embedder = EsmCEmbedder(ESMC_MODELS[args.esmc_model], cache_dir=args.cache_dir, pooling=args.pooling, finetuned_ckpt=args.finetuned_ckpt) 

    elif args.esm2_model:
        model_version = args.esm2_model
        embedder = Esm2Embedder(ESM2_MODELS[args.esm2_model], cache_dir=args.cache_dir, pooling=args.pooling, finetuned_ckpt=args.finetuned_ckpt) 

    # Load data
    data_dir = "../../../../data/dms"
    original_set = pd.read_csv(os.path.join(data_dir, "mutation_combined_DMS_OLD.csv"))

    # Embed sequences
    embeddings = []
    protein_ids = []

    N = None  # Set to None to embed all sequences
    df_iter = original_set if N is None else original_set.head(N)
    
    for row in tqdm(df_iter.itertuples(index=False), total=len(df_iter), desc="Embedding sequences"):
        protein_id = row.label
        seq = row.sequence
        emb = embedder(seq)
        
        protein_ids.append(protein_id)
        embeddings.append(emb)

    # Save train embeddings to pt
    embeddings_tensor = torch.stack(embeddings, dim=0)
    save_path = os.path.join("embeddings", f"DMS_OLD_{args.pooling}_{'finetuned' if args.finetuned_ckpt else 'base'}_embeddings-{model_version}.pt")
    torch.save({"protein_id": protein_ids, "embeddings": embeddings_tensor}, save_path)

    print(f"Saving embeddings to {save_path}")
    print(f"Saved {len(protein_ids)} protein embeddings with shape {embeddings_tensor.shape}")
    