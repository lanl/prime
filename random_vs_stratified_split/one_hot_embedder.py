#!/usr/bin/env python
"""
One-Hot embedder.
"""
import os
import torch
import argparse
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
        
class OneHotEmbedder(torch.nn.Module):
    def __init__(self, alphabet="ACDEFGHIKLMNPQRSTVWY", pooling="mean", max_length=None):
        super().__init__()
        self.alphabet = alphabet
        self.aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
        self.embedding_dim = len(alphabet)
        self.pooling = pooling
        self.max_length = max_length
    
    def tokenize(self, seq):
        indices = [self.aa_to_idx[aa] for aa in seq if aa in self.aa_to_idx]
        if len(indices) == 0:
            raise ValueError("Sequence contains no valid canonical amino acids.")
        
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, seq):
        # Tokenize sequence
        tokenized_seq = self.tokenize(seq)

        # Apply one-hot
        one_hot = torch.nn.functional.one_hot(tokenized_seq, num_classes=self.embedding_dim).float()  

        if self.pooling == "mean":
            embedding = one_hot.mean(dim=0) # shape: [embedding_dim], where embedding_dim == alphabet_len 

        elif self.pooling == "per-position":
            if self.max_length is None:
                raise ValueError("max_length must be set when pooling='per-position'.")

            seq_len = one_hot.shape[0]

            if seq_len > self.max_length:
                one_hot = one_hot[:self.max_length]  # truncate
                seq_len = self.max_length

            padded = torch.zeros(self.max_length, self.embedding_dim, dtype=one_hot.dtype)
            padded[:seq_len] = one_hot
            embedding = padded  # shape: [max_length, embedding_dim]

        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")
        return embedding.to("cpu")

def load_fasta_sequences(fasta_file):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        protein_id = record.id.split("|")[1] if "train" in fasta_file else record.id  # Extract UniProt ID
        sequences[protein_id] = str(record.seq)

    print(f"Number of sequences in {fasta_file}:", len(sequences))
    return sequences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one-hot Embedder")
    parser.add_argument("--alphabet", type=str, default="ACDEFGHIKLMNPQRSTVWY", 
                        help="Alphabet used to map sequence characters to one-hot vectors (default: %(default)s)")
    parser.add_argument("--pooling", type=str, default="mean", choices=list(["mean", "per-position"]), 
                    help="Pooling method for the embeddings (default: %(default)s). Available: %(choices)s")
    args = parser.parse_args() 

    # Load data
    data_dir = "../../../../data/dms"
    original_set = pd.read_csv(os.path.join(data_dir, "mutation_combined_DMS_OLD.csv"))
    max_length = original_set["sequence"].str.len().max()
    print(f"Max sequence length in dataset: {max_length}")

    # Embedder setup
    embedder = OneHotEmbedder(args.alphabet, args.pooling, max_length)

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
    save_path = os.path.join("embeddings", f"DMS_OLD_{args.pooling}_embeddings-one_hot.pt")
    torch.save({"protein_id": protein_ids, "embeddings": embeddings_tensor}, save_path)

    print(f"Saving embeddings to {save_path}")
    print(f"Saved {len(protein_ids)} protein embeddings with shape {embeddings_tensor.shape}")
    