"""
Data module for ESM-initialized BERT-MLM model runner, converted to PyTorch Lightning.
"""
import os
import sys
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

# DATA
class RBDDataset(Dataset):
    def __init__(self, csv_file:str):
        try:
            self.full_df = pd.read_csv(csv_file, sep=',', header=0)
        except (FileNotFoundError, pd.errors.ParserError, Exception) as e:
            print(f"Error reading in .csv file: {csv_file}\n{e}", file=sys.stderr)
            sys.exit(1)

    def __len__(self) -> int:
        return len(self.full_df)

    def __getitem__(self, idx):
        # label, seq
        return self.full_df['seq_id'][idx], self.full_df['sequence'][idx]

class RBDDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, seed: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Set seeds globally for reproducibility
        pl.seed_everything(self.seed, workers=True)

    def setup(self, stage):
        # Called on every GPU
        if stage == 'fit':
            self.train_dataset = RBDDataset(os.path.join(self.data_dir, "spikeprot0528.clean.uniq.noX.RBD.metadata.variants_train.csv"))
            self.val_dataset = RBDDataset(os.path.join(self.data_dir, "spikeprot0528.clean.uniq.noX.RBD.metadata.variants_test.csv"))

        if stage == 'test':
            self.test_dataset = RBDDataset(os.path.join(self.data_dir, "spikeprot0528.clean.uniq.noX.RBD.metadata.variants_test.csv"))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
            pin_memory=True
        )

def test_data_module():
    # Test case parameters
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data/rbd')
    batch_size = 32
    num_workers = 4

    # Initialize the data module
    data_module = RBDDataModule(data_dir, batch_size, num_workers)

    # == Test case 1: Setup and train dataloader == 
    print("Test case 1: Setup and train dataloader")
    data_module.setup(stage='fit')
    train_loader = data_module.train_dataloader()
    
    # Check if the train loader is not None and has the correct batch size
    assert train_loader is not None, "Train dataloader is None"
    assert train_loader.batch_size == batch_size, f"Expected batch size {batch_size}, got {train_loader.batch_size}"

    # Get a batch from the train loader
    batch = next(iter(train_loader))
    assert len(batch) == 2, f"Expected batch to contain 2 elements, got {len(batch)}"
    seq_ids, sequences = batch
    print(f"Train batch shape: {len(sequences)}")
    print(f"Sample seq_id: {seq_ids[0]}")
    print(f"Sample sequence: {sequences[0]}")

    # == Test case 2: Setup and val dataloader ==
    print("\nTest case 2: Setup and val dataloader")
    data_module.setup(stage='validate')  # Set up for validation stage
    val_loader = data_module.val_dataloader()  # Get the validation dataloader

    # Check if the validation loader is not None and has the correct batch size
    assert val_loader is not None, "Validation dataloader is None"
    assert val_loader.batch_size == batch_size, f"Expected batch size {batch_size}, got {val_loader.batch_size}"

    # Get a batch from the validation loader
    batch = next(iter(val_loader))
    assert len(batch) == 2, f"Expected batch to contain 2 elements, got {len(batch)}"
    seq_ids, sequences = batch
    print(f"Validation batch shape: {len(sequences)}")
    print(f"Sample seq_id: {seq_ids[0]}")
    print(f"Sample sequence: {sequences[0]}")

    # == Test case 3: Setup and test dataloader ==
    print("\nTest case 2: Setup and test dataloader")
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()
    
    # Check if the test loader is not None and has the correct batch size
    assert test_loader is not None, "Test dataloader is None"
    assert test_loader.batch_size == batch_size, f"Expected batch size {batch_size}, got {test_loader.batch_size}"

    # Get a batch from the test loader
    batch = next(iter(test_loader))
    assert len(batch) == 2, f"Expected batch to contain 2 elements, got {len(batch)}"
    seq_ids, sequences = batch
    print(f"Test batch shape: {len(sequences)}")
    print(f"Sample seq_id: {seq_ids[0]}")
    print(f"Sample sequence: {sequences[0]}")

    # == Test case 3: Check reproducibility ==
    print("\nTest case 3: Check reproducibility")
    data_module.setup(stage='fit')
    train_loader1 = data_module.train_dataloader()
    train_loader2 = data_module.train_dataloader()

    batch1 = next(iter(train_loader1))
    batch2 = next(iter(train_loader2))

    assert all(b1 == b2 for b1, b2 in zip(batch1[0], batch2[0])), "Seq IDs from two iterations are not identical. Seeding might not be working correctly."
    assert all(b1 == b2 for b1, b2 in zip(batch1[1], batch2[1])), "Sequences from two iterations are not identical. Seeding might not be working correctly."
    print("Reproducibility check passed: Batches from two iterations are identical.")

    print("\nAll test cases passed successfully!")

    # These should be successful!

if __name__ == '__main__':
    test_data_module()
