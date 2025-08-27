#!/usr/bin/env python
"""
PyTorch Lightning BERT-MLM (with masked language head) or ESM-MLM model runner.
"""
import argparse
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
 
from pnlp.ESM_MLM.rbd_plotter import AAHeatmapFigureCallback

from pnlp.BERT_MLM.model.language import BERT, ProteinLM
from pnlp.BERT_MLM.embedding.tokenizer import ProteinTokenizer, token_to_index

from transformers import EsmTokenizer, EsmForMaskedLM

class LightningProteinBERT(L.LightningModule):
    def __init__(self, 
                 from_saved: str,
                 max_len: int, mask_prob: float, embedding_dim: int, dropout: float, n_transformer_layers: int, n_attn_heads: int, vocab_size: int):
        super().__init__()
        self.save_hyperparameters()  # Save all init parameters to self.hparams
        self.tokenizer = ProteinTokenizer(max_len, mask_prob)
        self.token_to_aa = {i:aa for i, aa in enumerate('ACDEFGHIKLMNPQRSTUVWXY')} 
        self.bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
        self.model = ProteinLM(self.bert, vocab_size=vocab_size)
        self.validation_step_aa_preds = []

        # Load fine-tuned weights 
        if from_saved is not None:
            print(f"Loading model checkpoint from {from_saved}...")
            ckpt = torch.load(from_saved, map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=True)

        # Freeze weights
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False
        print("Model weights are frozen!")

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return None
    
    def step(self, batch):
        _, seqs = batch
        batch_size = len(seqs)

        # Tokenize sequences
        masked_tokenized_seqs = self.tokenizer(seqs).to(self.device)
        unmasked_tokenized_seqs = self.tokenizer._batch_pad(seqs).to(self.device)

        mask_arr = masked_tokenized_seqs == token_to_index['<MASK>']
        labels = unmasked_tokenized_seqs.clone()
        labels[~mask_arr] = -100    # Ignore everything that's not masked

        preds = self(masked_tokenized_seqs)

        # Make sure calculating only on amino acids present at masked positions, no special tokens
        predicted_ids = torch.argmax(preds, dim=-1)
        mask = (labels != -100)       
    
        original_tokens = unmasked_tokenized_seqs[mask]
        predicted_tokens = predicted_ids[mask]

        aa_ids_tensor = torch.tensor([token_to_index.get(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'], device=self.device)
        is_aa_only = torch.isin(original_tokens, aa_ids_tensor) & torch.isin(predicted_tokens, aa_ids_tensor)
        aa_only_original = original_tokens[is_aa_only]
        aa_only_predicted = predicted_tokens[is_aa_only]

        # Calculate accuracy 
        correct = (aa_only_original == aa_only_predicted).sum().item()
        total = is_aa_only.sum().item()
        accuracy = (correct / total) * 100 if total > 0 else 0.0

        return batch_size, aa_only_original, aa_only_predicted, accuracy
    
    def training_step(self, batch, batch_idx):
        return None

    def validation_step(self, batch, batch_idx):
        batch_size, aa_only_original, aa_only_predicted, accuracy = self.step(batch)

        # Track amino acid predictions 
        aa_keys = [
            f"{self.token_to_aa.get(o)}->{self.token_to_aa.get(p)}"
            for o, p in zip(aa_only_original.tolist(), aa_only_predicted.tolist())
        ]
        self.validation_step_aa_preds.extend(aa_keys)

        # Log metrics
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
    
    def on_validation_epoch_end(self):   
        # Prediction tracking
        aa_preds_counter = Counter(self.validation_step_aa_preds)

        # Create a unique filename for each epoch/rank
        aa_preds_dir = os.path.join(self.logger.log_dir, "aa_preds")
        os.makedirs(aa_preds_dir, exist_ok = True)
        preds_csv_path = os.path.join(aa_preds_dir, f"aa_predictions_epoch{self.current_epoch}_rank{self.global_rank}.csv")

        with open(preds_csv_path, "w") as fb:
            # Only write a header row
            fb.write(f"expected_aa->predicted_aa,count\n") # changed the header row to only include count

            # Write each expected aa->predicted aa and count directly to the csv file.
            for substitution, count in aa_preds_counter.items():
                fb.write(f"{substitution},{count}\n")

        # Clear the stored outputs, as the current epoch counts have already been recorded.
        self.validation_step_aa_preds.clear()

class LightningProteinESM(L.LightningModule):
    def __init__(self, 
                 from_saved: str,
                 max_len: int, mask_prob: float, esm_version="facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        self.save_hyperparameters()  # Save all init parameters to self.hparams
        self.tokenizer = EsmTokenizer.from_pretrained(esm_version, cache_dir="../../.cache")
        self.model = EsmForMaskedLM.from_pretrained(esm_version, cache_dir="../../.cache")
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.validation_step_aa_preds = []

        # Load fine-tuned weights 
        if from_saved is not None:
            print(f"Loading model checkpoint from {from_saved}...")
            ckpt = torch.load(from_saved, map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=True)

        # Freeze weights
        for param in self.model.parameters():
            param.requires_grad = False
        print("Model weights are frozen!")

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def configure_optimizers(self):
        return None
    
    def step(self, batch):
        _, seqs = batch
        batch_size = len(seqs)

        # Tokenize sequences
        tokenized_seqs = self.tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        tokenized_seqs = {k: v.to(self.device) for k, v in tokenized_seqs.items()}
        original_ids = tokenized_seqs["input_ids"]
        attention_mask = tokenized_seqs["attention_mask"]

        # Generate new mask for each epoch
        rand = torch.rand(original_ids.shape, device=self.device)
        mask_arr = (rand < self.mask_prob) * \
               (original_ids != self.tokenizer.cls_token_id) * \
               (original_ids != self.tokenizer.eos_token_id) * \
               (original_ids != self.tokenizer.pad_token_id)
    
        masked_original_ids = original_ids.clone()
        masked_original_ids[mask_arr] = self.tokenizer.mask_token_id

        outputs = self(input_ids=masked_original_ids, attention_mask=attention_mask, labels=original_ids)
        preds = outputs.logits

        # Make sure calculating only on amino acids present at masked positions, no special tokens
        predicted_ids = torch.argmax(preds, dim=-1)
        mask = (masked_original_ids == self.tokenizer.mask_token_id)
        
        original_tokens = original_ids[mask]
        predicted_tokens = predicted_ids[mask]

        aa_ids_tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"], device=self.device)
        is_aa_only = torch.isin(original_tokens, aa_ids_tensor) & torch.isin(predicted_tokens, aa_ids_tensor)
        aa_only_original = original_tokens[is_aa_only]
        aa_only_predicted = predicted_tokens[is_aa_only]

        # Calculate accuracy 
        correct = (aa_only_original == aa_only_predicted).sum().item()
        total = is_aa_only.sum().item()
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        
        return batch_size, aa_only_original, aa_only_predicted, accuracy

    def training_step(self, batch, batch_idx):
        return None
    
    def validation_step(self, batch, batch_idx):
        batch_size, aa_only_original, aa_only_predicted, accuracy = self.step(batch)

        # Track amino acid predictions
        aa_keys = [
            f"{self.tokenizer.convert_ids_to_tokens(o)}->{self.tokenizer.convert_ids_to_tokens(p)}"
            for o, p in zip(aa_only_original.tolist(), aa_only_predicted.tolist())
        ]
        self.validation_step_aa_preds.extend(aa_keys)

        # Log metrics
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
    
    def on_validation_epoch_end(self):   
        # Prediction tracking
        aa_preds_counter = Counter(self.validation_step_aa_preds)

        # Create a unique filename for each epoch/rank
        aa_preds_dir = os.path.join(self.logger.log_dir, "aa_preds")
        os.makedirs(aa_preds_dir, exist_ok = True)
        preds_csv_path = os.path.join(aa_preds_dir, f"aa_predictions_epoch{self.current_epoch}_rank{self.global_rank}.csv")

        with open(preds_csv_path, "w") as fb:
            # Only write a header row
            fb.write(f"expected_aa->predicted_aa,count\n") # changed the header row to only include count

            # Write each expected aa->predicted aa and count directly to the csv file.
            for substitution, count in aa_preds_counter.items():
                fb.write(f"{substitution},{count}\n")

        # Clear the stored outputs, as the current epoch counts have already been recorded.
        self.validation_step_aa_preds.clear()

class DmsDataset(Dataset):
    def __init__(self, csv_file:str):
        """
        Load from csv file into pandas:
        - sequence label ('labels'), 
        - 'sequence'
        """
        try:
            self.full_df = pd.read_csv(csv_file, sep=',', header=0)
        except (FileNotFoundError, pd.errors.ParserError, Exception) as e:
            print(f"Error reading in .csv file: {csv_file}\n{e}", file=sys.stderr)
            sys.exit(1)

    def __len__(self) -> int:
        return len(self.full_df)

    def __getitem__(self, idx):
        # label, seq
        return self.full_df['label'][idx], self.full_df['sequence'][idx]

class DmsDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, seed: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Set seeds globally for reproducibility
        L.seed_everything(self.seed, workers=True)

    def setup(self, stage):
        # Called on every GPU
        if stage == 'fit':
            self.val_dataset = DmsDataset(os.path.join(self.data_dir, "mutation_combined_DMS_OLD.csv"))
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
            pin_memory=True
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run MLM models with Lightning on the DMS dataset.")
    parser.add_argument("--model_version", type=str, default=None, help="ESM or BERT.")
    parser.add_argument("--use_finetuned", action="store_true", help="Use the finetuned model version on RBD dataset or not. Abscence of flag sets to False.")
    args = parser.parse_args()

    # Random seed
    seed = 0
    L.seed_everything(seed)  # Set seed for reproducibility

    # Logger 
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    logger = CSVLogger(save_dir="logs", name=None, version=f"version_{slurm_job_id}-model.{args.model_version}_finetuned.{args.use_finetuned}" if slurm_job_id is not None else None)

    # Get correct number of nodes/devices from slurm
    num_nodes = os.environ.get("SLURM_JOB_NUM_NODES")
    ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")

    num_nodes = int(num_nodes) if num_nodes else 1
    ntasks_per_node = int(ntasks_per_node) if ntasks_per_node else 1

    print(f"Nodes allocated: {num_nodes}, devices allocated per node: {ntasks_per_node}")

    # Trainer setup 
    trainer= L.Trainer(
        max_epochs=1,
        limit_train_batches=0,    # 1.0 is 100% of batches
        limit_val_batches=1.0,      # 1.0 is 100% of batches
        strategy="auto", 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_nodes=num_nodes,
        devices=ntasks_per_node,
        logger=logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=25),   # Update every 25 batches
            AAHeatmapFigureCallback()           # For final/best AA heatmap
        ]
    )

    # Data directory (no results_dir needed since versioning handles it automatically)
    data_dir= os.path.join(os.path.dirname(__file__), f"../../data/dms")

    # Initialize DataModule and model  
    dm = DmsDataModule(
        data_dir=data_dir,
        batch_size=64,
        num_workers=4, 
        seed=seed
    )

    if args.model_version.lower() == "esm":
        if args.use_finetuned:
            from_saved = "../../src/pnlp/ESM_MLM/logs/version_21768307/ckpt/best_model-epoch=73.val_loss=0.0022.val_accuracy=99.6612.ckpt"
        else:
            from_saved = None

        model = LightningProteinESM(
            from_saved=from_saved,    
            max_len=280,
            mask_prob=0.15,
            esm_version="facebook/esm2_t6_8M_UR50D"
        )

    else:
        if args.use_finetuned:
            from_saved = "../../src/pnlp/BERT_MLM/logs/version_22088461/ckpt/best_model-epoch=98.val_loss=0.0423.val_accuracy=99.0325.ckpt"
        else:
            from_saved = None

        model = LightningProteinBERT(
            from_saved=from_saved,    
            max_len=280,
            mask_prob=0.15,
            embedding_dim=320,
            dropout=0.1,
            n_transformer_layers=12, 
            n_attn_heads=10,
            vocab_size=len(token_to_index)
        )

    # Run model validation
    start_time = time.perf_counter()
    trainer.fit(model, dm) # Inference only
    duration = datetime.timedelta(seconds=time.perf_counter()-start_time)
    if trainer.global_rank == 0: print(f"[Timing] Trainer.fit(...) took: {duration} (hh:mm:ss).")
