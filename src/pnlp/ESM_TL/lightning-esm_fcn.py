#!/usr/bin/env python
"""
PyTorch Lightning ESM (with masked language head) model runner.
"""
import os
import time
import datetime
import torch
import numpy as np
from torch import nn
from collections import Counter

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, Callback
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from transformers import EsmTokenizer, EsmModel

from pnlp.ESM_TL.dms_data_module import DMSDataModule  
from pnlp.ESM_TL.dms_plotter import LossFigureCallback

class FCN(L.LightningModule):
    """ Fully Connected Network """

    def __init__(self,
                 fcn_input_size,    # The number of input features
                 fcn_hidden_size,   # The number of features in hidden layer of FCN.
                 fcn_num_layers):   # The number of fcn layers  
        super().__init__()

        # Creating a list of layers for the FCN
        # Subsequent layers after 1st should be equal to hidden_size for input_size
        layers = []
        input_size = fcn_input_size

        for _ in range(fcn_num_layers):
            layers.append(nn.Linear(input_size, fcn_hidden_size))
            layers.append(nn.ReLU())
            input_size = fcn_hidden_size

        # FCN layers
        self.fcn = nn.Sequential(*layers)

        # FCN output layer 
        self.out = nn.Linear(fcn_hidden_size, 1)

    def forward(self, x):
        fcn_out = self.fcn(x)
        prediction = self.out(fcn_out).squeeze(1)  # [batch_size]
        return prediction

class LightningEsmFcn(L.LightningModule):
    def __init__(self, lr: float, max_len: int, fcn_num_layers:int, fcn_size=320, esm_version="facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        self.save_hyperparameters()  # Save all init parameters to self.hparams
        self.tokenizer = EsmTokenizer.from_pretrained(esm_version, cache_dir="../../../.cache")
        self.esm = EsmModel.from_pretrained(esm_version, cache_dir="../../../.cache")
        self.fcn = FCN(fcn_input_size=fcn_size, fcn_hidden_size=fcn_size, fcn_num_layers=fcn_num_layers)
        self.loss_fn = nn.MSELoss(reduction='sum')
        self.lr = lr
        self.max_len = max_len
        self.training_loss = 0.0
        self.training_items = 0
        self.validation_loss = 0.0
        self.validation_items = 0

    def forward(self, input_ids, attention_mask):
        esm_last_hidden_state = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # shape: [batch_size, sequence_length, embedding_dim]
        esm_cls_embedding = esm_last_hidden_state[:, 0, :]  # CLS token embedding (sequence-level representations), [batch_size, embedding_dim]
        return self.fcn(esm_cls_embedding)  # [batch_size]
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def step(self, batch):
        _, seqs, targets = batch
        targets = targets.to(self.device).float()

        # Tokenize sequences
        tokenized_seqs = self.tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        tokenized_seqs = {k: v.to(self.device) for k, v in tokenized_seqs.items()}
        original_ids = tokenized_seqs["input_ids"]
        attention_mask = tokenized_seqs["attention_mask"]

        # Forward pass, calculate loss
        outputs = self(input_ids=original_ids, attention_mask=attention_mask)
        preds = outputs.logits
        loss = self.loss_fn(preds, targets)

        return loss, len(targets)
                
    def training_step(self, batch, batch_idx):
        loss, num_items = self.step(batch)
        self.training_loss += loss.item()
        self.training_items += num_items
        return loss

    def validation_step(self, batch, batch_idx):
        loss, num_items = self.step(batch)
        self.validation_loss += loss.item()
        self.validation_items += num_items
        return loss
    
    def training_epoch_end(self, outputs):
        mse = self.training_loss / self.training_items
        rmse = np.sqrt(mse)
        self.training_loss = 0.0
        self.training_items = 0
        self.log('train_rmse', rmse, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        mse = self.validation_loss / self.validation_items
        rmse = np.sqrt(mse)
        self.validation_loss = 0.0
        self.validation_items = 0
        self.log('val_rmse', rmse, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

class MetricsCallback(Callback):
    def on_train_epoch_end(self, trainer: L.Trainer, lightning_module: L.LightningModule):
        """
        This method is called when a training epoch ends.
        Validation end callbacks are triggered before training end callbacks.
        """    
        if trainer.global_rank == 0:
            train_rmse = trainer.callback_metrics.get("train_rmse")
            val_rmse = trainer.callback_metrics.get("val_rmse")

            print(
                f"\n[Epoch {trainer.current_epoch}] "
                f"Train RMSE Loss: {train_rmse:.4f} | Val RMSE Loss: {val_rmse:.4f}",
                flush=True
            )

if __name__ == '__main__':

    # Random seed
    seed = 0
    L.seed_everything(seed)  # Set seed for reproducibility

    # Logger 
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    logger = CSVLogger(save_dir="logs", name=None, version=f"version_{slurm_job_id}" if slurm_job_id is not None else None)

    # Save ONLY the best model in logs/version_x/ckpt
    best_model_checkpoint = ModelCheckpoint(
        filename="best_model-epoch={epoch:02d}.val_rmse={val_rmse:.4f}",
        monitor="val_rmse",
        mode="max",
        save_top_k=1,
        save_last=False,
        dirpath=None,  # Let PyTorch Lightning manage the directory
        auto_insert_metric_name=False,
    )

    # Save EVERY epoch in logs/version_x/ckpt/all_epochs
    all_epochs_checkpoint = ModelCheckpoint(
        filename="{epoch:02d}",
        every_n_epochs=1,
        save_top_k=-1,
        save_last=False,
        dirpath=None,  # Let PyTorch Lightning manage the directory
    )

    # Get correct number of nodes/devices from slurm
    num_nodes = os.environ.get('SLURM_JOB_NUM_NODES')
    ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")

    num_nodes = int(num_nodes) if num_nodes else 1
    ntasks_per_node = int(ntasks_per_node) if ntasks_per_node else 1

    print(f"Nodes allocated: {num_nodes}, devices allocated per node: {ntasks_per_node}")

    # Trainer setup 
    trainer= L.Trainer(
        max_epochs=5,
        limit_train_batches=0.01,    # 1.0 is 100% of batches
        limit_val_batches=0.01,      # 1.0 is 100% of batches
        #strategy='deepspeed',
        strategy=DDPStrategy(find_unused_parameters=True), 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_nodes=num_nodes,
        devices=ntasks_per_node,
        logger=logger,
        callbacks=[
            MetricsCallback(),                  # For printing metrics after every epoch
            best_model_checkpoint, 
            all_epochs_checkpoint, 
            TQDMProgressBar(refresh_rate=25),   # Update every 25 batches
            LossFigureCallback(),               # For loss plots
        ]
    )

    # Manually set the checkpoint directory after Trainer initialization
    ckpt_dir = os.path.join(trainer.logger.log_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)  # Ensure directory exists
    best_model_checkpoint.dirpath = ckpt_dir
    all_epochs_checkpoint.dirpath = os.path.join(ckpt_dir, "all_epochs")

    # Data directory (no results_dir needed since versioning handles it automatically)
    data_dir= os.path.join(os.path.dirname(__file__), f'../../../data/dms')

    # Initialize DataModule and model
    dm = DMSDataModule(
        data_dir=data_dir,
        batch_size=64,
        num_workers=4, 
        seed=seed
    )

    model = LightningEsmFcn(
        lr=1e-5,
        max_len=280,
        fcn_num_layers=5, 
        fcn_size=320,   # Match embedding dim of ESM model
        esm_version="facebook/esm2_t6_8M_UR50D"
    )

    start_time = time.perf_counter()
    trainer.fit(model, dm)  # Train model
    duration = datetime.timedelta(seconds=time.perf_counter()-start_time)
    print(f"[Timing] Trainer.fit(...) took: {duration} (hh:mm:ss).")