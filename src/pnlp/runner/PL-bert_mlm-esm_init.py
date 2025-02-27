#!/usr/bin/env python
"""
PyTorch Lightning Model runner for ESM-initialized BERT-MLM model.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from typing import Union
from collections import defaultdict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from pnlp.model.language import BERT, ProteinLM
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index
from pnlp.runner.rbd_data_module import RBDDataModule  
from pnlp.runner.rbd_plotter import SaveFiguresCallback, plot_aa_preds_heatmap

class ScheduledOptimWrapper(_LRScheduler):
    def __init__(self, optimizer, d_model: int, n_warmup_steps: int, last_epoch=-1):
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Calculate learning rate based on current step.
        Warmup schedule with inverse square root decay for learning rates,
        similar to class ScheduledOptim() for BERT-pytorch.
        """
        current_step = max(1, self.last_epoch + 1)
        return [
            (
                np.power(self.d_model, -0.5) *
                min(np.power(current_step, -0.5), current_step * np.power(self.n_warmup_steps, -1.5))
            )
            for _ in self.optimizer.param_groups
        ]

class PlProteinMLM(pl.LightningModule):
    def __init__(self, embedding_dim: int, dropout: float, max_len: int, mask_prob: float,
                 n_transformer_layers: int, n_attn_heads: int, vocab_size: int, lr: float, esm_weights: str):
        super().__init__()
        self.save_hyperparameters()  # Save all init parameters to self.hparams

        self.bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
        self.bert.embedding.load_pretrained_embeddings(esm_weights, no_grad=False)
        self.model = ProteinLM(self.bert, vocab_size=vocab_size)
        self.tokenizer = ProteinTokenizer(max_len, mask_prob)
        self.lr = lr
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.01)

        # Dynamically calculate n_warmup_steps
        train_dataloader = self.trainer.datamodule.train_dataloader()
        dataset_size = len(train_dataloader.dataset)
        batch_size = train_dataloader.batch_size
        steps_per_epoch = dataset_size // batch_size
        n_warmup_steps = steps_per_epoch * 0.1

        # Use ScheduledOptimWrapper for compatibility with Lightning
        scheduler = ScheduledOptimWrapper(optimizer, d_model=self.hparams.embedding_dim, n_warmup_steps=n_warmup_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Updates per step (batch) instead of per epoch, like the BERT-pytorch scheduler
            },
        }
                
    def training_step(self, batch):
        _, seqs = batch
        masked_tokenized_seqs = self.tokenizer(seqs).to(self.device)
        unmasked_tokenized_seqs = self.tokenizer._batch_pad(seqs).to(self.device)

        preds = self(masked_tokenized_seqs)
        ce_loss = F.cross_entropy(preds.transpose(1, 2), unmasked_tokenized_seqs)

        predicted_tokens = torch.max(preds, dim=-1)[1]
        masked_locations = torch.nonzero(torch.eq(masked_tokenized_seqs, token_to_index['<MASK>']), as_tuple=True)
        correct_predictions = torch.eq(predicted_tokens[masked_locations], unmasked_tokenized_seqs[masked_locations]).sum().item()
        total_masked = masked_locations[0].numel()
        accuracy = (correct_predictions / total_masked) * 100

        # Log metrics
        self.log('train_loss', ce_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=len(seqs))
        self.log('train_accuracy', accuracy, prog_bar=False, on_step=False, on_epoch=True, batch_size=len(seqs))

        return ce_loss
    
    def validation_step(self, batch):
        _, seqs = batch
        masked_tokenized_seqs = self.tokenizer(seqs).to(self.device)
        unmasked_tokenized_seqs = self.tokenizer._batch_pad(seqs).to(self.device)

        preds = self(masked_tokenized_seqs)
        ce_loss = F.cross_entropy(preds.transpose(1, 2), unmasked_tokenized_seqs)

        predicted_tokens = torch.max(preds, dim=-1)[1]
        masked_locations = torch.nonzero(torch.eq(masked_tokenized_seqs, token_to_index['<MASK>']), as_tuple=True)
        correct_predictions = torch.eq(predicted_tokens[masked_locations], unmasked_tokenized_seqs[masked_locations]).sum().item()
        total_masked = masked_locations[0].numel()
        accuracy = (correct_predictions / total_masked) * 100

        # Track amino acid predictions
        token_to_aa = {i: aa for i, aa in enumerate('ACDEFGHIKLMNPQRSTUVWXY')}
        aa_keys = [f"{token_to_aa.get(token.item())}->{token_to_aa.get(pred_token.item())}" for token, pred_token in 
                   zip(unmasked_tokenized_seqs[masked_locations], predicted_tokens[masked_locations])]
        
        aa_pred_counter = defaultdict(int)
        
        for aa_key in aa_keys:
            aa_pred_counter[aa_key] += 1
        
        # Save results for later analysis
        self.validation_step_outputs.append(aa_pred_counter)

        # Log metrics
        self.log('val_loss', ce_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=len(seqs))
        self.log('val_accuracy', accuracy, prog_bar=False, on_step=False, on_epoch=True, batch_size=len(seqs))

        return ce_loss

    def test_step(self, batch):
        _, seqs = batch
        masked_tokenized_seqs = self.tokenizer(seqs).to(self.device)
        unmasked_tokenized_seqs = self.tokenizer._batch_pad(seqs).to(self.device)

        preds = self(masked_tokenized_seqs)
        ce_loss = F.cross_entropy(preds.transpose(1, 2), unmasked_tokenized_seqs)

        predicted_tokens = torch.max(preds, dim=-1)[1]
        masked_locations = torch.nonzero(torch.eq(masked_tokenized_seqs, token_to_index['<MASK>']), as_tuple=True)
        correct_predictions = torch.eq(predicted_tokens[masked_locations], unmasked_tokenized_seqs[masked_locations]).sum().item()
        total_masked = masked_locations[0].numel()
        accuracy = (correct_predictions / total_masked) * 100

        # Log metrics
        self.log('test_loss', ce_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=len(seqs))
        self.log('test_accuracy', accuracy, prog_bar=False, on_step=False, on_epoch=True, batch_size=len(seqs))

    def on_validation_epoch_end(self):
        aa_preds_tracker = defaultdict(lambda: defaultdict(int))
        
        for aa_pred_counter in self.validation_step_outputs:
            for key in aa_pred_counter:
                aa_preds_tracker[key][self.current_epoch] += aa_pred_counter[key]

        # Create a unique filename for each epoch
        aa_preds_dir = os.path.join(self.logger.log_dir, "aa_preds")
        os.makedirs(aa_preds_dir, exist_ok = True)

        preds_csv_path = os.path.join(aa_preds_dir, f"val_aa_predictions-epoch={self.current_epoch}.csv")
        preds_img_path = os.path.join(aa_preds_dir, f"val_aa_predictions-epoch={self.current_epoch}.heatmap.pdf")

        with open(preds_csv_path, 'w') as fb:
            # Only write a header row
            fb.write(f"expected_aa->predicted_aa,count\n") # changed the header row to only include count

            for key in aa_preds_tracker:
                # Write each prediction and count directly to the csv file.
                total_count = sum(aa_preds_tracker[key].values())
                fb.write(f"{key},{total_count}\n")

        # Clear the stored outputs, as the current epoch counts have already been recorded.
        self.validation_step_outputs.clear()
        plot_aa_preds_heatmap(preds_csv_path, preds_img_path)

if __name__ == '__main__':

    # Random seed
    seed = 0
    pl.seed_everything(seed)  # Set seed for reproducibility

    # Logger 
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "default")
    logger = CSVLogger(save_dir="logs", name=None, version=f"version_{slurm_job_id}")

    # Save ONLY the best model in logs/version_x/ckpt
    best_model_checkpoint = ModelCheckpoint(
        filename="best_model-epoch={epoch:02d}.val_accuracy={val_accuracy:.4f}.val_loss={val_loss:.4f}",
        monitor="val_accuracy",
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

    # Trainer setup 
    trainer= pl.Trainer(
        max_epochs=10,
        limit_train_batches=0.01,   # 1.0 is 100% of batches
        limit_val_batches=0.01, # 1.0 is 100% of batches
        limit_test_batches=1.0, # 1.0 is 100% of batches
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto", # Use all available GPUs for training
        logger=logger,
        callbacks=[
            best_model_checkpoint, 
            all_epochs_checkpoint, 
            TQDMProgressBar(refresh_rate=25), # Update every 25 batches
            SaveFiguresCallback()
        ],
    )

    # Manually set the checkpoint directory after Trainer initialization
    ckpt_dir = os.path.join(trainer.logger.log_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)  # Ensure directory exists
    best_model_checkpoint.dirpath = ckpt_dir
    all_epochs_checkpoint.dirpath = os.path.join(ckpt_dir, "all_epochs")

    # Data directory (no results_dir needed since versioning handles it automatically)
    data_dir= os.path.join(os.path.dirname(__file__), f'../../../data/rbd')

    # Initialize DataModule and model
    dm = RBDDataModule(
        data_dir = data_dir,
        batch_size = 64,
        num_workers = 4 if trainer.training else 1,  # Use 1 for testing
        seed = seed
    )

    model = PlProteinMLM(
        max_len = 280,
        mask_prob = 0.15,
        embedding_dim = 320,
        n_transformer_layers = 12,
        n_attn_heads = 10,
        vocab_size = len(token_to_index),
        dropout = 0.1,
        lr = 1e-5,
        esm_weights = os.path.join(data_dir, 'esm_weights-embedding_dim320.pth')
    )

    trainer.fit(model, dm)  # Train model

    # Test on the best model
    best_model_path = best_model_checkpoint.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        print(f"Loading best model from checkpoint: {best_model_path}")
        trained_model = PlProteinMLM.load_from_checkpoint(best_model_path)
    else:
        raise FileNotFoundError(
            f"Best model checkpoint not found at {best_model_path}. "
            "Ensure checkpointing is enabled and model has been saved."
        )
    
    test_trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,  # Force test to use only 1 GPU
    )

    test_trainer.test(trained_model, dm)  # Test model