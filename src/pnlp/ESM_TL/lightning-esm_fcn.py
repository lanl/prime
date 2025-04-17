#!/usr/bin/env python
"""
PyTorch Lightning ESM-FCN model runner.
"""
import argparse
import os
import time
import datetime
import torch
from torch import nn

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from transformers import EsmTokenizer, EsmModel

from pnlp.ESM_TL.dms_models import FCN
from pnlp.ESM_TL.dms_data_module import DmsDataModule  
from pnlp.ESM_TL.dms_plotter import LossFigureCallback

class LightningEsmFcn(L.LightningModule):
    def __init__(self, 
                 binding_or_expression:str, from_checkpoint:str, # Only set for hparams save
                 lr: float, max_len: int, fcn_model: FCN, esm_version="facebook/esm2_t6_8M_UR50D", freeze_esm_weights=True, from_esm_mlm=None):
        super().__init__()
        self.save_hyperparameters(ignore=["fcn_model"])  # Save all init parameters to self.hparams
        self.tokenizer = EsmTokenizer.from_pretrained(esm_version, cache_dir="../../../.cache")
        self.esm = EsmModel.from_pretrained(esm_version, cache_dir="../../../.cache")
        self.fcn = fcn_model
        self.loss_fn = nn.MSELoss(reduction="sum")
        self.lr = lr
        self.max_len = max_len

        # Load fine-tuned weights from Lightning ESM_MLM ckpt
        if from_esm_mlm is not None:
            if trainer.global_rank == 0: print(f"Loading ESM_MLM checkpoint from {from_esm_mlm}...")

            ckpt = torch.load(from_esm_mlm, map_location="cpu")
            state_dict = ckpt["state_dict"]

            # Remove "model." prefix and filter out EsmMaskedLM specific keys
            new_state_dict = {}
            for key, value in state_dict.items():
                # Remove "model." prefix
                new_key = key.replace("model.", "")

                # Filter out EsmMaskedLM keys (e.g., lm_head.*)
                if "lm_head" not in new_key:
                    new_state_dict[new_key] = value

            # Load weights non-strictly
            missing, unexpected = self.load_state_dict(new_state_dict, strict=False)

            # Define keys to ignore in missing list, these are from ESM_FCN and won't exist in the ESM_MLM
            ignored_missing = {
                "esm.pooler.dense.weight", "esm.pooler.dense.bias",
                "fcn.fcn.0.weight", "fcn.fcn.0.bias", "fcn.fcn.2.weight", "fcn.fcn.2.bias",
                "fcn.fcn.4.weight", "fcn.fcn.4.bias", "fcn.fcn.6.weight", "fcn.fcn.6.bias",
                "fcn.fcn.8.weight", "fcn.fcn.8.bias", "fcn.out.weight", "fcn.out.bias"
            }

            # Filter out ignored missing keys
            filtered_missing = [k for k in missing if k not in ignored_missing]

            # Raise error if any unexpected keys are missing
            if filtered_missing: raise RuntimeError(f"Missing unexpected keys from ESM_MLM checkpoint: {filtered_missing}")

            if trainer.global_rank == 0:
                print("ESM_MLM checkpoint loaded successfully.")
                if unexpected: print("Unexpected keys:", unexpected)

        # Freeze ESM weights
        if freeze_esm_weights:
            for param in self.esm.parameters():
                param.requires_grad = False
            if trainer.global_rank == 0: print("ESM weights are frozen!")

    def forward(self, input_ids, attention_mask):
        esm_last_hidden_state = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # shape: [batch_size, sequence_length, embedding_dim]
        esm_cls_embedding = esm_last_hidden_state[:, 0, :]  # CLS token embedding (sequence-level representations), [batch_size, embedding_dim]
        return self.fcn(esm_cls_embedding)  # [batch_size]
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def step(self, batch):
        _, seqs, targets = batch
        targets = targets.to(self.device).float()
        batch_size = targets.size(0)

        tokenized = self.tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        preds = self(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"])
        loss = self.loss_fn(preds, targets)  # Sum of squared errors (sse)

        return loss, batch_size

    def training_step(self, batch, batch_idx):
        loss, batch_size = self.step(batch)
        
        # Accumulate (reduce_fx="sum") losses and total items in batch
        self.log("train_sum_se", loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("train_total_items", batch_size, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        return loss

    def on_train_epoch_end(self):
        sum_se = self.trainer.callback_metrics.get("train_sum_se")
        total_items = self.trainer.callback_metrics.get("train_total_items")
        
        if sum_se is not None and total_items and total_items > 0:
            train_mse = sum_se / total_items
            train_rmse = torch.sqrt(train_mse)
            self.log("train_mse", train_mse, prog_bar=True) # Already synced in step
            self.log("train_rmse", train_rmse, prog_bar=True) # Already synced in step

    def validation_step(self, batch, batch_idx):
        loss, batch_size = self.step(batch)
        
        # Accumulate (reduce_fx="sum") losses and total items in batches
        self.log("val_sum_se", loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("val_total_items", batch_size, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        return loss

    def on_validation_epoch_end(self):
        sum_se = self.trainer.callback_metrics.get("val_sum_se")
        total_items = self.trainer.callback_metrics.get("val_total_items")
        
        if sum_se is not None and total_items and total_items > 0:
            val_mse = sum_se / total_items
            val_rmse = torch.sqrt(val_mse)
            self.log("val_mse", val_mse, prog_bar=True) # Already synced in step
            self.log("val_rmse", val_rmse, prog_bar=True) # Already synced in step

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run ESM-FCN model with Lightning")
    parser.add_argument("--binding_or_expression", type=str, default="binding", help="Set 'binding' or 'expression' as target.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--from_checkpoint", type=str, default=None, help="Path to existing checkpoint to resume training from.")
    parser.add_argument("--from_esm_mlm", type=str, default=None, help="Path to pretrained ESM_MLM checkpoint.")
    parser.add_argument("--freeze_esm", action="store_true", help="Whether to freeze ESM model weights. Abscence of flag sets to False.")
    args = parser.parse_args()

    # Random seed
    seed = 0
    L.seed_everything(seed)  # Set seed for reproducibility

    # Logger 
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    logger = CSVLogger(save_dir="logs", name=f"esm_fcn", version=f"version_{slurm_job_id}" if slurm_job_id is not None else None)

    # Save ONLY the best model in logs/version_x/ckpt
    best_model_checkpoint = ModelCheckpoint(
        filename="best_model-epoch={epoch:02d}.val_rmse={val_rmse:.4f}",
        monitor="val_rmse",
        mode="min",
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
    num_nodes = os.environ.get("SLURM_JOB_NUM_NODES")
    ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")

    num_nodes = int(num_nodes) if num_nodes else 1
    ntasks_per_node = int(ntasks_per_node) if ntasks_per_node else 1

    print(f"Nodes allocated: {num_nodes}, devices allocated per node: {ntasks_per_node}")

    # Trainer setup 
    trainer= L.Trainer(
        max_epochs=args.num_epochs,
        limit_train_batches=1.0,    # 1.0 is 100% of batches
        limit_val_batches=1.0,      # 1.0 is 100% of batches
        strategy=DDPStrategy(find_unused_parameters=True), 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_nodes=num_nodes,
        devices=ntasks_per_node,
        logger=logger,
        callbacks=[
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
    data_dir= os.path.join(os.path.dirname(__file__), f"../../../data/dms")

    # FCN input, size should match used ESM model embedding_dim size
    size = 320
    fcn_input_size = size  
    fcn_hidden_size = size
    fcn_num_layers = 5
    fcn = FCN(fcn_input_size, fcn_hidden_size, fcn_num_layers)

    # Initialize DataModule and model
    binding_or_expression = args.binding_or_expression
    from_checkpoint = args.from_checkpoint
    from_esm_mlm = args.from_esm_mlm

    if from_checkpoint is not None and from_esm_mlm is not None:
        if trainer.global_rank == 0: print(f"NOTICE: 'from_checkpoint' is set, so 'from_esm_mlm' ({from_esm_mlm}) will be ignored.")
        from_esm_mlm = None

    dm = DmsDataModule(
        data_dir=data_dir,
        binding_or_expression=binding_or_expression,  
        torch_geometric_tag=False, 
        batch_size=64,
        num_workers=4, 
        seed=seed
    )

    model = LightningEsmFcn(
        binding_or_expression=binding_or_expression,                  
        from_checkpoint=from_checkpoint,    
        lr=args.lr,
        max_len=280,
        fcn_model=fcn,
        esm_version="facebook/esm2_t6_8M_UR50D",
        freeze_esm_weights=args.freeze_esm,
        from_esm_mlm=from_esm_mlm
    )

    # Run model train/validation, load from_checkpoint if set
    start_time = time.perf_counter()
    if from_checkpoint is not None:
        trainer.fit(model, dm, ckpt_path=from_checkpoint)  # Train model from checkpoint
    else:
        trainer.fit(model, dm)  # Train model
    duration = datetime.timedelta(seconds=time.perf_counter()-start_time)
    if trainer.global_rank == 0: print(f"[Timing] Trainer.fit(...) took: {duration} (hh:mm:ss).")