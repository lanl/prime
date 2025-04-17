#!/usr/bin/env python
"""
PyTorch Lightning ESM-GCN BE model runner.
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

from torch_geometric.data import Data, Batch

from pnlp.ESM_TL.dms_models import GraphSAGE_BE
from pnlp.ESM_TL.dms_data_module import DmsBeDataModule  
from pnlp.ESM_TL.dms_plotter import LossBeFigureCallback

class LightningEsmGcn(L.LightningModule):
    def __init__(self, 
                 from_checkpoint:str, # Only set for hparams save
                 lr: float, max_len: int, gcn_model: GraphSAGE_BE, esm_version="facebook/esm2_t6_8M_UR50D", freeze_esm_weights=True, from_esm_mlm=None):
        super().__init__()
        self.save_hyperparameters(ignore=["gcn_model"])  # Save all init parameters to self.hparams
        self.tokenizer = EsmTokenizer.from_pretrained(esm_version, cache_dir="../../../.cache")
        self.esm = EsmModel.from_pretrained(esm_version, cache_dir="../../../.cache")
        self.gcn = gcn_model
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
            print(missing)

            # Define keys to ignore in missing list, these are from ESM_GCN BE and won't exist in the ESM_MLM
            ignored_missing = {
                "esm.pooler.dense.weight", "esm.pooler.dense.bias",
                "gcn.conv1.lin_l.weight", "gcn.conv1.lin_l.bias", "gcn.conv1.lin_r.weight", "gcn.conv2.lin_l.weight", "gcn.conv2.lin_l.bias", "gcn.conv2.lin_r.weight",
                "gcn.fcn.0.weight", "gcn.fcn.0.bias", "gcn.fcn.2.weight", "gcn.fcn.2.bias", "gcn.fcn.4.weight", 
                "gcn.fcn.4.bias", "gcn.fcn.6.weight", "gcn.fcn.6.bias", "gcn.fcn.8.weight", "gcn.fcn.8.bias",
                "gcn.binding_out.weight", "gcn.binding_out.bias", "gcn.expression_out.weight", "gcn.expression_out.bias"
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

    def forward(self, input_ids, attention_mask, binding_targets, expression_targets):
        esm_last_hidden_state = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # shape: [batch_size, sequence_length, embedding_dim]
        esm_aa_embedding = esm_last_hidden_state[:, 1:-1, :] # Amino Acid-level representations, [batch_size, sequence_length-2, embedding_dim], excludes 1st and last tokens
        
        # Graph Construction
        graphs = []
        for embedding, b_target, e_target in zip(esm_aa_embedding, binding_targets, expression_targets):
            edges = [(i, i+1) for i in range(embedding.size(0) - 1)]
            edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()
            graphs.append(Data(
                x=embedding,
                edge_index=edge_index,
                y=torch.tensor([[b_target, e_target]], dtype=torch.float32)  # Add an extra dimension
            ))
        
        batch_graph = Batch.from_data_list(graphs).to(self.device)
        binding_output, expression_output = self.gcn(batch_graph.x, batch_graph.edge_index, batch_graph.batch)

        return binding_output, expression_output, batch_graph.y
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def step(self, batch):
        _, seqs, binding_targets, expression_targets = batch
        binding_targets = binding_targets.to(self.device).float()
        expression_targets = expression_targets.to(self.device).float()
        batch_size = binding_targets.size(0)

        tokenized = self.tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        binding_preds, expression_preds, y = self(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"], binding_targets=binding_targets, expression_targets=expression_targets)
        binding_loss = self.loss_fn(binding_preds, binding_targets)  # Sum of squared errors (sse)
        expression_loss = self.loss_fn(expression_preds, expression_targets)  # Sum of squared errors (sse)
        be_loss = binding_loss + expression_loss

        return be_loss, binding_loss, expression_loss, batch_size

    def training_step(self, batch, batch_idx):
        be_loss, binding_loss, expression_loss, batch_size = self.step(batch)
        
        # Accumulate (reduce_fx="sum") losses and total items in batch
        self.log("train_binding_sum_se", binding_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("train_expression_sum_se", expression_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("train_be_sum_se", be_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("train_total_items", batch_size, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        return be_loss 
    
    def on_train_epoch_end(self):
        binding_sum_se = self.trainer.callback_metrics.get("train_binding_sum_se")
        expression_sum_se = self.trainer.callback_metrics.get("train_expression_sum_se")
        be_sum_se = self.trainer.callback_metrics.get("train_be_sum_se")
        total_items = self.trainer.callback_metrics.get("train_total_items")
        
        if all(i is not None for i in [binding_sum_se, expression_sum_se, be_sum_se]) and total_items and total_items > 0:
            train_binding_mse = binding_sum_se / total_items
            train_binding_rmse = torch.sqrt(train_binding_mse)
            train_expression_mse = expression_sum_se / total_items
            train_expression_rmse = torch.sqrt(train_expression_mse)
            train_be_mse = be_sum_se / total_items
            train_be_rmse = torch.sqrt(train_be_mse)
            self.log("train_binding_mse", train_binding_mse, prog_bar=True) # Already synced in step
            self.log("train_binding_rmse", train_binding_rmse, prog_bar=True) # Already synced in step
            self.log("train_expression_mse", train_expression_mse, prog_bar=True) # Already synced in step
            self.log("train_expression_rmse", train_expression_rmse, prog_bar=True) # Already synced in step
            self.log("train_be_mse", train_be_mse, prog_bar=True) # Already synced in step
            self.log("train_be_rmse", train_be_rmse, prog_bar=True) # Already synced in step

    def validation_step(self, batch, batch_idx):
        be_loss, binding_loss, expression_loss, batch_size = self.step(batch)
        
        # Accumulate (reduce_fx="sum") losses and total items in batch
        self.log("val_binding_sum_se", binding_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("val_expression_sum_se", expression_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("val_be_sum_se", be_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("val_total_items", batch_size, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        return be_loss 

    def on_validation_epoch_end(self):
        binding_sum_se = self.trainer.callback_metrics.get("val_binding_sum_se")
        expression_sum_se = self.trainer.callback_metrics.get("val_expression_sum_se")
        be_sum_se = self.trainer.callback_metrics.get("val_be_sum_se")
        total_items = self.trainer.callback_metrics.get("val_total_items")
        
        if all(i is not None for i in [binding_sum_se, expression_sum_se, be_sum_se]) and total_items and total_items > 0:
            val_binding_mse = binding_sum_se / total_items
            val_binding_rmse = torch.sqrt(val_binding_mse)
            val_expression_mse = expression_sum_se / total_items
            val_expression_rmse = torch.sqrt(val_expression_mse)
            val_be_mse = be_sum_se / total_items
            val_be_rmse = torch.sqrt(val_be_mse)
            self.log("val_binding_mse", val_binding_mse, prog_bar=True) # Already synced in step
            self.log("val_binding_rmse", val_binding_rmse, prog_bar=True) # Already synced in step
            self.log("val_expression_mse", val_expression_mse, prog_bar=True) # Already synced in step
            self.log("val_expression_rmse", val_expression_rmse, prog_bar=True) # Already synced in step
            self.log("val_be_mse", val_be_mse, prog_bar=True) # Already synced in step
            self.log("val_be_rmse", val_be_rmse, prog_bar=True) # Already synced in step


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run ESM-GCN BE model with Lightning")
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
    logger = CSVLogger(save_dir="logs", name=f"esm_gcn_be", version=f"version_{slurm_job_id}" if slurm_job_id is not None else None)

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
            LossBeFigureCallback(),               # For loss plots
        ]
    )

    # Manually set the checkpoint directory after Trainer initialization
    ckpt_dir = os.path.join(trainer.logger.log_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)  # Ensure directory exists
    best_model_checkpoint.dirpath = ckpt_dir
    all_epochs_checkpoint.dirpath = os.path.join(ckpt_dir, "all_epochs")

    # Data directory (no results_dir needed since versioning handles it automatically)
    data_dir= os.path.join(os.path.dirname(__file__), f"../../../data/dms")

    # GraphSAGE input, size should match used ESM model embedding_dim size
    size = 320
    input_channels = size   
    hidden_channels = size
    fcn_num_layers = 5
    gcn = GraphSAGE_BE(input_channels, hidden_channels, fcn_num_layers)

    # Initialize DataModule and model
    binding_or_expression = args.binding_or_expression
    from_checkpoint = args.from_checkpoint
    from_esm_mlm = args.from_esm_mlm

    if from_checkpoint is not None and from_esm_mlm is not None:
        if trainer.global_rank == 0: print(f"NOTICE: 'from_checkpoint' is set, so 'from_esm_mlm' ({from_esm_mlm}) will be ignored.")
        from_esm_mlm = None

    dm = DmsBeDataModule(
        data_dir=data_dir,
        binding_or_expression=binding_or_expression,  
        torch_geometric_tag=True, 
        batch_size=64,
        num_workers=4, 
        seed=seed
    )

    model = LightningEsmGcn(
        binding_or_expression=binding_or_expression,                  
        from_checkpoint=from_checkpoint,    
        lr=args.lr,
        max_len=280,
        gcn_model=gcn,
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