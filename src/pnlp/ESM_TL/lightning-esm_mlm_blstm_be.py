#!/usr/bin/env python
"""
PyTorch Lightning ESM-MLM-BLSTM BE (with masked language head) model runner.
""" 
import argparse
import os
import time
import datetime
import torch
from torch import nn
from collections import Counter

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from transformers import EsmTokenizer, EsmForMaskedLM

from pnlp.ESM_TL.dms_models import BLSTM_BE
from pnlp.ESM_TL.dms_data_module import DmsBeDataModule  
from pnlp.ESM_TL.dms_plotter import LossBeFigureCallback
from pnlp.ESM_MLM.rbd_plotter import AccuracyLossFigureCallback, AAHeatmapFigureCallback

class LightningEsmBlstmBe(L.LightningModule):
    def __init__(self, 
                 from_checkpoint:str, # Only set for hparams save
                 lr: float, max_len: int, mask_prob: float, blstm_model: BLSTM_BE, esm_version="facebook/esm2_t6_8M_UR50D", freeze_esm_weights=False, from_esm_mlm=None):        
        super().__init__()
        self.save_hyperparameters(ignore=["blstm_model"])  # Save all init parameters to self.hparams
        self.tokenizer = EsmTokenizer.from_pretrained(esm_version, cache_dir="../../../.cache")
        self.esm = EsmForMaskedLM.from_pretrained(esm_version, cache_dir="../../../.cache")
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        self.blstm = blstm_model
        self.regression_loss_fn = nn.MSELoss(reduction="sum")
        self.lr = lr
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.validation_step_aa_preds = []        

        # Load fine-tuned weights from Lightning ESM_MLM ckpt
        if from_esm_mlm is not None:
            if trainer.global_rank == 0: print(f"Loading ESM_MLM checkpoint from {from_esm_mlm}...")

            ckpt = torch.load(from_esm_mlm, map_location="cpu")
            state_dict = ckpt["state_dict"]

            # Convert keys from ckpt state dict
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model.esm."):
                    new_key = key.replace("model.esm.", "esm.esm.")
                    new_state_dict[new_key] = value
                elif key.startswith("model.lm_head."):
                    new_key = key.replace("model.lm_head.", "esm.lm_head.")
                    new_state_dict[new_key] = value

              # Load weights non-strictly
            missing, unexpected = self.load_state_dict(new_state_dict, strict=False)

            # Define keys to ignore in missing list
            ignored_missing = {
                # Regression head (not present in MLM)
                "blstm.lstm.weight_ih_l0", "blstm.lstm.weight_hh_l0", "blstm.lstm.bias_ih_l0", "blstm.lstm.bias_hh_l0",
                "blstm.lstm.weight_ih_l0_reverse", "blstm.lstm.weight_hh_l0_reverse", "blstm.lstm.bias_ih_l0_reverse", "blstm.lstm.bias_hh_l0_reverse",
                "blstm.fcn.0.weight", "blstm.fcn.0.bias", "blstm.fcn.2.weight", "blstm.fcn.2.bias", "blstm.fcn.4.weight", 
                "blstm.fcn.4.bias", "blstm.fcn.6.weight", "blstm.fcn.6.bias", "blstm.fcn.8.weight", "blstm.fcn.8.bias", 
                "blstm.binding_out.weight", "blstm.binding_out.bias", "blstm.expression_out.weight", "blstm.expression_out.bias"
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
        esm_output = self.esm(input_ids=input_ids, attention_mask=attention_mask, labels=None, output_hidden_states=True)
        esm_last_hidden_state = esm_output.hidden_states[-1] # shape: [batch_size, sequence_length, embedding_dim]
        esm_aa_embedding = esm_last_hidden_state[:, 1:-1, :] # Amino Acid-level representations, [batch_size, sequence_length-2, embedding_dim], excludes 1st and last tokens
        binding_preds, expression_preds = self.blstm(esm_aa_embedding) # [batch_size], [batch_size]
        return esm_output, binding_preds, expression_preds
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def step(self, batch):
        _, seqs, binding_targets, expression_targets = batch
        binding_targets = binding_targets.to(self.device).float()
        expression_targets = expression_targets.to(self.device).float()
        batch_size = binding_targets.size(0)

        # Tokenize sequences
        tokenized = self.tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        original_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Generate new mask for each epoch (ignore special tokens)
        rand = torch.rand(original_ids.shape, device=self.device)
        mask_arr = (rand < self.mask_prob) * \
               (original_ids != self.tokenizer.cls_token_id) * \
               (original_ids != self.tokenizer.eos_token_id) * \
               (original_ids != self.tokenizer.pad_token_id)
    
        masked_original_ids = original_ids.clone()
        masked_original_ids[mask_arr] = self.tokenizer.mask_token_id

        # Forward pass, calculate loss
        mlm_output, binding_regression_preds, expression_regression_preds = self(input_ids=masked_original_ids, attention_mask=attention_mask)
        mlm_preds = mlm_output.logits   # [batch_size, sequence_length, vocab_size]

        mlm_labels = original_ids.clone()
        mlm_labels[~mask_arr] = -100  # Ignore everything that's not masked

        mlm_loss = self.mlm_loss_fn(mlm_preds.view(-1, mlm_preds.size(-1)), mlm_labels.view(-1))
        binding_regression_loss = self.regression_loss_fn(binding_regression_preds, binding_targets)  # Sum of squared errors (sse)
        expression_regression_loss = self.regression_loss_fn(expression_regression_preds, expression_targets)  # Sum of squared errors (sse)
        be_regression_loss = binding_regression_loss + expression_regression_loss
        #print(f"mlm loss: {mlm_loss * 0.1}, regression loss: {be_regression_loss}")
        loss = (mlm_loss * 0.1) + be_regression_loss

        # Make sure calculating only on amino acids present at masked positions, no special tokens
        predicted_ids = torch.argmax(mlm_preds, dim=-1)
        mask = (mlm_labels != -100)        

        original_tokens = original_ids[mask]
        predicted_tokens = predicted_ids[mask]

        aa_ids_tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"], device=self.device)
        is_aa_only = torch.isin(original_tokens, aa_ids_tensor) & torch.isin(predicted_tokens, aa_ids_tensor)
        aa_only_original = original_tokens[is_aa_only]
        aa_only_predicted = predicted_tokens[is_aa_only]

        # Calculate mlm accuracy 
        correct = (aa_only_original == aa_only_predicted).sum().item()
        total = is_aa_only.sum().item()
        mlm_accuracy = (correct / total) * 100 if total > 0 else 0.0
       
        return batch_size, loss, be_regression_loss, binding_regression_loss, expression_regression_loss, mlm_loss, aa_only_original, aa_only_predicted, mlm_accuracy

    def training_step(self, batch, batch_idx):
        batch_size, loss, be_regression_loss, binding_regression_loss, expression_regression_loss, mlm_loss, _, _, mlm_accuracy = self.step(batch)

        # Accumulate (reduce_fx="sum") losses and total items in batch for regression calculations
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train_mlm_loss", mlm_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train_mlm_accuracy", mlm_accuracy, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train_binding_sum_se", binding_regression_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("train_expression_sum_se", expression_regression_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("train_be_sum_se", be_regression_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("train_total_items", batch_size, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        return loss
    
    def on_train_epoch_end(self):
        # RMSE calculation
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
        batch_size, loss, be_regression_loss, binding_regression_loss, expression_regression_loss, mlm_loss, aa_only_original, aa_only_predicted, mlm_accuracy = self.step(batch)

        # Track amino acid predictions
        aa_keys = [
            f"{self.tokenizer.convert_ids_to_tokens(o)}->{self.tokenizer.convert_ids_to_tokens(p)}"
            for o, p in zip(aa_only_original.tolist(), aa_only_predicted.tolist())
        ]
        self.validation_step_aa_preds.extend(aa_keys)

        # Accumulate (reduce_fx="sum") losses and total items in batch for regression calculations
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val_mlm_loss", mlm_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val_mlm_accuracy", mlm_accuracy, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val_binding_sum_se", binding_regression_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("val_expression_sum_se", expression_regression_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("val_be_sum_se", be_regression_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("val_total_items", batch_size, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        return loss

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

        # RMSE calculation
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

    parser = argparse.ArgumentParser(description="Run ESM-MLM-BLSTM BE model with Lightning")
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
    logger = CSVLogger(save_dir="logs", name=f"esm_mlm_blstm_be", version=f"version_{slurm_job_id}" if slurm_job_id is not None else None)

    # Save ONLY the best BE model in logs/version_x/ckpt
    best_be_model_checkpoint = ModelCheckpoint(
        filename="best_model-epoch={epoch:02d}.val_be_rmse={val_be_rmse:.4f}.val_binding_rmse={val_binding_rmse:.4f}.val_expression_rmse={val_expression_rmse:.4f}.val_mlm_accuracy={val_mlm_accuracy:.4f}",
        monitor="val_be_rmse", # Our focus is on the regression task, so RMSE priority
        mode="min",
        save_top_k=1,
        save_last=False,
        dirpath=None,  # Let PyTorch Lightning manage the directory
        auto_insert_metric_name=False,
    )

    # Save ONLY the best binding model in logs/version_x/ckpt
    best_binding_model_checkpoint = ModelCheckpoint(
        filename="best_model-epoch={epoch:02d}.val_binding_rmse={val_binding_rmse:.4f}.val_mlm_accuracy={val_mlm_accuracy:.4f}",
        monitor="val_binding_rmse",
        mode="min",
        save_top_k=1,
        save_last=False,
        dirpath=None,  # Let PyTorch Lightning manage the directory
        auto_insert_metric_name=False,
    )

    # Save ONLY the best expression model in logs/version_x/ckpt
    best_expression_model_checkpoint = ModelCheckpoint(
        filename="best_model-epoch={epoch:02d}.val_expression_rmse={val_expression_rmse:.4f}.val_mlm_accuracy={val_mlm_accuracy:.4f}",
        monitor="val_expression_rmse",
        mode="min",
        save_top_k=1,
        save_last=False,
        dirpath=None,  # Let PyTorch Lightning manage the directory
        auto_insert_metric_name=False,
    )

    # Save EVERY n epoch in logs/version_x/ckpt/all_epochs
    all_epochs_checkpoint = ModelCheckpoint(
        filename="{epoch:02d}",
        every_n_epochs=100,
        save_top_k=-1,
        save_last=True,
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
            best_be_model_checkpoint, 
            best_binding_model_checkpoint,
            best_expression_model_checkpoint,
            all_epochs_checkpoint, 
            TQDMProgressBar(refresh_rate=25),   # Update every 25 batches
            LossBeFigureCallback(),             # For loss plots
            AAHeatmapFigureCallback()           # For final/best AA heatmap
        ]
    )

    # Manually set the checkpoint directory after Trainer initialization
    ckpt_dir = os.path.join(trainer.logger.log_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)  # Ensure directory exists
    best_be_model_checkpoint.dirpath = ckpt_dir
    best_binding_model_checkpoint.dirpath = ckpt_dir
    best_expression_model_checkpoint.dirpath = ckpt_dir
    all_epochs_checkpoint.dirpath = os.path.join(ckpt_dir, "all_epochs")

    # Data directory (no results_dir needed since versioning handles it automatically)
    data_dir= os.path.join(os.path.dirname(__file__), f"../../../data/dms")

    # BLSTM input, size should match used ESM model embedding_dim size
    size = 320
    lstm_input_size = size
    lstm_hidden_size = size
    lstm_num_layers = 1        
    lstm_bidrectional = True   
    fcn_hidden_size = size
    fcn_num_layers = 5
    blstm = BLSTM_BE(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_bidrectional, fcn_hidden_size, fcn_num_layers)

    # Initialize DataModule and model
    from_checkpoint = args.from_checkpoint
    from_esm_mlm = args.from_esm_mlm

    if from_checkpoint is not None and from_esm_mlm is not None:
        if trainer.global_rank == 0: print(f"NOTICE: 'from_checkpoint' is set, so 'from_esm_mlm' ({from_esm_mlm}) will be ignored.")
        from_esm_mlm = None

    dm = DmsBeDataModule(
        data_dir=data_dir,
        torch_geometric_tag=False, 
        batch_size=64,
        num_workers=4, 
        seed=seed
    )

    model = LightningEsmBlstmBe(
        from_checkpoint=from_checkpoint,    
        lr=args.lr,
        max_len=280,
        mask_prob=0.15,
        blstm_model=blstm,
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