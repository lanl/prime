#!/usr/bin/env python
"""
PyTorch Lightning BERT-BLSTM model runner.
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

from pnlp.ESM_TL.dms_models import BLSTM
from pnlp.ESM_TL.dms_data_module import DmsDataModule  
from pnlp.ESM_TL.dms_plotter import LossFigureCallback
from pnlp.ESM_MLM.rbd_plotter import AccuracyLossFigureCallback, AAHeatmapFigureCallback

from pnlp.BERT_MLM.model.language import BERT, ProteinMaskedLanguageModel
from pnlp.BERT_MLM.embedding.tokenizer import ProteinTokenizer, token_to_index

class LightningBertBlstm(L.LightningModule):
    def __init__(self, 
                 binding_or_expression:str, from_checkpoint:str, # Only set for hparams save
                 lr: float, max_len: int, mask_prob:float, blstm_model: BLSTM, 
                 embedding_dim: int, dropout: float, n_transformer_layers: int, n_attn_heads: int, vocab_size: int, freeze_bert_weights=False, from_bert_mlm=None):
        super().__init__()
        self.save_hyperparameters(ignore=["blstm_model"])  # Save all init parameters to self.hparams
        self.tokenizer = ProteinTokenizer(max_len, mask_prob)
        self.token_to_aa = {i:aa for i, aa in enumerate('ACDEFGHIKLMNPQRSTUVWXY')} 
        self.bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
        self.mlm_model = ProteinMaskedLanguageModel(self.bert.hidden, vocab_size=vocab_size)
        self.mlm_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.blstm = blstm_model
        self.regression_loss_fn = nn.MSELoss(reduction="sum")
        self.lr = lr
        self.validation_step_aa_preds = []

        # Load fine-tuned weights from Lightning BERT_MLM ckpt
        if from_bert_mlm is not None:
            if trainer.global_rank == 0: print(f"Loading BERT_MLM checkpoint from {from_bert_mlm}...")

            ckpt = torch.load(from_bert_mlm, map_location="cpu")
            state_dict = ckpt["state_dict"]

            # Convert keys from ckpt state dict
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model.bert."):
                    new_key = key.replace("model.bert.", "bert.")
                    new_state_dict[new_key] = value
                elif key.startswith("model.mlm."):
                    new_key = key.replace("model.mlm.", "mlm_model.")
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
                "blstm.out.weight", "blstm.out.bias"
            }

            # Filter out ignored missing keys
            filtered_missing = [k for k in missing if k not in ignored_missing]

            # Raise error if any unexpected keys are missing
            if filtered_missing: raise RuntimeError(f"Missing unexpected keys from BERT_MLM checkpoint: {filtered_missing}")

            if trainer.global_rank == 0:
                print("BERT_MLM checkpoint loaded successfully.")
                if unexpected: print("Unexpected keys:", unexpected)

        # Freeze BERT MLM weights
        if freeze_bert_weights:
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.mlm_model.parameters():
                param.requires_grad = False
            if trainer.global_rank == 0: print("BERT MLM weights are frozen!")

    def forward(self, masked_tokenized_seqs):
        bert_output = self.bert(masked_tokenized_seqs)
        return self.mlm_model(bert_output), self.blstm(bert_output)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def step(self, batch):
        _, seqs, targets = batch
        targets = targets.to(self.device).float()
        batch_size = targets.size(0)

        # Tokenize sequences
        masked_tokenized_seqs = self.tokenizer(seqs).to(self.device)
        unmasked_tokenized_seqs = self.tokenizer._batch_pad(seqs).to(self.device)

        mask_arr = masked_tokenized_seqs == token_to_index['<MASK>']
        labels = unmasked_tokenized_seqs.clone()
        labels[~mask_arr] = -100    # Ignore everything that's not masked

        # Forward pass, calculate loss
        mlm_preds, regression_preds = self(masked_tokenized_seqs)
        mlm_loss = self.mlm_loss_fn(mlm_preds.transpose(1,2), labels)
        regression_loss = self.regression_loss_fn(regression_preds, targets)  # Sum of squared errors (sse)
        #print(f"mlm loss: {mlm_loss * 0.1}, regression loss: {regression_loss}")
        loss = (mlm_loss * 0.1) + regression_loss

        # Make sure calculating only on amino acids present at masked positions, no special tokens
        predicted_ids = torch.argmax(mlm_preds, dim=-1)
        mask = (labels != -100)       
    
        # Extract values at masked positions
        original_tokens = unmasked_tokenized_seqs[mask]
        predicted_tokens = predicted_ids[mask]

        aa_ids_tensor = torch.tensor([token_to_index.get(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'], device=self.device)
        is_aa_only = torch.isin(original_tokens, aa_ids_tensor) & torch.isin(predicted_tokens, aa_ids_tensor)
        aa_only_original = original_tokens[is_aa_only]
        aa_only_predicted = predicted_tokens[is_aa_only]

        # Calculate mlm accuracy 
        correct = (aa_only_original == aa_only_predicted).sum().item()
        total = is_aa_only.sum().item()
        mlm_accuracy = (correct / total) * 100 if total > 0 else 0.0

        return batch_size, loss, regression_loss, mlm_loss, aa_only_original, aa_only_predicted, mlm_accuracy

    def training_step(self, batch, batch_idx):
        batch_size, loss, regression_loss, mlm_loss, _, _, mlm_accuracy = self.step(batch)
        
        # Accumulate (reduce_fx="sum") losses and total items in batch for regression calculations
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train_mlm_loss", mlm_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train_mlm_accuracy", mlm_accuracy, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train_sum_se", regression_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("train_total_items", batch_size, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        return loss

    def on_train_epoch_end(self):
        # RMSE calculation
        sum_se = self.trainer.callback_metrics.get("train_sum_se")
        total_items = self.trainer.callback_metrics.get("train_total_items")
        
        if sum_se is not None and total_items and total_items > 0:
            train_mse = sum_se / total_items
            train_rmse = torch.sqrt(train_mse)
            self.log("train_mse", train_mse, prog_bar=True) # Already synced in step
            self.log("train_rmse", train_rmse, prog_bar=True) # Already synced in step

    def validation_step(self, batch, batch_idx):
        batch_size, loss, regression_loss, mlm_loss, aa_only_original, aa_only_predicted, mlm_accuracy = self.step(batch)

        # Track amino acid predictions
        aa_keys = [
            f"{self.token_to_aa.get(o)}->{self.token_to_aa.get(p)}"
            for o, p in zip(aa_only_original.tolist(), aa_only_predicted.tolist())
        ]
        self.validation_step_aa_preds.extend(aa_keys)
        
        # Accumulate (reduce_fx="sum") losses and total items in batches
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val_mlm_loss", mlm_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val_mlm_accuracy", mlm_accuracy, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val_sum_se", regression_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
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
        sum_se = self.trainer.callback_metrics.get("val_sum_se")
        total_items = self.trainer.callback_metrics.get("val_total_items")
        
        if sum_se is not None and total_items and total_items > 0:
            val_mse = sum_se / total_items
            val_rmse = torch.sqrt(val_mse)
            self.log("val_mse", val_mse, prog_bar=True) # Already synced in step
            self.log("val_rmse", val_rmse, prog_bar=True) # Already synced in step

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run BERT-BLSTM model with Lightning")
    parser.add_argument("--binding_or_expression", type=str, default="binding", help="Set 'binding' or 'expression' as target.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--from_checkpoint", type=str, default=None, help="Path to existing checkpoint to resume training from.")
    parser.add_argument("--from_bert_mlm", type=str, default=None, help="Path to pretrained BERT_MLM checkpoint.")
    parser.add_argument("--freeze_bert", action="store_true", help="Whether to freeze BERT MLM model weights. Abscence of flag sets to False.")
    args = parser.parse_args()

    # Random seed
    seed = 0
    L.seed_everything(seed)  # Set seed for reproducibility

    # Logger 
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    logger = CSVLogger(save_dir="logs", name=f"bert_blstm", version=f"version_{slurm_job_id}" if slurm_job_id is not None else None)

    # Save ONLY the best model in logs/version_x/ckpt
    best_model_checkpoint = ModelCheckpoint(
        filename="best_model-epoch={epoch:02d}.val_rmse={val_rmse:.4f}.val_mlm_accuracy={val_mlm_accuracy:.4f}", 
        monitor="val_rmse", # Our focus is on the regression task, so RMSE priority
        mode="min",
        save_top_k=1,
        save_last=False,
        dirpath=None,  # Let PyTorch Lightning manage the directory
        auto_insert_metric_name=False,
    )

    # Save EVERY epoch in logs/version_x/ckpt/all_epochs
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
            best_model_checkpoint, 
            all_epochs_checkpoint, 
            TQDMProgressBar(refresh_rate=25),   # Update every 25 batches
            LossFigureCallback(),               # For loss plots
            AAHeatmapFigureCallback()           # For final/best AA heatmap
        ]
    )

    # Manually set the checkpoint directory after Trainer initialization
    ckpt_dir = os.path.join(trainer.logger.log_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)  # Ensure directory exists
    best_model_checkpoint.dirpath = ckpt_dir
    all_epochs_checkpoint.dirpath = os.path.join(ckpt_dir, "all_epochs")

    # Data directory (no results_dir needed since versioning handles it automatically)
    data_dir= os.path.join(os.path.dirname(__file__), f"../../../data/dms")

    # BLSTM input, size should match used BERT model embedding_dim size
    size = 320
    lstm_input_size = size
    lstm_hidden_size = size
    lstm_num_layers = 1        
    lstm_bidrectional = True   
    fcn_hidden_size = size
    fcn_num_layers = 5
    blstm = BLSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_bidrectional, fcn_hidden_size, fcn_num_layers)

    # Initialize DataModule and model
    binding_or_expression = args.binding_or_expression
    from_checkpoint = args.from_checkpoint
    from_bert_mlm = args.from_bert_mlm

    if from_checkpoint is not None and from_bert_mlm is not None:
        if trainer.global_rank == 0: print(f"NOTICE: 'from_checkpoint' is set, so 'from_bert_mlm' ({from_bert_mlm}) will be ignored.")
        from_bert_mlm = None

    dm = DmsDataModule(
        data_dir=data_dir,
        binding_or_expression=binding_or_expression,  
        torch_geometric_tag=False, 
        batch_size=64,
        num_workers=4, 
        seed=seed
    )

    model = LightningBertBlstm(
        binding_or_expression=binding_or_expression,                  
        from_checkpoint=from_checkpoint,    
        lr=args.lr,
        max_len=280,
        mask_prob=0.15,
        blstm_model=blstm,
        embedding_dim=320,
        dropout=0.1,
        n_transformer_layers=12, 
        n_attn_heads=10,
        vocab_size=len(token_to_index),
        freeze_bert_weights=args.freeze_bert,
        from_bert_mlm=from_bert_mlm
    )

    # Run model train/validation, load from_checkpoint if set
    start_time = time.perf_counter()
    if from_checkpoint is not None:
        trainer.fit(model, dm, ckpt_path=from_checkpoint)  # Train model from checkpoint
    else:
        trainer.fit(model, dm)  # Train model
    duration = datetime.timedelta(seconds=time.perf_counter()-start_time)
    if trainer.global_rank == 0: print(f"[Timing] Trainer.fit(...) took: {duration} (hh:mm:ss).")