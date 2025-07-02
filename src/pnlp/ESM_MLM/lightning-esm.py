#!/usr/bin/env python
"""
PyTorch Lightning ESM-MLM (with masked language head) model runner.
"""
import os
import time
import datetime
import torch
from collections import Counter

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from transformers import EsmTokenizer, EsmForMaskedLM

from pnlp.ESM_MLM.rbd_data_module import RbdDataModule  
from pnlp.ESM_MLM.rbd_plotter import AccuracyLossFigureCallback, AAHeatmapFigureCallback

class LightningProteinESM(L.LightningModule):
    def __init__(self, 
                 from_checkpoint:str,   # Only set for hparams save
                 lr: float, max_len: int, mask_prob: float, esm_version="facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        self.save_hyperparameters()  # Save all init parameters to self.hparams
        self.tokenizer = EsmTokenizer.from_pretrained(esm_version, cache_dir="../../../.cache")
        self.model = EsmForMaskedLM.from_pretrained(esm_version, cache_dir="../../../.cache")
        self.lr = lr
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.validation_step_aa_preds = []

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
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

        # Forward pass, calculate loss
        outputs = self(input_ids=masked_original_ids, attention_mask=attention_mask, labels=original_ids)
        loss = outputs.loss
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
        
        return batch_size, loss, aa_only_original, aa_only_predicted, accuracy
                
    def training_step(self, batch, batch_idx):
        batch_size, loss, _, _, accuracy = self.step(batch)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_size, loss, aa_only_original, aa_only_predicted, accuracy = self.step(batch)

        # Track amino acid predictions
        aa_keys = [
            f"{self.tokenizer.convert_ids_to_tokens(o)}->{self.tokenizer.convert_ids_to_tokens(p)}"
            for o, p in zip(aa_only_original.tolist(), aa_only_predicted.tolist())
        ]
        self.validation_step_aa_preds.extend(aa_keys)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

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

if __name__ == "__main__":

    # Random seed
    seed = 0
    L.seed_everything(seed)  # Set seed for reproducibility

    # Logger 
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    logger = CSVLogger(save_dir="logs", name=None, version=f"version_{slurm_job_id}" if slurm_job_id is not None else None)

    # Save ONLY the best model in logs/version_x/ckpt
    best_model_checkpoint = ModelCheckpoint(
        filename="best_model-epoch={epoch:02d}.val_loss={val_loss:.4f}.val_accuracy={val_accuracy:.4f}",
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

    # Get correct number of nodes/devices from slurm
    num_nodes = os.environ.get("SLURM_JOB_NUM_NODES")
    ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")

    num_nodes = int(num_nodes) if num_nodes else 1
    ntasks_per_node = int(ntasks_per_node) if ntasks_per_node else 1

    print(f"Nodes allocated: {num_nodes}, devices allocated per node: {ntasks_per_node}")

    # Trainer setup 
    trainer= L.Trainer(
        max_epochs=100,
        limit_train_batches=1.0,    # 1.0 is 100% of batches
        limit_val_batches=1.0,      # 1.0 is 100% of batches
        strategy=DDPStrategy(find_unused_parameters=True), 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_nodes=num_nodes,
        devices=ntasks_per_node,
        logger=logger,
        callbacks=[
            best_model_checkpoint, 
            #all_epochs_checkpoint, 
            TQDMProgressBar(refresh_rate=25),   # Update every 25 batches
            AccuracyLossFigureCallback(),       # For accuracy/loss plots
            AAHeatmapFigureCallback()           # For final/best AA heatmap
        ]
    )

    # Manually set the checkpoint directory after Trainer initialization
    ckpt_dir = os.path.join(trainer.logger.log_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)  # Ensure directory exists
    best_model_checkpoint.dirpath = ckpt_dir
    all_epochs_checkpoint.dirpath = os.path.join(ckpt_dir, "all_epochs")

    # Data directory (no results_dir needed since versioning handles it automatically)
    data_dir= os.path.join(os.path.dirname(__file__), f"../../../data/rbd")

    # Initialize DataModule and model
    from_checkpoint = None

    dm = RbdDataModule(
        data_dir=data_dir,
        batch_size=64,
        num_workers=4, 
        seed=seed
    )

    model = LightningProteinESM(
        from_checkpoint=from_checkpoint,    
        lr=1e-5,
        max_len=280,
        mask_prob=0.15,
        esm_version="facebook/esm2_t6_8M_UR50D"
    )

    # Run model train/validation, load from_checkpoint if set
    start_time = time.perf_counter()
    if from_checkpoint is not None:
        trainer.fit(model, dm, ckpt_path=from_checkpoint)  # Train model from checkpoint
    else:
        trainer.fit(model, dm)  # Train model
    duration = datetime.timedelta(seconds=time.perf_counter()-start_time)
    if trainer.global_rank == 0: print(f"[Timing] Trainer.fit(...) took: {duration} (hh:mm:ss).")