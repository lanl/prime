#!/usr/bin/env python
"""
PyTorch Lightning ESMC-MLM (with masked language) model runner.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import datetime
import argparse
import torch
from collections import Counter

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from pnlp.ESM_MLM.rbd_data_module import RbdDataModule  
from pnlp.ESM_MLM.rbd_plotter import AccuracyLossFigureCallback, AAHeatmapFigureCallback
from pnlp.ESM_TL.random_vs_stratified_split.esmc_util import load_from_cache

class LightningProteinESM(L.LightningModule):
    def __init__(self, 
                 from_checkpoint:str,   # Only set for hparams save
                 lr: float, max_len: int, mask_prob: float, esm_version="esmc_600m_local"):
        super().__init__()
        self.save_hyperparameters()  # Save all init parameters to self.hparams
        self.model = load_from_cache(esm_version, cache_dir="../../../.cache")
        self.tokenizer = self.model.tokenizer
        self.lr = lr
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.validation_step_aa_preds = []

        id2token = {v: k for k, v in self.tokenizer.get_vocab().items()}
        tokens_sorted = [id2token[i] for i in range(len(id2token))]
        aa_ids = [i for i, tok in enumerate(tokens_sorted) if tok in list("ACDEFGHIKLMNPQRSTVWY")]
        self.aa_ids_tensor = torch.tensor(aa_ids)

    def forward(self, masked_ids):
        return self.model.forward(sequence_tokens=masked_ids)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def step(self, batch):
        _, seqs = batch
        batch_size = len(seqs)

        # Tokenize sequences
        tokenized_seqs = self.tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        tokenized_seqs = {k: v.to(self.device) for k, v in tokenized_seqs.items()}
        input_ids = tokenized_seqs["input_ids"]

        # Generate new mask for each epoch
        rand = torch.rand(input_ids.shape, device=self.device)
        mask_arr = (rand < self.mask_prob) * \
               (input_ids != self.tokenizer.cls_token_id) * \
               (input_ids != self.tokenizer.eos_token_id) * \
               (input_ids != self.tokenizer.pad_token_id)
    
        # Copy and replace selected positions with mask token
        masked_ids = input_ids.clone()
        masked_ids[mask_arr] = self.tokenizer.mask_token_id

        # Set labels to -100 for non-masked positions
        labels = input_ids.clone()
        labels[~mask_arr] = -100 

        # Forward pass
        output = self(masked_ids=masked_ids)

        # Calculate loss
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')  
        loss = loss_fn(output.sequence_logits.transpose(1,2), labels)

        # Make sure calculating only on amino acids present at masked positions, no special tokens
        pred_ids = torch.argmax(output.sequence_logits, dim=-1)
        masked_input_ids = input_ids[mask_arr] 
        masked_pred_ids  = pred_ids[mask_arr]  

        # Filter to ensure only the 20 canonical amino acids (for heatmap)
        is_aa_only = torch.isin(masked_input_ids, self.aa_ids_tensor.to(self.device)) & torch.isin(masked_pred_ids, self.aa_ids_tensor.to(self.device))
        aa_only_input = masked_input_ids[is_aa_only]
        aa_only_pred = masked_pred_ids[is_aa_only]

        # Evaluate on masked positions where the true label is a canonical amino acid
        is_truth_label_aa = torch.isin(masked_input_ids, self.aa_ids_tensor.to(self.device))
        truth_label_aa_input = masked_input_ids[is_truth_label_aa]
        truth_label_aa_pred = masked_pred_ids[is_truth_label_aa]

        # Calculate accuracy 
        accuracy = (truth_label_aa_input == truth_label_aa_pred).float().mean() * 100
        
        return batch_size, loss, aa_only_input, aa_only_pred, accuracy
                
    def training_step(self, batch, batch_idx):
        batch_size, loss, _, _, accuracy = self.step(batch)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_size, loss, aa_only_input, aa_only_pred, accuracy = self.step(batch)

        # Track amino acid predictions
        aa_keys = [
            f"{self.tokenizer.convert_ids_to_tokens(o)}->{self.tokenizer.convert_ids_to_tokens(p)}"
            for o, p in zip(aa_only_input.tolist(), aa_only_pred.tolist())
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

    ESMC_MODELS = {
        "esmc_300m": "esmc_300m_local", 
        "esmc_600m": "esmc_600m_local",
    }

    parser = argparse.ArgumentParser(description="Run ESM Embedder")
    parser.add_argument("--esmc_model", type=str, choices=list(ESMC_MODELS.keys()), default="esmc_300m",
                        help="ESMC model version to use. Available: %(choices)s")
    args = parser.parse_args()

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

    last_checkpoint = ModelCheckpoint(
        filename="{epoch:02d}",# name not for the last epoch
        save_last=True,        # always keep only the latest checkpoint
        save_top_k=0,          # DO NOT keep any others
        every_n_epochs=10,      # checkpoint per n epochs
        dirpath=None,          # Let PyTorch Lightning manage the directory
    )

    # Get correct number of nodes/devices from slurm
    num_nodes = os.environ.get("SLURM_JOB_NUM_NODES")
    ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")

    num_nodes = int(num_nodes) if num_nodes else 1
    ntasks_per_node = int(ntasks_per_node) if ntasks_per_node else 1

    print(f"Nodes allocated: {num_nodes}, devices allocated per node: {ntasks_per_node}")

    # Trainer setup 
    trainer= L.Trainer(
        max_epochs=15,
        limit_train_batches=1.0,    # 1.0 is 100% of batches
        limit_val_batches=1.0,      # 1.0 is 100% of batches
        strategy=DDPStrategy(find_unused_parameters=True), 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_nodes=num_nodes,
        devices=ntasks_per_node,
        logger=logger,
        callbacks=[
            best_model_checkpoint, 
            last_checkpoint, 
            TQDMProgressBar(refresh_rate=25),   # Update every 25 batches
            AccuracyLossFigureCallback(),       # For accuracy/loss plots
            AAHeatmapFigureCallback()           # For final/best AA heatmap
        ]
    )

    # Manually set the checkpoint directory after Trainer initialization
    ckpt_dir = os.path.join(trainer.logger.log_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)  # Ensure directory exists
    best_model_checkpoint.dirpath = ckpt_dir
    last_checkpoint.dirpath = ckpt_dir

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
        esm_version=ESMC_MODELS[args.esmc_model]
    )

    # Run model train/validation, load from_checkpoint if set
    start_time = time.perf_counter()
    if from_checkpoint is not None:
        trainer.fit(model, dm, ckpt_path=from_checkpoint)  # Train model from checkpoint
    else:
        trainer.fit(model, dm)  # Train model
    duration = datetime.timedelta(seconds=time.perf_counter()-start_time)
    if trainer.global_rank == 0: print(f"[Timing] Trainer.fit(...) took: {duration} (hh:mm:ss).")