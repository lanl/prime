#!/usr/bin/env python
"""
PyTorch Lightning ESM-FCN model runner for R^2 values.
"""
import argparse
import os
import torch
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from transformers import EsmTokenizer, EsmModel
from pnlp.ESM_TL.dms_models import FCN
from pnlp.ESM_TL.dms_data_module import DmsDataModule  

class LightningEsmFcn(L.LightningModule):
    def __init__(self, 
                 script_letter:str, binding_or_expression:str, from_saved:str,
                 max_len: int, fcn_model: FCN, esm_version="facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        self.script_letter = script_letter
        self.binding_or_expression = binding_or_expression
        self.tokenizer = EsmTokenizer.from_pretrained(esm_version, cache_dir="../../.cache")
        self.esm = EsmModel.from_pretrained(esm_version, cache_dir="../../.cache")
        self.fcn = fcn_model
        self.max_len = max_len
        self.batch_dataframes = []

        # Load fine-tuned weights 
        if from_saved is not None:
            print(f"Loading model checkpoint from {from_saved}...")
            ckpt = torch.load(from_saved, map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=True)

        # Freeze weights
        for param in self.esm.parameters():
            param.requires_grad = False
        for param in self.fcn.parameters():
            param.requires_grad = False
        print("Model weights are frozen!")

    def forward(self, input_ids, attention_mask):
        esm_last_hidden_state = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # shape: [batch_size, sequence_length, embedding_dim]
        esm_cls_embedding = esm_last_hidden_state[:, 0, :]  # CLS token embedding (sequence-level representations), [batch_size, embedding_dim]
        return self.fcn(esm_cls_embedding)  # [batch_size]
    
    def configure_optimizers(self):
        return None
    
    def step(self, batch):
        seq_ids, seqs, targets = batch
        targets = targets.to(self.device).float()
        batch_size = targets.size(0)

        tokenized = self.tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        preds = self(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"])

        return seq_ids, targets, preds, batch_size

    def training_step(self, batch, batch_idx):
        seq_ids, targets, preds, batch_size = self.step(batch)

        # Create a DataFrame for the batch
        batch_df = pd.DataFrame({
            "seq_id": seq_ids,
            "measured_value": [target.item() for target in targets],
            "predicted_value": [pred.item() for pred in preds],
            "mode": ["train"] * batch_size
        })

        self.batch_dataframes.append(batch_df)

    def validation_step(self, batch, batch_idx):
        seq_ids, targets, preds, batch_size = self.step(batch)

        # Create a DataFrame for the batch
        batch_df = pd.DataFrame({
            "seq_id": seq_ids,
            "measured_value": [target.item() for target in targets],
            "predicted_value": [pred.item() for pred in preds],
            "mode": ["test"] * batch_size
        })

        self.batch_dataframes.append(batch_df)

    def on_fit_end(self):
        # Concatenate all batch DataFrames into one and save
        save_as = f"esm_fcn/esm_fcn.{self.script_letter}.{self.binding_or_expression}-predicted_vs_measured.values.csv"
        result_df = pd.concat(self.batch_dataframes, ignore_index=True)
        result_df.to_csv(save_as, index=False)
        print(f"Saved predictions to: {save_as}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run ESM-FCN model with Lightning")
    parser.add_argument("--script_letter", type=str, default=None, help="Slurm model script letter.")
    parser.add_argument("--binding_or_expression", type=str, default=None, help="Binding or Expression.")
    parser.add_argument("--from_saved", type=str, default=None, help="Path to existing model to extract weights from.")
    args = parser.parse_args()

    # Random seed
    seed = 0
    L.seed_everything(seed)  # Set seed for reproducibility

    # Trainer setup 
    trainer= L.Trainer(
        max_epochs=1,
        limit_train_batches=1.0,    # 1.0 is 100% of batches
        limit_val_batches=1.0,      # 1.0 is 100% of batches
        strategy="auto", 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_nodes=1,
        devices=1,
        callbacks=[TQDMProgressBar(refresh_rate=25)]
    )

    # Data directory (no results_dir needed since versioning handles it automatically)
    data_dir= os.path.join(os.path.dirname(__file__), f"../../../../data/dms")

    # FCN input, size should match used ESM model embedding_dim size
    size = 320
    fcn_input_size = size  
    fcn_hidden_size = size
    fcn_num_layers = 5
    fcn = FCN(fcn_input_size, fcn_hidden_size, fcn_num_layers)

    # Initialize DataModule and model
    dm = DmsDataModule(
        data_dir=data_dir,
        binding_or_expression=args.binding_or_expression,  
        torch_geometric_tag=False, 
        batch_size=64,
        num_workers=4, 
        seed=seed
    )

    model = LightningEsmFcn(   
        script_letter=args.script_letter,
        binding_or_expression=args.binding_or_expression,              
        from_saved=args.from_saved,    
        max_len=280,
        fcn_model=fcn,
        esm_version="facebook/esm2_t6_8M_UR50D"
    )

    trainer.fit(model, dm)