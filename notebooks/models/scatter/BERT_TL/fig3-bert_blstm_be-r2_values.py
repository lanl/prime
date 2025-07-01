#!/usr/bin/env python
"""
PyTorch Lightning BERT-BLSTM BE model runner for R^2 values.
"""
import argparse
import os
import torch
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from pnlp.ESM_TL.dms_models import BLSTM_BE
from pnlp.ESM_TL.dms_data_module import DmsBeDataModule  
from pnlp.BERT_MLM.model.language import BERT, ProteinMaskedLanguageModel
from pnlp.BERT_MLM.embedding.tokenizer import ProteinTokenizer, token_to_index

class LightningBertBlstmBe(L.LightningModule):
    def __init__(self, 
                 script_letter:str, from_saved:str,
                 max_len: int, mask_prob:float, blstm_model: BLSTM_BE, 
                 embedding_dim: int, dropout: float, n_transformer_layers: int, n_attn_heads: int, vocab_size: int):
        super().__init__()
        self.script_letter = script_letter
        self.tokenizer = ProteinTokenizer(max_len, mask_prob)
        self.token_to_aa = {i:aa for i, aa in enumerate('ACDEFGHIKLMNPQRSTUVWXY')} 
        self.bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
        self.mlm_model = ProteinMaskedLanguageModel(self.bert.hidden, vocab_size=vocab_size)
        self.blstm = blstm_model
        self.batch_dataframes = []

        # Load fine-tuned weights 
        if from_saved is not None:
            print(f"Loading model checkpoint from {from_saved}...")
            ckpt = torch.load(from_saved, map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=True)

        # Freeze weights
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.mlm_model.parameters():
            param.requires_grad = False
        for param in self.blstm.parameters():
            param.requires_grad = False
        print("Model weights are frozen!")

    def forward(self, masked_tokenized_seqs):
        bert_output = self.bert(masked_tokenized_seqs)
        binding_preds, expression_preds = self.blstm(bert_output)
        return self.mlm_model(bert_output), binding_preds, expression_preds
    
    def configure_optimizers(self):
        return None
    
    def step(self, batch):
        seq_ids, seqs, binding_targets, expression_targets = batch
        binding_targets = binding_targets.to(self.device).float()
        expression_targets = expression_targets.to(self.device).float()
        batch_size = binding_targets.size(0)

        masked_tokenized_seqs = self.tokenizer(seqs).to(self.device)

        _, binding_preds, expression_preds = self(masked_tokenized_seqs)

        return seq_ids, binding_targets, binding_preds, expression_targets, expression_preds, batch_size

    def training_step(self, batch, batch_idx):
        seq_ids, binding_targets, binding_preds, expression_targets, expression_preds, batch_size = self.step(batch)

        # Create a DataFrame for the batch
        batch_df = pd.DataFrame({
            "seq_id": seq_ids,
            "binding_measured_value": [target.item() for target in binding_targets],
            "binding_predicted_value": [pred.item() for pred in binding_preds],
            "expression_measured_value": [target.item() for target in expression_targets],
            "expression_predicted_value": [pred.item() for pred in expression_preds],
            "mode": ["train"] * batch_size
        })

        self.batch_dataframes.append(batch_df)

    def validation_step(self, batch, batch_idx):
        seq_ids, binding_targets, binding_preds, expression_targets, expression_preds, batch_size = self.step(batch)

        # Create a DataFrame for the batch
        batch_df = pd.DataFrame({
            "seq_id": seq_ids,
            "binding_measured_value": [target.item() for target in binding_targets],
            "binding_predicted_value": [pred.item() for pred in binding_preds],
            "expression_measured_value": [target.item() for target in expression_targets],
            "expression_predicted_value": [pred.item() for pred in expression_preds],
            "mode": ["test"] * batch_size
        })

        self.batch_dataframes.append(batch_df)

    def on_fit_end(self):
        # Concatenate all batch DataFrames into one and save
        save_as = f"bert_blstm_be/bert_blstm_be.{self.script_letter}-predicted_vs_measured.values.csv"
        result_df = pd.concat(self.batch_dataframes, ignore_index=True)
        result_df.to_csv(save_as, index=False)
        print(f"Saved predictions to: {save_as}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run BERT-BLSTM BE model with Lightning")
    parser.add_argument("--script_letter", type=str, default=None, help="Slurm model script letter.")
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
    dm = DmsBeDataModule(
        data_dir=data_dir,
        torch_geometric_tag=False, 
        batch_size=64,
        num_workers=4, 
        seed=seed
    )

    model = LightningBertBlstmBe(   
        script_letter=args.script_letter,              
        from_saved=args.from_saved,    
        max_len=280,
        mask_prob=0.15,
        blstm_model=blstm,
        embedding_dim=320,
        dropout=0.1,
        n_transformer_layers=12, 
        n_attn_heads=10,
        vocab_size=len(token_to_index)
    )

    trainer.fit(model, dm)