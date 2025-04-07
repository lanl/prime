"""
Plotter for model runner, converted to PyTorch Lightning.
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import lightning as L
from lightning.pytorch.callbacks import Callback

class LossFigureCallback(Callback):
    def on_train_end(self, trainer: L.Trainer, lightning_module: L.LightningModule):
        """
        This method is called right after training ends. Loss plot.
        """
        log_dir = trainer.logger.log_dir

        # Extract data from metrics file
        df = pd.read_csv(
            os.path.join(log_dir, "metrics.csv"), 
            sep=',', 
            header=0,
            usecols=['epoch', 'train_rmse', 'val_rmse']
        )

        # Merge rows for each epoch and keep 'epoch' as a column
        merged_df = df.groupby("epoch").ffill().bfill().drop_duplicates().reset_index()

        # Ensure 'epoch' is correctly named and exists
        if "epoch" not in merged_df.columns:
            merged_df.rename(columns={"index": "epoch"}, inplace=True)  # Rename if it was reset incorrectly

        # Divide epoch by 2 to correct numbering
        merged_df.loc[:, "epoch"] = (merged_df["epoch"] // 2).astype(int)

        # Ensure sorting is correct
        merged_df = merged_df.sort_values("epoch").reset_index(drop=True)

        # Plot
        sns.set_style('darkgrid')
        sns.set_context('talk')
        plt.figure(figsize=(8, 4.5))
        
        # Plot CE Loss
        plt.plot(merged_df['epoch'], merged_df['train_rmse'], label='Train RMSE Loss', color='tab:red', linewidth=1.5)
        plt.plot(merged_df['epoch'], merged_df['val_rmse'], label='Validation RMSE Loss', color='tab:orange', linewidth=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        
        # Save the figure to the "figures" subdirectory.
        save_path = os.path.join(log_dir, "metrics")
        plt.savefig(f"{save_path}.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()