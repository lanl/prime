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
    def on_fit_end(self, trainer: L.Trainer, lightning_module: L.LightningModule):
        """
        This method is called at the end of `trainer.fit`. Loss plot.
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
        
        # Plot RMSE Loss
        plt.plot(merged_df['epoch'], merged_df['train_rmse'], label='Train', color='tab:blue', linewidth=1.5)
        plt.plot(merged_df['epoch'], merged_df['val_rmse'], label='Validation', color='tab:orange', linewidth=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE Loss')
        plt.legend(loc='best')
        
        # Save the figure to the "figures" subdirectory.
        save_path = os.path.join(log_dir, "metrics")
        plt.savefig(f"{save_path}.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()

class LossBeFigureCallback(Callback):
    def on_fit_end(self, trainer: L.Trainer, lightning_module: L.LightningModule):
        """
        This method is called at the end of `trainer.fit`. Loss plot.
        """
        log_dir = trainer.logger.log_dir

        # Extract data from metrics file
        df = pd.read_csv(
            os.path.join(log_dir, "metrics.csv"), 
            sep=',', 
            header=0,
            usecols=['epoch', 'train_binding_rmse', 'val_binding_rmse', 'train_expression_rmse', 'val_expression_rmse', 'train_be_rmse', 'val_be_rmse']
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
        fig, axs = plt.subplots(2, 1, figsize=(8, 9), sharex=True)  # 2 rows, 1 column
        plt.subplots_adjust(hspace=0.05)

        # Plot Training Loss
        axs[0].plot(merged_df['epoch'], merged_df['train_binding_rmse'], label='Train Binding', color='tab:blue', linewidth=1.5)
        axs[0].plot(merged_df['epoch'], merged_df['train_expression_rmse'], label='Train Expression', color='tab:orange', linewidth=1.5)
        axs[0].plot(merged_df['epoch'], merged_df['train_be_rmse'], label='Train BE', color='tab:green', linewidth=1.5)
        axs[0].set_ylabel('RMSE Loss')
        axs[0].legend(loc='best')

        # Plot Testing Loss
        axs[1].plot(merged_df['epoch'], merged_df['val_binding_rmse'], label='Validation Binding', color='tab:blue', linewidth=1.5)
        axs[1].plot(merged_df['epoch'], merged_df['val_expression_rmse'], label='Validation Expression', color='tab:orange', linewidth=1.5)
        axs[1].plot(merged_df['epoch'], merged_df['val_be_rmse'], label='Validation BE', color='tab:green', linewidth=1.5)
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('RMSE Loss')
        axs[1].legend(loc='best')
        
        # Save the figure to the "figures" subdirectory.
        save_path = os.path.join(log_dir, "metrics")
        plt.savefig(f"{save_path}.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()