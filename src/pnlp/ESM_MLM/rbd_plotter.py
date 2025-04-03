"""
Plotter for model runner, converted to PyTorch Lightning.
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

import lightning as L
from lightning.pytorch.callbacks import Callback

class AccuracyLossFigureCallback(Callback):
    def on_train_end(self, trainer: L.Trainer, lightning_module: L.LightningModule):
        """
        This method is called right after training ends. Accuracy and loss plot.
        """
        log_dir = trainer.logger.log_dir

        # Extract data from metrics file
        df = pd.read_csv(
            os.path.join(log_dir, "metrics.csv"), 
            sep=',', 
            header=0,
            usecols=['epoch', 'train_accuracy', 'train_loss', 'val_accuracy', 'val_loss']
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
        fig.subplots_adjust(hspace=0.1)
        
        # Plot CE Loss
        axs[0].plot(merged_df['epoch'], merged_df['train_loss'], label='Train CE Loss', color='tab:red', linewidth=1.5)
        axs[0].plot(merged_df['epoch'], merged_df['val_loss'], label='Test (Validation) CE Loss', color='tab:orange', linewidth=1.5)
        axs[0].set_ylabel('Loss')
        axs[0].legend(loc='best')

        # Plot Accuracy
        axs[1].plot(merged_df['epoch'], merged_df['train_accuracy'], label='Train Accuracy', color='tab:blue', linewidth=1.5)
        axs[1].plot(merged_df['epoch'], merged_df['val_accuracy'], label='Test (Validation) Accuracy', color='tab:green', linewidth=1.5)
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy (%)')
        axs[1].set_ylim(0, 100)
        axs[1].legend(loc='best')

        # Save the figure to the "figures" subdirectory.
        save_path = os.path.join(log_dir, "metrics")
        plt.savefig(f"{save_path}.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()

class AAHeatmapFigureCallback(Callback):
    def on_train_end(self, trainer: L.Trainer, lightning_module: L.LightningModule):
        """
        This method is called right after training ends. Amino acid prediction heatmap.
        """
        log_dir = trainer.logger.log_dir
        aa_preds_dir = os.path.join(log_dir, "aa_preds")

        ALL_AAS = 'ACDEFGHIKLMNPQRSTVWY'
        all_combinations = [(e_aa, p_aa) for e_aa in ALL_AAS for p_aa in ALL_AAS]
        all_df = pd.DataFrame(all_combinations, columns=["Expected", "Predicted"])

        # Aggregate counts
        total_data = defaultdict(int)  # key: "Expected->Predicted", value: total count

        for epoch in range(trainer.max_epochs):
            for rank in range(trainer.world_size):
                csv_path = os.path.join(aa_preds_dir, f"aa_predictions_epoch{epoch}_rank{rank}.csv")
                if not os.path.exists(csv_path):
                    continue

                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    key = row['expected_aa->predicted_aa']
                    count = int(row['count'])
                    total_data[key] += count

        # Convert dict to DataFrame
        records = []
        for aa_pair, count in total_data.items():
            expected, predicted = aa_pair.split('->')
            records.append({'Expected': expected, 'Predicted': predicted, 'Total Count': count})

        agg_df = pd.DataFrame(records)

        # Merge with all combinations to fill in missing zero entries
        df = pd.merge(all_df, agg_df, how="left", on=["Expected", "Predicted"])
        df["Total Count"] = df["Total Count"].fillna(0)

        # Compute expected totals and error percentage
        expected_totals = df.groupby("Expected")["Total Count"].sum()
        df["Expected Total"] = df["Expected"].map(expected_totals)
        df["Error Percentage"] = (df["Total Count"] / df["Expected Total"]) * 100
        df["Error Percentage"] = df["Error Percentage"].fillna(0)

        # Pivot for heatmap
        heatmap_data = df.pivot_table(index="Predicted", columns="Expected", values="Error Percentage")

        # Plot heatmap
        sns.set_style('ticks')
        sns.set_context('talk')
        plt.figure(figsize=(16, 9))

        heatmap = sns.heatmap(
            heatmap_data,
            annot=True, fmt=".2f",
            linewidth=.5,
            cmap="rocket_r", vmin=0, vmax=100,
            annot_kws={"size": 12},
            cbar_kws={'drawedges': False, 'label': 'Prediction Rate (%)'}
        )

        colorbar = heatmap.collections[0].colorbar
        colorbar.set_label('Prediction Rate (%)')

        plt.xlabel('Expected Amino Acid')
        plt.ylabel('Predicted Amino Acid')
        plt.title("Amino Acid Prediction Error Across Epochs")

        # Save the figure to the "figures" subdirectory.
        save_path = os.path.join(aa_preds_dir , "aa_predictions_all_epochs")
        plt.savefig(f"{save_path}.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()
