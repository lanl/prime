"""
Plotter for model runner, converted to PyTorch Lightning.
"""
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from torch.distributed import barrier

class AccuracyLossFigureCallback(Callback):
    def on_fit_end(self, trainer: L.Trainer, lightning_module: L.LightningModule):
        """
        This method is called at the end of `trainer.fit`. Accuracy and loss plot.
        """
        # Skip non-zero ranks
        if trainer.global_rank != 0:
            return  

        log_dir = trainer.logger.log_dir

        # Check if metrics file exists
        metrics_path = os.path.join(log_dir, "metrics.csv")
        if not os.path.exists(metrics_path):
            print(f"Metrics file not found at {metrics_path}. Skipping plot generation.")
            return

        # Extract data from metrics file
        df = pd.read_csv(
            metrics_path, 
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
    def generate_heatmap(self, trainer: L.Trainer, epoch: int, tag: str):
        """
        Amino acid prediction heatmap (only single epoch data, aggregated across ranks).
        """
        log_dir = trainer.logger.log_dir
        aa_preds_dir = os.path.join(log_dir, "aa_preds")

        ALL_AAS = 'ACDEFGHIKLMNPQRSTVWY'
        all_combinations = [(e_aa, p_aa) for e_aa in ALL_AAS for p_aa in ALL_AAS]
        all_df = pd.DataFrame(all_combinations, columns=["Expected", "Predicted"])

        # Aggregate counts for current epoch
        total_data = defaultdict(int)  # key: "Expected->Predicted", value: total count

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
        plt.title(f"{tag} Amino Acid Prediction Error")

        # Save the figure
        save_path = os.path.join(aa_preds_dir , f"aa_predictions_heatmap-{tag.lower()}_epoch{epoch}")
        print(f"{tag} figure located at {save_path}")
        plt.savefig(f"{save_path}.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()

    def on_fit_end(self, trainer: L.Trainer, lightning_module: L.LightningModule):
        """
        This method is called at the end of `trainer.fit`. Calls generate_heatmap 
        for the best epoch, and the last epoch.
        """
        # Skip non-zero ranks
        if trainer.global_rank != 0:
            return  
        
        # Existing best/final epoch determination
        best_epoch = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and (cb.monitor == "val_accuracy" or cb.monitor == "val_be_rmse" or cb.monitor == "val_rmse"):
                best_model_path = cb.best_model_path
                if not os.path.exists(best_model_path):
                    print(f"Best file not found at {best_model_path}. Skipping plot generation.")
                    continue

                else:
                    if "epoch=" in best_model_path:
                        print(f"Best file found at {best_model_path}. Plotting.")
                        best_epoch = int(best_model_path.split("epoch=")[1].split(".")[0])
                        print(best_epoch)
        
        final_epoch = trainer.max_epochs-1

        # Verify all needed files exist
        log_dir = trainer.logger.log_dir
        aa_preds_dir = os.path.join(log_dir, "aa_preds")

        def all_files_ready():
            for epoch in [best_epoch, final_epoch]:
                if epoch is not None:
                    for rank in range(trainer.world_size):
                        csv_path = os.path.join(aa_preds_dir, f"aa_predictions_epoch{epoch}_rank{rank}.csv")
                        try:
                            if not os.path.exists(csv_path):
                                return False
                            pd.read_csv(csv_path)  # Try reading to confirm it's valid
                        except Exception:
                            return False
            return True

        # Wait until all files exist
        timeout_seconds = 20
        start_time = time.time()

        print("Verifying prediction files...")
        while not all_files_ready():
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise FileNotFoundError(f"Files not found after {timeout_seconds} seconds. Ending check.")
            
            print(f"Waiting for all prediction files to be written... {elapsed_time}/{timeout_seconds} sec elapsed before giving up.")
            time.sleep(1)
        
        print("All files confirmed. Generating heatmaps...")

        # Generate heatmaps after verification
        if best_epoch is not None: self.generate_heatmap(trainer, best_epoch, "Best")
        self.generate_heatmap(trainer, final_epoch, "Final")
