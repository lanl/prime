"""
Plotter for ESM-initialized BERT-MLM model runner, converted to PyTorch Lightning.
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_lightning as pl

class SaveFiguresCallback(pl.callbacks.Callback):
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        This method is called right after training ends.
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
        sns.set_theme(style="darkgrid")
        fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)  # 2 rows, 1 column
        fontsize = 16
        
        # Plot CE Loss
        axes[0].plot(merged_df['epoch'], merged_df['val_loss'], label='Test (Validation) CE Loss', color='tab:orange', linewidth=3)
        axes[0].plot(merged_df['epoch'], merged_df['train_loss'], label='Train CE Loss', color='tab:red', linewidth=3)
        axes[0].set_ylabel('Loss', fontsize=fontsize)
        axes[0].tick_params(axis='y', labelsize=fontsize)
        axes[0].legend(loc='best', fontsize=fontsize)

        # Plot Accuracy
        axes[1].plot(merged_df['epoch'], merged_df['val_accuracy'], label='Test (Validation) Accuracy', color='tab:green', linewidth=3)
        axes[1].plot(merged_df['epoch'], merged_df['train_accuracy'], label='Train Accuracy', color='tab:blue', linewidth=3)
        axes[1].set_ylabel('Accuracy (%)', fontsize=fontsize)
        axes[1].tick_params(axis='y', labelsize=fontsize)
        axes[1].set_ylim(0, 100)
        axes[1].tick_params(axis='x', labelsize=fontsize)
        axes[1].legend(loc='best', fontsize=fontsize)

        plt.xlabel('Epoch', fontsize=fontsize)
        plt.tight_layout()

        # Save the figure to the "figures" subdirectory.
        save_path = os.path.join(log_dir, "metrics_plot.pdf")
        plt.savefig(save_path, format='pdf')
        plt.savefig(save_path.replace('.pdf', '.png'), format='png')
        plt.close(fig)

def plot_aa_preds_heatmap(preds_csv, preds_img):
    """ Plots heatmap of expected vs predicted amino acid incorrect prediction counts. Expected on x axis. """
    df = pd.read_csv(preds_csv)

    # Create a DataFrame with all possible amino acid combinations
    ALL_AAS = 'ACDEFGHIKLMNPQRSTVWY'
    all_combinations = [(e_aa, p_aa) for e_aa in ALL_AAS for p_aa in ALL_AAS]
    all_df = pd.DataFrame(all_combinations, columns=["Expected", "Predicted"])

    # Split 'expected_aa->predicted_aa' into separate columns
    df[['Expected', 'Predicted']] = df['expected_aa->predicted_aa'].str.split('->', expand=True)

    # Rename 'count' column to 'Total Count' for consistency
    df = df.rename(columns={'count': 'Total Count'})

    # Merge with all possible amino acid combinations so missing pairs get a count of 0
    df = pd.merge(all_df, df[['Expected', 'Predicted', 'Total Count']], how="left", on=["Expected", "Predicted"])
    
    # Fix inplace assignment issue
    df.loc[:, "Total Count"] = df["Total Count"].fillna(0)

    # Calculate the total counts for each expected amino acid
    total_counts = df.groupby("Expected")["Total Count"].sum()
    df.loc[:, "Expected Total"] = df["Expected"].map(total_counts)

    # Calculate error percentage
    df.loc[:, "Error Percentage"] = (df["Total Count"] / df["Expected Total"]) * 100
    df.loc[:, "Error Percentage"] = df["Error Percentage"].fillna(0)

    # Pivot the DataFrame to create a heatmap data structure
    heatmap_data = df.pivot_table(index="Predicted", columns="Expected", values="Error Percentage")

    # Set figure size
    plt.figure(figsize=(16, 9))
    fontsize = 16

    # Plot
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    heatmap = sns.heatmap(
        heatmap_data,
        annot=True, fmt=".2f",
        linewidth=.5,
        cmap=cmap, vmin=0, vmax=100,
        annot_kws={"size": 13},
        cbar_kws={'drawedges': False, 'label': 'Prediction Rate (%)'}
    )

    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=fontsize)  # Set colorbar tick label size
    colorbar.set_label('Prediction Rate (%)', size=fontsize)  # Set colorbar label size

    plt.xlabel('Expected Amino Acid', fontsize=fontsize)
    plt.xticks(rotation=0, fontsize=fontsize-2)
    plt.ylabel('Predicted Amino Acid', fontsize=fontsize)
    plt.yticks(fontsize=fontsize-2)

    plt.tight_layout()
    plt.savefig(preds_img, format='pdf')
    plt.savefig(preds_img.replace('.pdf', '.png'), format='png')
    plt.close()  # Close the figure to free up memory
