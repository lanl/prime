import os
import re
import pandas as pd

def main():
    data_dir = '../../src/pnlp/ESM_TL/logs'

    for model_folder in os.listdir(data_dir):
        model_folder_path = os.path.join(data_dir, model_folder)
        if os.path.isdir(model_folder_path):  # Check if the path is a directory

            for version_folder in os.listdir(model_folder_path):
                version_folder_path = os.path.join(model_folder_path, version_folder)
                if os.path.isdir(version_folder_path):

                    for file in os.listdir(version_folder_path):
                        file_path = os.path.join(version_folder_path, file)
                        if file == 'metrics.csv':
                            
                            # Extract data from metrics file
                            df = pd.read_csv(file_path, sep=',', header=0)

                            # Merge rows for each epoch and keep 'epoch' as a column
                            merged_df = df.groupby("epoch").ffill().bfill().drop_duplicates().reset_index()

                            # Ensure 'epoch' is correctly named and exists
                            if "epoch" not in merged_df.columns:
                                merged_df.rename(columns={"index": "epoch"}, inplace=True)  # Rename if it was reset incorrectly

                            # Divide epoch by 2 to correct numbering
                            merged_df.loc[:, "epoch"] = (merged_df["epoch"] // 2).astype(int)

                            # Ensure sorting is correct
                            merged_df = merged_df.sort_values("epoch").reset_index(drop=True)

                            if model_folder.endswith("be"):
                                # Get best (min) rmse row
                                best_rmse_row = merged_df.loc[merged_df["val_be_rmse"].idxmin()]
                                epoch = best_rmse_row["epoch"]
                                rmse = best_rmse_row["val_be_rmse"]
                                b_rmse = best_rmse_row["val_binding_rmse"]
                                e_rmse = best_rmse_row["val_expression_rmse"]
                                print(f"{model_folder}-{version_folder.split('_')[1]}: \n\t(best be) epoch {int(epoch)}, rmse {rmse:.4f}; b_rmse {b_rmse:.4f}, e_rmse {e_rmse:.4f}")

                                best_binding_rmse_row = merged_df.loc[merged_df["val_binding_rmse"].idxmin()]
                                epoch = best_binding_rmse_row["epoch"]
                                b_rmse = best_binding_rmse_row["val_binding_rmse"]
                                print(f"\t(best binding) epoch {int(epoch)}, b_rmse {b_rmse:.4f}")

                                best_expression_rmse_row = merged_df.loc[merged_df["val_expression_rmse"].idxmin()]
                                epoch = best_expression_rmse_row["epoch"]
                                e_rmse = best_expression_rmse_row["val_expression_rmse"]
                                print(f"\t(best expression) epoch {int(epoch)}, e_rmse {e_rmse:.4f}")

                            else:
                                # Get best (min) rmse row
                                best_rmse_row = merged_df.loc[merged_df["val_rmse"].idxmin()]
                                epoch = best_rmse_row["epoch"]
                                rmse = best_rmse_row["val_rmse"]
                                print(f"{model_folder}-{version_folder.split('_')[1]}: \n\t(best) epoch {int(epoch)}, rmse {rmse:.4f}")
                            
if __name__ == '__main__':
    main()