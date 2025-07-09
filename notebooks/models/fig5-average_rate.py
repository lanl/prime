import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

def generate_heatmap_df(epoch: int, log_dir: str):
    """
    Amino acid prediction heatmap (only single epoch data, aggregated across ranks).
    """
    aa_preds_dir = os.path.join(log_dir, "aa_preds")

    ALL_AAS = 'ACDEFGHIKLMNPQRSTVWY'
    all_combinations = [(e_aa, p_aa) for e_aa in ALL_AAS for p_aa in ALL_AAS]
    all_df = pd.DataFrame(all_combinations, columns=["Expected", "Predicted"])

    # Aggregate counts for current epoch
    total_data = defaultdict(int)  # key: "Expected->Predicted", value: total count

    for rank in range(8):
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

    # Print diagonal entries (correct predictions)
    correct_preds = df[df["Expected"] == df["Predicted"]][["Expected", "Total Count", "Error Percentage"]]
    #print("\nCorrect Predictions (Expected == Predicted):")
    #print(correct_preds.sort_values(by="Expected").to_string(index=False))
    average_rate = sum(correct_preds['Error Percentage'].values.tolist()) / len(correct_preds)
    print(f"average_rate: {average_rate:.2f}")

if __name__ == "__main__":

    bert_model_dir = "../../src/pnlp/BERT_TL/logs/bert_blstm_be"
    log_dirs = [os.path.join(bert_model_dir, d) for d in os.listdir(bert_model_dir)]

    for ld in log_dirs:
        script_letter = os.listdir(os.path.join(ld, "slurm_out"))[0].split(".")[0][-1]
        print(f"{script_letter}: {ld}")
        aa_preds_dir = os.path.join(ld, 'aa_preds')
        best_epoch = next((
            int(f.split("epoch")[1].split(".")[0]) 
            for f in os.listdir(aa_preds_dir) 
            if "best" in f and "epoch" in f
        ), None)

        if best_epoch is None:
            print(f"[SKIP] No best epoch found in {aa_preds_dir}")
            continue
        else:
            generate_heatmap_df(best_epoch, ld)

    print("")
    esm_model_dir = "../../src/pnlp/ESM_TL/logs/esm_mlm_blstm_be"
    log_dirs = [os.path.join(esm_model_dir, d) for d in os.listdir(esm_model_dir)]

    for ld in log_dirs:
        script_letter = os.listdir(os.path.join(ld, "slurm_out"))[0].split(".")[0][-1]
        print(f"{script_letter}: {ld}")
        aa_preds_dir = os.path.join(ld, 'aa_preds')
        best_epoch = next((
            int(f.split("epoch")[1].split(".")[0]) 
            for f in os.listdir(aa_preds_dir) 
            if "best" in f and "epoch" in f
        ), None)

        if best_epoch is None:
            print(f"[SKIP] No best epoch found in {aa_preds_dir}")
            continue
        else:
            generate_heatmap_df(best_epoch, ld)