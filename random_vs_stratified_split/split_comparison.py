#!/usr/bin/env python
"""
Testing random split vs position-stratified split. 

To isolate the impact of data partitioning, we compared a standard random split 
and a position-stratified split generated within the same evaluation framework 
using frozen ESM embeddings and an identical regression head. Note, this random 
split is different from how we originally random split the data, so the values 
do NOT match!
"""
import json
import tqdm
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from pnlp.ESM_TL.dms_models import FCN_BE

class SimpleFcnBeWrapper:
    def __init__(self, fcn_input_size, fcn_hidden_size, fcn_num_layers, lr, epochs, batch_size, device, random_seed=0):
        super().__init__()
        self.fcn_input_size = fcn_input_size
        self.fcn_hidden_size = fcn_hidden_size
        self.fcn_num_layers = fcn_num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.random_seed = random_seed
        self._set_seed()
        
        self.model = FCN_BE(self.fcn_input_size, self.fcn_hidden_size, self.fcn_num_layers).to(self.device)
        self.loss_fn = torch.nn.MSELoss()

    def _set_seed(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

    def fit(self, X, y_binding, y_expression):
        self._set_seed()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        yb = torch.tensor(y_binding, dtype=torch.float32, device=self.device)
        ye = torch.tensor(y_expression, dtype=torch.float32, device=self.device)

        self.model.train()
        for _ in tqdm.trange(self.epochs, desc="Training epochs"):
            perm = torch.randperm(len(X), device=self.device)   # Create a random permutation of indices, basically DataLoader(shuffle=True)
            for i in range(0, len(X), self.batch_size):
                idx = perm[i:i+self.batch_size]
                xbatch = X[idx]
                ybatch_b = yb[idx]
                ybatch_e = ye[idx]

                optimizer.zero_grad()
                pred_b, pred_e = self.model(xbatch)
                loss = self.loss_fn(pred_b, ybatch_b) + self.loss_fn(pred_e, ybatch_e)
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        pred_b, pred_e = self.model(X)
        return (pred_b.cpu().numpy(), pred_e.cpu().numpy())        

    def clone(self):
        return SimpleFcnBeWrapper(self.fcn_input_size, self.fcn_hidden_size, self.fcn_num_layers, self.lr, self.epochs, self.batch_size, self.device, self.random_seed)
    
class PositionStratifiedSplitter:
    """ 
    Position-stratified splitter for DMS datasets. 
    
    Args:
        wt_sequence: wild-type sequence
        n_splits: Number of folds for cross-validation
        random_state: Random seed for reproducibility
    """    
    def __init__(self, wt_sequence: str, n_splits: int = 5, random_state: int = 42):
        self.wt_sequence = wt_sequence
        self.n_splits = n_splits
        self.random_state = random_state
    
    def _extract_mutated_positions(self, sequence: str):
        """
        Extract positions where sequence differs from wild-type.
        Returns: list of positions (can be multiple for combinatorial mutations)
        """
        positions = []
        for i, (aa1, aa2) in enumerate(zip(self.wt_sequence, sequence)):
            if aa1 != aa2:
                positions.append(i)
        return positions

    def split(self, df):
        """
        Perform position-stratified k-fold split.
        Returns:
            List[dict], one dict per fold with:
              - fold
              - train_idx
              - test_idx
              - excluded_idx
              - train_positions
              - test_positions
        """
        # Identify all mutated positions for each sequence
        if 'mutated_positions' not in df.columns:
            df = df.copy()
            df['mutated_positions'] = df['sequence'].apply(self._extract_mutated_positions)
        
        # Collect unique positions across all mutations
        all_positions = set()
        for positions in df['mutated_positions']:
            all_positions.update(positions)
        all_positions = sorted(list(all_positions))
        
        print(f"Total unique positions mutated: {len(all_positions)}")
        
        # Create position-based folds
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        fold_splits = []
        for fold_idx, (train_positions_idx, test_positions_idx) in enumerate(kf.split(all_positions)):
            train_positions = sorted([all_positions[i] for i in train_positions_idx])
            test_positions = sorted([all_positions[i] for i in test_positions_idx])
            train_positions_set = set(train_positions)
            test_positions_set = set(test_positions)
            
            # Assign sequences to train/test based on their mutated positions
            train_mask = df['mutated_positions'].apply(
                lambda pos_list: all(pos in train_positions_set for pos in pos_list)
            )
            test_mask = df['mutated_positions'].apply(
                lambda pos_list: any(pos in test_positions_set for pos in pos_list) and
                                 all(pos not in train_positions_set or pos in test_positions_set for pos in pos_list)
            )
            excluded_mask = ~(train_mask | test_mask)
            
            train_idx = df.index[train_mask].tolist()
            test_idx = df.index[test_mask].tolist()
            excluded_idx = df.index[excluded_mask].tolist()
            
            fold_info = {
                "fold": fold_idx,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "excluded_idx": excluded_idx,
                "train_positions": train_positions,
                "test_positions": test_positions,
            }
            fold_splits.append(fold_info)
            
            print(f"Fold {fold_idx + 1}:")
            print(f"  Train positions: {len(train_positions)} | Train sequences: {len(train_idx)}")
            print(f"    Train mutation positions: {train_positions}")
            print(f"    Train indices (first 20): {train_idx[:20]}{'...' if len(train_idx) > 20 else ''}")
            print(f"  Test positions: {len(test_positions)} | Test sequences: {len(test_idx)}")
            print(f"    Test mutation positions: {test_positions}")
            print(f"    Test indices (first 20): {test_idx[:20]}{'...' if len(test_idx) > 20 else ''}")
            print(f"  Excluded sequences: {len(excluded_idx)}")
            print(f"    Excluded indices (first 20): {excluded_idx[:20]}{' ...' if len(excluded_idx) > 20 else ''}")
        
        raw_df, distribution_df = self.summarize_mutation_count_distribution(df, fold_splits)
        print(f"\n{distribution_df.to_string(index=False)}")

        return fold_splits, raw_df, distribution_df
        
    def summarize_mutation_count_distribution(self, df, fold_splits):
        """ Summarize mutation-count distribution per fold for train/test/excluded sets. """
        df = df.copy()

        if "mutated_positions" not in df.columns:
            raise ValueError("df must contain 'mutated_positions' before calling this function.")

        df["num_mutations"] = df["mutated_positions"].apply(len)

        target_cols = [
            col for col in ["ACE2-binding_affinity", "RBD_expression"]
            if col in df.columns
        ]

        raw_rows = []
        summary_rows = []
        for fold_info in fold_splits:
            fold_idx = fold_info["fold"] + 1 # use 1-based fold labels

            split_map = {
                "train": fold_info["train_idx"],
                "test": fold_info["test_idx"],
                "excluded": fold_info.get("excluded_idx", []),
            }

            for split_name, idx in split_map.items():
                sub = df.loc[idx].copy()
                num_mut = sub["num_mutations"]

                summary_rows.append({
                    "fold": fold_idx,
                    "split": split_name,
                    "n_sequences": len(sub),
                    "n_single": int((num_mut == 1).sum()),
                    "n_double": int((num_mut == 2).sum()),
                    "n_triple_plus": int((num_mut >= 3).sum()),
                    "pct_single": float((num_mut == 1).mean() * 100) if len(sub) else np.nan,
                    "pct_double": float((num_mut == 2).mean() * 100) if len(sub) else np.nan,
                    "pct_triple_plus": float((num_mut >= 3).mean() * 100) if len(sub) else np.nan,
                    "mean_mutations": float(num_mut.mean()) if len(sub) else np.nan,
                    "std_mutations": float(num_mut.std(ddof=0)) if len(sub) else np.nan,
                    "min_mutations": int(num_mut.min()) if len(sub) else np.nan,
                    "max_mutations": int(num_mut.max()) if len(sub) else np.nan,
                })

                for original_idx, row in sub.iterrows():
                    raw_rows.append({
                        "fold": fold_idx,
                        "split": split_name,
                        "original_index": original_idx,
                        "sequence": row["sequence"],
                        "mutated_positions": row["mutated_positions"],
                        "num_mutations": row["num_mutations"],
                        **{col: row[col] for col in target_cols},
                    })

        raw_df = pd.DataFrame(raw_rows)
        distribution_df  = pd.DataFrame(summary_rows)

        return raw_df, distribution_df 
    
def evaluate_position_stratified(model, df, embeddings, fold_splits=None, n_splits=5, random_seeds=[42]):
    """ Evaluate model performance using position-stratified cross-validation. """
    seed_results = []
    all_distribution_dfs = []
    all_distribution_raw_dfs = []
    for i, seed in enumerate(random_seeds):
        print(f"\n== Training seed ({seed}) {i+1}/{len(random_seeds)}... ==")

        ps_splitter = PositionStratifiedSplitter(
            wt_sequence = "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST", 
            n_splits=n_splits,
            random_state=seed
        )
        fold_splits, raw_df, distribution_df = ps_splitter.split(df)

        # Raw and summarized df about the fold break down, same for every model
        for df_out in [raw_df, distribution_df]:
            if "random_seed" in df_out.columns:
                df_out.drop(columns="random_seed", inplace=True)

            df_out.insert(0, "random_seed", seed)
        
        all_distribution_dfs.append(distribution_df)
        all_distribution_raw_dfs.append(raw_df)

        fold_results = []
        for fold_info in fold_splits:

            fold_idx = fold_info["fold"]
            train_idx = fold_info["train_idx"]
            test_idx = fold_info["test_idx"]
        
            print(f"\nTraining fold {fold_idx + 1}/{len(fold_splits)}...")
            
            # Prepare training data
            X_train = embeddings[train_idx]
            y_train_binding = df.loc[train_idx, 'ACE2-binding_affinity'].values
            y_train_expression = df.loc[train_idx, 'RBD_expression'].values
            
            # Prepare test data
            X_test = embeddings[test_idx]
            y_test_binding = df.loc[test_idx, 'ACE2-binding_affinity'].values
            y_test_expression = df.loc[test_idx, 'RBD_expression'].values
            
            # Train model (reinitialize for each fold)
            model_fold = model.clone()  # Or reinitialize your model
            model_fold.fit(X_train, y_train_binding, y_train_expression)
            
            # Predict on test set
            y_pred_binding, y_pred_expression = model_fold.predict(X_test)

            fold_results.append({
                "random_seed": seed,
                "fold": fold_idx+1,
                "binding_r2":r2_score(y_test_binding, y_pred_binding),
                "binding_rmse":np.sqrt(mean_squared_error(y_test_binding, y_pred_binding)),
                "expression_r2":r2_score(y_test_expression, y_pred_expression),
                "expression_rmse":np.sqrt(mean_squared_error(y_test_expression, y_pred_expression)),
            })

        # Per-fold results for this seed
        fold_results_df = pd.DataFrame(fold_results)
        seed_results.append(fold_results_df)
        print(f"\n{fold_results_df.to_string(index=False)}")

        # Results summary across folds for this seed
        fold_summary_df = pd.DataFrame({
            "mean": fold_results_df.drop(columns=["random_seed", "fold"]).mean(), 
            "std": fold_results_df.drop(columns=["random_seed", "fold"]).std(ddof=0)
        }).T
        print(f"\n{fold_summary_df.to_string()}")

    all_distribution_df = pd.concat(all_distribution_dfs, ignore_index=True)
    all_distribution_raw_df = pd.concat(all_distribution_raw_dfs, ignore_index=True)

    print(f"\n== RESULTS SUMMARY ==")

    # One row per fold per seed
    seed_fold_results_df = pd.concat(seed_results, ignore_index=True)
    print(f"One row per fold per seed\n{seed_fold_results_df.to_string(index=False)}")

    # One row per seed, averaged across folds 
    seed_results_df = (
        seed_fold_results_df
        .drop(columns=["fold"])
        .groupby("random_seed", as_index=False)
        .mean()
    )
    print(f"\nOne row per seed, averaged across folds\n(same as mean/std tables above, without std stated)\n{seed_results_df.to_string(index=False)}")

    # Results summary across seeds (mean/std)
    seed_summary_df = pd.DataFrame({
        "mean": seed_results_df.drop(columns=["random_seed"]).mean(), 
        "std": seed_results_df.drop(columns=["random_seed"]).std(ddof=0)
    }).T
    print(f"\nSummary across seeds\n{seed_summary_df.to_string()}\n")
    
    return seed_fold_results_df, seed_results_df, seed_summary_df, all_distribution_df, all_distribution_raw_df
    return all_distribution_df, all_distribution_raw_df

def evaluate_random_split(model, df, embeddings, random_seeds=[42]):
    """ Evaluate model performance using random split. """
    seed_results = []
    for i, seed in enumerate(random_seeds):

        train_idx, test_idx = train_test_split(
            range(len(df)), test_size=0.2, random_state=seed
        )

        X_train = embeddings[train_idx]
        X_test = embeddings[test_idx]
        y_train_binding = df.iloc[train_idx]['ACE2-binding_affinity'].values
        y_test_binding = df.iloc[test_idx]['ACE2-binding_affinity'].values
        y_train_expression = df.iloc[train_idx]['RBD_expression'].values
        y_test_expression = df.iloc[test_idx]['RBD_expression'].values

        print(f"\n== Training seed ({seed}) {i+1}/{len(random_seeds)}... ==")
        print(f"  Train binding: {len(y_train_binding)}; Train expression: {len(y_train_expression)}")
        print(f"    Train indices (first 20): {train_idx[:20]}{'...' if len(train_idx) > 20 else ''}")
        print(f"  Test binding: {len(y_test_binding)}; Test expression: {len(y_test_expression)}")
        print(f"    Test indices (first 20): {test_idx[:20]}{'...' if len(test_idx) > 20 else ''}")
        model.fit(X_train, y_train_binding, y_train_expression)
        y_pred_binding, y_pred_expression = model.predict(X_test)

        seed_results.append({
            "random_seed": seed,
            "binding_r2":r2_score(y_test_binding, y_pred_binding),
            "binding_rmse":np.sqrt(mean_squared_error(y_test_binding, y_pred_binding)),
            "expression_r2":r2_score(y_test_expression, y_pred_expression),
            "expression_rmse":np.sqrt(mean_squared_error(y_test_expression, y_pred_expression)),
        })

    print(f"\n== RESULTS SUMMARY ==")

    # One row per seed
    seed_results_df = pd.DataFrame(seed_results)
    print(f"\nOne row per seed\n{seed_results_df.to_string(index=False)}")

    # Results summary across seeds (mean/std)
    seed_summary_df = pd.DataFrame({
        "mean": seed_results_df.drop(columns=["random_seed"]).mean(), 
        "std": seed_results_df.drop(columns=["random_seed"]).std(ddof=0)
    }).T
    print(f"\nSummary across seeds\n{seed_summary_df.to_string()}")
    
    return seed_results_df, seed_summary_df

def plot_comparison(random_results, stratified_summary, save_as):
    """
    Create visualization comparing random vs stratified splits
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    metrics = ['binding_r2', 'expression_r2']
    titles = ['Binding Affinity R²', 'Expression Level R²']
    
    for ax, metric, title in zip(axes, metrics, titles):
        random_val = random_results[metric]
        stratified_mean = stratified_summary[f'{metric}_mean']
        stratified_std = stratified_summary[f'{metric}_std']
        
        x = ['Random Split', 'Position-Stratified']
        y = [random_val, stratified_mean]
        yerr = [0, stratified_std]
        
        bars = ax.bar(x, y, yerr=yerr, capsize=10, alpha=0.7,
                      color=['tab:blue', 'tab:orange'])
        ax.set_ylabel('R² Score')
        ax.set_title(title)
        ax.set_ylim([-1, 1])
        ax.axhline(y=0, color='black', lw=1, linestyle='-')
        
        # Add value labels on bars
        for bar, val, err in zip(bars, y, yerr):
            height = bar.get_height()

            if height >= 0:
                y_text = height + err + 0.03
                va = 'bottom'
            else:
                y_text = height - err - 0.03
                va = 'top'

            label = f"{val:.3f} ± {err:.3f}" if err > 0 else f"{val:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, y_text,
                    label, ha='center', va=va, zorder=10)
    
    plt.tight_layout()
    save_file = f'plotting_results/splitting_strategy_comparison-{save_as}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f'\nSplit comparison image saved at {save_file}')
    plt.show()

def compare_splitting_strategies(model, df, embeddings, save_as):
    """ Compare random split vs position-stratified split. """
    split_seeds = [0, 1, 2]

    # Random split  
    print("\n" + "=" * 60)
    print("RANDOM SPLIT EVALUATION")
    print("=" * 60)
    r_per_seed_results, r_per_seed_summary = evaluate_random_split(model, df, embeddings, random_seeds=split_seeds)   

    # Position-stratified split
    print("\n" + "=" * 60)
    print("POSITION-STRATIFIED EVALUATION")
    print("=" * 60)
    ps_per_fold_results, ps_per_seed_results, ps_per_seed_summary, ps_all_distribution_df, ps_all_distribution_raw_df = evaluate_position_stratified(model, df, embeddings, n_splits=5, random_seeds=split_seeds)

    ps_all_distribution_df.to_csv("position_stratified_split_mutation_summary.csv", index=False)
    ps_all_distribution_raw_df.to_csv("position_stratified_split_mutation_summary_raw_values.csv", index=False)

    # # Visualize comparison
    # plot_comparison(random_results, summary, save_as)


def main(args):
    # Load your data
    df = pd.read_csv("../../../../data/dms/mutation_combined_DMS_OLD.csv")    
    emb_dict = torch.load(EMB_FILES[args.emb_file], map_location="cpu")
    protein_ids = emb_dict["protein_id"]
    emb_tensor = emb_dict["embeddings"] 

    # protein_id -> row index in embeddings
    id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

    # Align embeddings to df order
    embeddings = np.stack([
        emb_tensor[id_to_idx[row["label"]]].cpu().numpy()
        for _, row in df.iterrows()
    ])

    # Model input
    size = embeddings.shape[1]  # Should match ESM model embedding_dim size
    model = SimpleFcnBeWrapper(
        fcn_input_size=size,  
        fcn_hidden_size=size,
        fcn_num_layers=5,
        lr=args.lr,               
        epochs=args.num_epochs,     
        batch_size=64,      
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_seed=0,
    ) 

    # Run comparison
    print("=" * 60)
    print("INFO")
    print("=" * 60)  
    print(f"Using: ")
    print(f"\tEmbedding file choice {args.emb_file}: {EMB_FILES[args.emb_file]}")
    print(f"\tEpochs: {args.num_epochs}")
    print(f"\tLR: {args.lr}")

    compare_splitting_strategies(
        model=model,
        df=df,
        embeddings=embeddings,
        save_as=f"{args.emb_file}.ep{args.num_epochs}.lr{args.lr}"
    )

if __name__ == "__main__":
    EMB_FILES = {
        "ONE-HOT_mean": "embeddings/DMS_OLD_mean_embeddings-one_hot.pt",
        "ESM2_8M_base_cls": "embeddings/DMS_OLD_cls_base_embeddings-esm2_8m.pt",
        "ESM2_8M_finetuned_cls": "embeddings/DMS_OLD_cls_finetuned_embeddings-esm2_8m.pt",
        "ESM2_8M_base_mean": "embeddings/DMS_OLD_mean_base_embeddings-esm2_8m.pt",
        "ESM2_8M_finetuned_mean": "embeddings/DMS_OLD_mean_finetuned_embeddings-esm2_8m.pt",
        "ESM2_150M_base_cls": "embeddings/DMS_OLD_cls_base_embeddings-esm2_150m.pt",
        "ESM2_150M_finetuned_cls": "embeddings/DMS_OLD_cls_finetuned_embeddings-esm2_150m.pt",
        "ESM2_150M_base_mean": "embeddings/DMS_OLD_mean_base_embeddings-esm2_150m.pt",
        "ESM2_150M_finetuned_mean": "embeddings/DMS_OLD_mean_finetuned_embeddings-esm2_150m.pt",
        "ESM2_650M_base_cls": "embeddings/DMS_OLD_cls_base_embeddings-esm2_650m.pt",
        "ESM2_650M_finetuned_cls": "embeddings/DMS_OLD_cls_finetuned_embeddings-esm2_650m.pt",
        "ESM2_650M_base_mean": "embeddings/DMS_OLD_mean_base_embeddings-esm2_650m.pt",
        "ESM2_650M_finetuned_mean": "embeddings/DMS_OLD_mean_finetuned_embeddings-esm2_650m.pt",
        "ESMC_300M_base_cls": "embeddings/DMS_OLD_cls_base_embeddings-esmc_300m.pt",
        "ESMC_300M_finetuned_cls": "embeddings/DMS_OLD_cls_finetuned_embeddings-esmc_300m.pt",
        "ESMC_300M_base_mean": "embeddings/DMS_OLD_mean_base_embeddings-esmc_300m.pt",
        "ESMC_300M_finetuned_mean": "embeddings/DMS_OLD_mean_finetuned_embeddings-esmc_300m.pt",
        "ESMC_600M_base_cls": "embeddings/DMS_OLD_cls_base_embeddings-esmc_600m.pt",
        "ESMC_600M_finetuned_cls": "embeddings/DMS_OLD_cls_finetuned_embeddings-esmc_600m.pt",
        "ESMC_600M_base_mean": "embeddings/DMS_OLD_mean_base_embeddings-esmc_600m.pt",
        "ESMC_600M_finetuned_mean": "embeddings/DMS_OLD_mean_finetuned_embeddings-esmc_600m.pt"
    }

    parser = argparse.ArgumentParser(description="Run comparison of random vs position-stratified split.")
    parser.add_argument("--emb_file", type=str, choices=list(EMB_FILES.keys()), default="ESM2_8M_base", 
                        help="Sequences embedded by an ESM model or one-hot. (default: %(default)s). Available: %(choices)s")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")

    args = parser.parse_args()
    main(args)