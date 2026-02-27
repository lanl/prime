#!/usr/bin/env python
"""
Testing random split vs position-stratified split. 

To isolate the impact of data partitioning, we compared a standard random split 
and a position-stratified split generated within the same evaluation framework 
using frozen ESM embeddings and an identical regression head. Note, this random 
split is different from how we originally random split the data, so the values 
do NOT match!
"""
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
    def __init__(self, fcn_input_size, fcn_hidden_size, fcn_num_layers, lr, epochs, device):
        super().__init__()
        self.fcn_input_size = fcn_input_size
        self.fcn_hidden_size = fcn_hidden_size
        self.fcn_num_layers = fcn_num_layers
        self.lr = lr
        self.epochs = epochs
        self.device = device
        
        self.model = FCN_BE(self.fcn_input_size, self.fcn_hidden_size, self.fcn_num_layers).to(self.device)
        self.loss_fn = torch.nn.MSELoss()

    def fit(self, X, y_binding, y_expression):
        batch_size = 64
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        yb = torch.tensor(y_binding, dtype=torch.float32, device=self.device)
        ye = torch.tensor(y_expression, dtype=torch.float32, device=self.device)

        self.model.train()
        for _ in tqdm.trange(self.epochs, desc="Training epochs"):
            perm = torch.randperm(len(X))   # Create a random permutation of indices, basically DataLoader(shuffle=True)
            for i in range(0, len(X), batch_size):
                idx = perm[i:i+batch_size]
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
        return SimpleFcnBeWrapper(self.fcn_input_size, self.fcn_hidden_size, self.fcn_num_layers, self.lr, self.epochs, self.device)
    
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

        Returns: List of (train_idx, test_idx) tuples
        """
        # Identify all mutated positions for each sequence
        if 'mutated_positions' not in df.columns:
            df = df.copy()
            df['mutated_positions'] = df.apply(
                lambda row: self._extract_mutated_positions(row['sequence']),
                axis=1
            )
        
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
            train_positions = set([all_positions[i] for i in train_positions_idx])
            test_positions = set([all_positions[i] for i in test_positions_idx])
            
            # Assign sequences to train/test based on their mutated positions
            # Important: If a sequence has mutations at multiple positions,
            # decide based on whether ANY position is in test set
            train_mask = df['mutated_positions'].apply(
                lambda pos_list: all(pos in train_positions for pos in pos_list)
            )
            test_mask = df['mutated_positions'].apply(
                lambda pos_list: any(pos in test_positions for pos in pos_list) and 
                                    all(pos not in train_positions or pos in test_positions for pos in pos_list)
            )
            
            train_idx = df[train_mask].index.tolist()
            test_idx = df[test_mask].index.tolist()
            
            fold_splits.append((train_idx, test_idx))
            
            print(f"Fold {fold_idx + 1}:")
            print(f"  Train positions: {len(train_positions)} | Train sequences: {len(train_idx)}")
            print(f"  Test positions: {len(test_positions)} | Test sequences: {len(test_idx)}")
        
        return fold_splits

def evaluate_position_stratified(model, df, embeddings, n_splits=5):
    """
    Evaluate model performance using position-stratified cross-validation
    
    Args:
        model: Your trained model class
        df: DataFrame with DMS data
        embeddings: Pre-computed ESM embeddings
        n_splits: Number of cross-validation folds
    
    Returns:
        Dictionary with performance metrics
    """
    ps_splitter = PositionStratifiedSplitter(
        wt_sequence = "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST", 
        n_splits=n_splits,
        random_state = 42
    )
    
    fold_splits = ps_splitter.split(df)
    
    results = {
        'binding_r2': [],
        'binding_rmse': [],
        'expression_r2': [],
        'expression_rmse': []
    }
    
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        print(f"\nTraining fold {fold_idx + 1}/{n_splits}...")
        
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
        
        # Calculate metrics
        results['binding_r2'].append(r2_score(y_test_binding, y_pred_binding))
        results['binding_rmse'].append(np.sqrt(mean_squared_error(y_test_binding, y_pred_binding)))
        results['expression_r2'].append(r2_score(y_test_expression, y_pred_expression))
        results['expression_rmse'].append(np.sqrt(mean_squared_error(y_test_expression, y_pred_expression)))
    
    # Calculate mean and std across folds
    summary = {
        'binding_r2_mean': np.mean(results['binding_r2']),
        'binding_r2_std': np.std(results['binding_r2']),
        'binding_rmse_mean': np.mean(results['binding_rmse']),
        'binding_rmse_std': np.std(results['binding_rmse']),
        'expression_r2_mean': np.mean(results['expression_r2']),
        'expression_r2_std': np.std(results['expression_r2']),
        'expression_rmse_mean': np.mean(results['expression_rmse']),
        'expression_rmse_std': np.std(results['expression_rmse'])
    }
    
    return results, summary

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
    save_file = f'splitting_strategy_comparison-{save_as}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f'\nSplit comparison image saved at {save_file}')
    plt.show()

def compare_splitting_strategies(model, df, embeddings, save_as):
    """
    Compare random split vs position-stratified split.
    """
    print("\n" + "=" * 60)
    print("RANDOM SPLIT EVALUATION")
    print("=" * 60)
    
    # Random split
    train_idx, test_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=42
    )
    
    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]
    y_train_binding = df.iloc[train_idx]['ACE2-binding_affinity'].values
    y_test_binding = df.iloc[test_idx]['ACE2-binding_affinity'].values
    y_train_expression = df.iloc[train_idx]['RBD_expression'].values
    y_test_expression = df.iloc[test_idx]['RBD_expression'].values

    print(f"train binding: {len(y_train_binding)}; test binding: {len(y_test_binding)}")
    print(f"train expression: {len(y_train_expression)}; test expression: {len(y_test_expression)}")
    
    model.fit(X_train, y_train_binding, y_train_expression)
    y_pred_binding, y_pred_expression = model.predict(X_test)
    
    random_results = {
        'binding_r2': r2_score(y_test_binding, y_pred_binding),
        'binding_rmse': np.sqrt(mean_squared_error(y_test_binding, y_pred_binding)),
        'expression_r2': r2_score(y_test_expression, y_pred_expression),
        'expression_rmse': np.sqrt(mean_squared_error(y_test_expression, y_pred_expression))
    }
    
    print(f"Binding - R²: {random_results['binding_r2']:.4f}, RMSE: {random_results['binding_rmse']:.4f}")
    print(f"Expression - R²: {random_results['expression_r2']:.4f}, RMSE: {random_results['expression_rmse']:.4f}")
    
    print("\n" + "=" * 60)
    print("POSITION-STRATIFIED EVALUATION")
    print("=" * 60)
    
    # Position-stratified split
    results, summary = evaluate_position_stratified(model, df, embeddings, n_splits=5)
    
    print(f"\nBinding - R²: {summary['binding_r2_mean']:.4f} ± {summary['binding_r2_std']:.4f}")
    print(f"          RMSE: {summary['binding_rmse_mean']:.4f} ± {summary['binding_rmse_std']:.4f}")
    print(f"Expression - R²: {summary['expression_r2_mean']:.4f} ± {summary['expression_r2_std']:.4f}")
    print(f"          RMSE: {summary['expression_rmse_mean']:.4f} ± {summary['expression_rmse_std']:.4f}")
    
    # Visualize comparison
    plot_comparison(random_results, summary, save_as)

if __name__ == "__main__":
    EMB_FILES = {
        "ESM2_8M_base": "embeddings/DMS_OLD_cls_base_embeddings-esm2_8m.pt",
        "ESM2_8M_finetuned": "embeddings/DMS_OLD_cls_finetuned_embeddings-esm2_8m.pt",
        "ESM2_150M_base": "embeddings/DMS_OLD_cls_base_embeddings-esm2_150m.pt",
        "ESM2_150M_finetuned": "embeddings/DMS_OLD_cls_finetuned_embeddings-esm2_150m.pt",
        "ESM2_650M_base": "embeddings/DMS_OLD_cls_base_embeddings-esm2_650m.pt",
        "ESM2_650M_finetuned": "embeddings/DMS_OLD_cls_finetuned_embeddings-esm2_650m.pt",
        "ESMC_300M_base": "embeddings/DMS_OLD_cls_base_embeddings-esmc_300m.pt",
        "ESMC_300M_finetuned": "embeddings/DMS_OLD_cls_finetuned_embeddings-esmc_300m.pt",
        "ESMC_600M_base_cls": "embeddings/DMS_OLD_cls_base_embeddings-esmc_600m.pt",
        "ESMC_600M_finetuned_cls": "embeddings/DMS_OLD_cls_finetuned_embeddings-esmc_600m.pt",
        "ESMC_600M_base_mean": "embeddings/DMS_OLD_mean_base_embeddings-esmc_600m.pt",
        "ESMC_600M_finetuned_mean": "embeddings/DMS_OLD_mean_finetuned_embeddings-esmc_600m.pt"
    }

    parser = argparse.ArgumentParser(description="Run ESM comparison of random vs position-stratified split.")
    parser.add_argument("--emb_file", type=str, choices=list(EMB_FILES.keys()), default="ESM2_8M_base", 
                        help="Sequences embedded by an ESM model. (default: %(default)s). Available: %(choices)s")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    args = parser.parse_args()

    # Random seed
    torch.manual_seed(0)
    np.random.seed(0)

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
        device="cuda" if torch.cuda.is_available() else "cpu",
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