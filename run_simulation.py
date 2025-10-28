"""
Extended Federated Isolation Forest Simulation
-------------------------------------------------
Runs multiple experiments, logs metrics, and selects the best model automatically.
"""

import argparse
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import os
from itertools import product
from pfliforest.utils import generate_temperature_dataset, split_to_clients, evaluate_scores
from pfliforest.client import SimClient
from pfliforest.server import FederatedServer
from pfliforest.model import FederatedIsolationTree, FederatedIsolationForest
from sklearn.ensemble import IsolationForest as SKIsolationForest

def simulate_pf_liforest(X, y, n_clients=5, num_trees=25, max_depth=6, subsample_size=256, non_iid=False, rng_seed=0):
    """
    Simulate federated isolation forest.
    
    Args:
        X: Feature array
        y: Labels
        n_clients: Number of federated clients
        num_trees: Number of trees in forest
        max_depth: Maximum depth of each tree
        subsample_size: Subsample size per tree (like in standard iForest)
        non_iid: Whether to create non-iid data distribution
        rng_seed: Random seed
    """
    # Partition data
    client_datas, client_labels = split_to_clients(X, y, n_clients=n_clients, non_iid=non_iid, rng_seed=rng_seed)
    
    # Calculate subsample size per client (proportional to client data size)
    total_samples = sum(len(cd) for cd in client_datas)
    client_subsample_sizes = [max(2, int(subsample_size * len(cd) / total_samples)) for cd in client_datas]
    
    # Create client objects with subsampling
    clients = [SimClient(d, rng_seed + i, subsample_size=ss) 
               for i, (d, ss) in enumerate(zip(client_datas, client_subsample_sizes))]
    server = FederatedServer(client_count=len(clients))

    # Build forest
    trees = []
    rng = np.random.RandomState(rng_seed)
    
    for t in tqdm(range(num_trees), desc="Building trees", disable=False):
        # Prepare clients for new tree (subsample if needed)
        for c in clients:
            c.prepare_for_tree()
            
        splits = []
        
        # For each depth, request splits, aggregate, distribute
        for depth in range(max_depth):
            client_layer_splits = [c.compute_layer_split() for c in clients]
            # Add randomness to aggregation (weighted average with noise)
            weights = rng.dirichlet(np.ones(len(clients)))
            global_split = float(np.average(client_layer_splits, weights=weights))
            
            # apply global split to clients
            for c in clients:
                c.apply_global_split(global_split)
            splits.append(global_split)
        
        # Collect sample data from clients for building tree structure
        # Use a subsample to build tree representation
        tree_sample_data = []
        for c in clients:
            # Get a small sample from each client
            sample_size = min(len(c.data), max(1, subsample_size // (n_clients * 2)))
            if sample_size > 0:
                sample_indices = rng.choice(len(c.data), size=sample_size, replace=False)
                tree_sample_data.extend(c.data[sample_indices])
        
        tree_sample_data = np.array(tree_sample_data)
        
        # finished one tree
        for c in clients:
            c.reset_for_next_tree()
        
        # Build tree with structure
        trees.append(FederatedIsolationTree(splits, sample_data=tree_sample_data))

    # Use subsample_size for normalization (standard iForest uses 256)
    # Total sample size for reference
    train_sample_size = sum([c.get_local_sample_size() for c in clients])

    forest = FederatedIsolationForest(
        trees, 
        train_sample_size=train_sample_size,
        subsample_size=subsample_size
    )
    
    # Evaluate on full dataset
    scores = forest.scores(X.flatten())
    metrics = evaluate_scores(y, scores)
    return forest, metrics, scores


def baseline_isolation_forest(X, y, contamination=0.1):
    clf = SKIsolationForest(contamination=contamination, random_state=0)
    clf.fit(X)
    # sklearn score_samples: higher means more normal, so we invert
    raw = -clf.score_samples(X)
    from pfliforest.utils import evaluate_scores
    metrics = evaluate_scores(y, raw)
    return clf, metrics, raw


def run_experiments(
    n_samples_list=[5000, 10000],
    n_clients_list=[3, 5],
    num_trees_list=[25, 50, 75],
    max_depth_list=[6, 8, 10],
    subsample_size=256,  # Standard iForest subsample size
    anomaly_fraction=0.1,
    metric_to_rank="f1",
    non_iid=False,
    output_file="experiments_results.csv",
):
    """
    Runs all combinations of hyperparameters and logs metrics.
    """

    results = []

    for n_samples, n_clients, num_trees, max_depth in tqdm(
        list(product(n_samples_list, n_clients_list, num_trees_list, max_depth_list)),
        desc="Running Experiments",
    ):
        X, y = generate_temperature_dataset(n_samples=n_samples, anomaly_fraction=anomaly_fraction)
        forest, pf_metrics, pf_scores = simulate_pf_liforest(
            X, y,
            n_clients=n_clients,
            num_trees=num_trees,
            max_depth=max_depth,
            subsample_size=subsample_size,
            non_iid=non_iid,
        )

        row = {
            "n_samples": n_samples,
            "n_clients": n_clients,
            "num_trees": num_trees,
            "max_depth": max_depth,
            "auc_roc": pf_metrics["auc_roc"],
            "auc_pr": pf_metrics["auc_pr"],
            "precision": pf_metrics["prec"],
            "recall": pf_metrics["recall"],
            "f1": pf_metrics["f1"],
            "threshold": pf_metrics["threshold"],
        }
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")

    # Identify best configuration
    best_idx = df[metric_to_rank].idxmax()
    best_config = df.loc[best_idx].to_dict()

    print("\n🏆 Best configuration based on", metric_to_rank, ":")
    for k, v in best_config.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return df, best_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="f1", help="Metric to choose best model (f1, auc_pr, auc_roc)")
    parser.add_argument("--non_iid", action="store_true")
    parser.add_argument("--out", type=str, default="experiments_results.csv")
    args = parser.parse_args()

    df, best = run_experiments(
        metric_to_rank=args.metric,
        non_iid=args.non_iid,
        output_file=args.out,
    )

    # Optional baseline comparison
    print("\n⚖️ Running centralized IsolationForest baseline on best configuration...")
    n_samples, frac = int(best["n_samples"]), 0.1
    from pfliforest.utils import generate_temperature_dataset
    X, y = generate_temperature_dataset(n_samples=n_samples, anomaly_fraction=frac)
    _, base_metrics, _ = baseline_isolation_forest(X, y, contamination=frac)
    print("Baseline iForest metrics:", json.dumps(base_metrics, indent=2))


if __name__ == "__main__":
    main()
