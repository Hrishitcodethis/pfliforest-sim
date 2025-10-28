"""
Simple demo of the improved federated isolation forest
"""
import numpy as np
from pfliforest.utils import generate_temperature_dataset, split_to_clients, evaluate_scores
from pfliforest.client import SimClient
from pfliforest.server import FederatedServer
from pfliforest.model import FederatedIsolationTree, FederatedIsolationForest

print("=" * 70)
print("Federated Isolation Forest - Quick Demo")
print("=" * 70)

# Generate synthetic temperature data
print("\n1️⃣  Generating temperature dataset...")
X, y = generate_temperature_dataset(n_samples=1000, anomaly_fraction=0.1, rng_seed=42)
print(f"   ✓ Generated {len(X)} samples ({(y==1).sum()} anomalies)")
print(f"   ✓ Normal temp range: {X[y==0].min():.1f}°C to {X[y==0].max():.1f}°C")
print(f"   ✓ Anomaly temp range: {X[y==1].min():.1f}°C to {X[y==1].max():.1f}°C")

# Split to federated clients
print("\n2️⃣  Partitioning data across federated clients...")
n_clients = 5
client_datas, client_labels = split_to_clients(X, y, n_clients=n_clients, rng_seed=42)
print(f"   ✓ Data split across {n_clients} clients")
for i, cd in enumerate(client_datas):
    n_anom = (client_labels[i] == 1).sum()
    print(f"   ✓ Client {i+1}: {len(cd)} samples ({n_anom} anomalies)")

# Create clients with subsampling
print("\n3️⃣  Creating federated clients...")
subsample_size_per_client = 40
clients = [SimClient(data, rng_seed=i, subsample_size=subsample_size_per_client) 
           for i, data in enumerate(client_datas)]
print(f"   ✓ Each client uses {subsample_size_per_client} samples per tree")

# Build federated forest
print("\n4️⃣  Building federated isolation forest...")
num_trees = 50
max_depth = 8
server = FederatedServer(client_count=len(clients))
trees = []

for t in range(num_trees):
    # Prepare clients (subsample data)
    for c in clients:
        c.prepare_for_tree()
    
    # Build tree layer by layer
    splits = []
    for depth in range(max_depth):
        # Clients compute local splits
        client_splits = [c.compute_layer_split() for c in clients]
        # Server aggregates
        global_split = server.aggregate_layer(client_splits)
        # Broadcast and apply
        for c in clients:
            c.apply_global_split(global_split)
        splits.append(global_split)
    
    # Collect sample data to build tree structure
    rng = np.random.RandomState(t)
    sample_data = []
    for c in clients:
        if len(c.data) > 0:
            n_samples = min(len(c.data), 20)
            indices = rng.choice(len(c.data), size=n_samples, replace=False)
            sample_data.extend(c.data[indices])
    
    trees.append(FederatedIsolationTree(splits, sample_data=np.array(sample_data)))
    
    # Reset for next tree
    for c in clients:
        c.reset_for_next_tree()

print(f"   ✓ Built {num_trees} trees with max depth {max_depth}")

# Create forest
print("\n5️⃣  Creating forest and computing anomaly scores...")
forest = FederatedIsolationForest(
    trees, 
    train_sample_size=len(X),
    subsample_size=subsample_size_per_client * n_clients
)
scores = forest.scores(X.flatten())
print(f"   ✓ Computed anomaly scores for {len(X)} samples")

# Evaluate
print("\n6️⃣  Evaluating performance...")
metrics = evaluate_scores(y, scores)
print(f"   ✓ AUC-ROC:   {metrics['auc_roc']:.4f}")
print(f"   ✓ AUC-PR:    {metrics['auc_pr']:.4f}")
print(f"   ✓ Precision: {metrics['prec']:.4f}")
print(f"   ✓ Recall:    {metrics['recall']:.4f}")
print(f"   ✓ F1-Score:  {metrics['f1']:.4f}")

# Show example predictions
print("\n7️⃣  Example predictions:")
print(f"   {'Value (°C)':<12} {'True Label':<12} {'Score':<10} {'Prediction'}")
print(f"   {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
for i in np.random.choice(len(X), 10, replace=False):
    val = X[i, 0]
    true_label = 'Anomaly' if y[i] == 1 else 'Normal'
    score = scores[i]
    pred = 'Anomaly' if score >= metrics['threshold'] else 'Normal'
    marker = '✓' if (y[i] == 1) == (score >= metrics['threshold']) else '✗'
    print(f"   {val:<12.2f} {true_label:<12} {score:<10.4f} {pred:<10} {marker}")

print("\n" + "=" * 70)
print("Demo completed successfully!")
print("See README.md and PERFORMANCE_IMPROVEMENTS.md for more details.")
print("=" * 70)
