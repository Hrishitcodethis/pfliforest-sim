"""
Test suite for federated isolation forest implementation
"""
import numpy as np
from pfliforest.utils import generate_temperature_dataset, split_to_clients, evaluate_scores
from pfliforest.client import SimClient
from pfliforest.server import FederatedServer
from pfliforest.model import FederatedIsolationTree, FederatedIsolationForest, c_n

def test_c_n_function():
    """Test normalization factor calculation"""
    print("Testing c_n function...")
    assert c_n(1) == 1.0
    assert c_n(2) > 0
    assert c_n(256) > c_n(128)  # Should increase with n
    print("  ✓ c_n function works correctly")

def test_data_generation():
    """Test data generation with proper separation"""
    print("Testing data generation...")
    X, y = generate_temperature_dataset(n_samples=1000, anomaly_fraction=0.1, rng_seed=42)
    assert X.shape == (1000, 1)
    assert y.shape == (1000,)
    assert (y == 1).sum() == 100  # 10% anomalies
    
    # Check separation
    normal_mean = X[y==0].mean()
    anomaly_mean = X[y==1].mean()
    assert anomaly_mean > normal_mean + 5  # At least 5°C separation
    print(f"  ✓ Data generation works (normal: {normal_mean:.1f}°C, anomaly: {anomaly_mean:.1f}°C)")

def test_client_split():
    """Test client split computation"""
    print("Testing client split computation...")
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    client = SimClient(data, rng_seed=42, subsample_size=5)
    client.prepare_for_tree()
    
    split = client.compute_layer_split()
    assert data.min() <= split <= data.max()
    print(f"  ✓ Client split computation works (split: {split:.2f})")

def test_tree_structure():
    """Test tree building with proper structure"""
    print("Testing tree structure...")
    data = np.random.randn(100)
    splits = [0.0, 0.5, 1.0]
    
    tree = FederatedIsolationTree(splits, sample_data=data)
    assert tree.root is not None
    assert tree.max_depth == 3
    
    # Test path length computation
    pl1 = tree.path_length(0.0)
    pl2 = tree.path_length(10.0)  # Far outlier
    assert pl1 > 0
    assert pl2 > 0
    print(f"  ✓ Tree structure works (path lengths: {pl1:.2f}, {pl2:.2f})")

def test_anomaly_scoring():
    """Test anomaly score computation"""
    print("Testing anomaly scoring...")
    X, y = generate_temperature_dataset(n_samples=200, anomaly_fraction=0.1, rng_seed=42)
    
    # Build simple forest
    clients = [SimClient(X.flatten(), 0, subsample_size=50)]
    trees = []
    for t in range(5):
        clients[0].prepare_for_tree()
        splits = []
        for d in range(4):
            split = clients[0].compute_layer_split()
            splits.append(split)
            clients[0].apply_global_split(split)
        trees.append(FederatedIsolationTree(splits, sample_data=X.flatten()[:50]))
        clients[0].reset_for_next_tree()
    
    forest = FederatedIsolationForest(trees, 50, 50)
    scores = forest.scores(X.flatten())
    
    # Check score properties
    assert len(scores) == len(X)
    assert all(0 <= s <= 1 for s in scores)
    
    # Check separation
    normal_scores = scores[y==0]
    anomaly_scores = scores[y==1]
    assert anomaly_scores.mean() > normal_scores.mean()
    print(f"  ✓ Anomaly scoring works (normal: {normal_scores.mean():.4f}, anomaly: {anomaly_scores.mean():.4f})")

def test_end_to_end():
    """Test complete federated learning workflow"""
    print("Testing end-to-end federated learning...")
    X, y = generate_temperature_dataset(n_samples=500, anomaly_fraction=0.1, rng_seed=42)
    
    # Split to clients
    n_clients = 3
    client_datas, _ = split_to_clients(X, y, n_clients=n_clients, rng_seed=42)
    clients = [SimClient(d, i, subsample_size=30) for i, d in enumerate(client_datas)]
    server = FederatedServer(client_count=n_clients)
    
    # Build forest
    trees = []
    rng = np.random.RandomState(42)
    for t in range(10):
        for c in clients:
            c.prepare_for_tree()
        
        splits = []
        for d in range(5):
            client_splits = [c.compute_layer_split() for c in clients]
            global_split = server.aggregate_layer(client_splits)
            for c in clients:
                c.apply_global_split(global_split)
            splits.append(global_split)
        
        # Build tree with sample data
        sample_data = []
        for c in clients:
            if len(c.data) > 0:
                sample_data.extend(c.data[:10])
        trees.append(FederatedIsolationTree(splits, sample_data=np.array(sample_data)))
        
        for c in clients:
            c.reset_for_next_tree()
    
    # Evaluate
    forest = FederatedIsolationForest(trees, len(X), subsample_size=90)
    scores = forest.scores(X.flatten())
    metrics = evaluate_scores(y, scores)
    
    # Check performance
    assert metrics['auc_roc'] > 0.9, f"AUC-ROC too low: {metrics['auc_roc']:.4f}"
    assert metrics['f1'] > 0.8, f"F1-score too low: {metrics['f1']:.4f}"
    print(f"  ✓ End-to-end works (AUC-ROC: {metrics['auc_roc']:.4f}, F1: {metrics['f1']:.4f})")

if __name__ == "__main__":
    print("=" * 70)
    print("Running Federated Isolation Forest Test Suite")
    print("=" * 70)
    print()
    
    try:
        test_c_n_function()
        test_data_generation()
        test_client_split()
        test_tree_structure()
        test_anomaly_scoring()
        test_end_to_end()
        
        print()
        print("=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"❌ Test failed: {e}")
        print("=" * 70)
        raise
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Unexpected error: {e}")
        print("=" * 70)
        raise
