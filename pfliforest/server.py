"""
Federated server orchestrator (simulated).

Responsible for:
- Requesting layer-splits from clients
- Aggregating splits (mean)
- Constructing global tree layers
- Building a forest by repeating tree construction
This code is synchronous for clarity. You can adapt to async for network simulation.
"""

from typing import List, Dict, Any
import numpy as np
from pfliforest.model import FederatedIsolationTree, FederatedIsolationForest


class FederatedServer:
    def __init__(self, client_count: int, aggregation_fn=None):
        self.client_count = client_count
        self.aggregation_fn = aggregation_fn or (lambda arr: float(np.mean(arr)))

    def aggregate_layer(self, client_splits: List[float]) -> float:
        """
        Aggregate split values from clients for one layer.
        Default: arithmetic mean.
        """
        if len(client_splits) == 0:
            raise ValueError("No client splits to aggregate")
        return self.aggregation_fn(client_splits)

    def build_tree(self, client_datasets: List[np.ndarray], max_depth: int) -> FederatedIsolationTree:
        """
        Build a single tree layer-by-layer:
        - For each depth:
           - ask each client for a split computed from its local partition
           - aggregate splits into global split for layer
        - Return FederatedIsolationTree
        """
        splits = []
        # For each layer, ask clients for their split value computed on their local view.
        for depth in range(max_depth):
            client_values = []
            for cd in client_datasets:
                # Each client provides a single split for current layer.
                # In the simulation the client provides a numeric split.
                # We'll call a method on the client dataset to compute split.
                # But here, client_datasets holds arrays; higher-level code will call client API.
                raise NotImplementedError("Server.build_tree should be used through build_forest() where clients are invoked.")
        return FederatedIsolationTree(splits)

    def build_forest(self, client_objects, num_trees: int, max_depth: int, train_sample_size: int):
        """
        Build an entire forest by repeatedly building trees.
        client_objects: list of client instances which expose `compute_layer_split(partition)` method.
        """
        trees = []
        for t in range(num_trees):
            splits = []
            # client_objects provide split at each layer; we aggregate per layer
            for depth in range(max_depth):
                client_layer_splits = []
                for c in client_objects:
                    split = c.compute_layer_split()
                    client_layer_splits.append(split)
                # aggregate
                global_split = self.aggregate_layer(client_layer_splits)
                # Distribute global_split back to clients so they can update local partitions
                for c in client_objects:
                    c.apply_global_split(global_split)
                splits.append(global_split)
            tree = FederatedIsolationTree(splits)
            trees.append(tree)
            # tell clients to reset partitions for next tree
            for c in client_objects:
                c.reset_for_next_tree()
        forest = FederatedIsolationForest(trees, train_sample_size=train_sample_size)
        return forest
