"""
Simulated client node.

Each client holds a local data partition (1D array of floats).
Clients compute local split candidates for the current layer, and can apply the returned global split to update their local partitioning state.

This is a simplified simulation of the clientProcessing() in the paper.
"""

import numpy as np
import random
from typing import Optional, List


class SimClient:
    def __init__(self, data: np.ndarray, rng_seed: Optional[int] = None, subsample_size: Optional[int] = None):
        """
        Initialize client with data partition.
        
        Args:
            data: 1D numpy array of feature values
            rng_seed: Random seed for reproducibility
            subsample_size: If provided, subsample data for each tree (like standard iForest)
        """
        # data should be 1D numpy array of feature values
        self.original_data = np.array(data).astype(float).flatten()
        self.data = self.original_data.copy()
        self.subsample_size = subsample_size
        self.partition_indices = np.arange(len(self.data))  # indices of current partition
        # We keep a stack/queue of partitions for the current tree to simulate splitting across layers:
        self.current_partitions = [self.partition_indices.copy()]
        self.random = random.Random(rng_seed)
        self.rng = np.random.RandomState(rng_seed)
        
    def prepare_for_tree(self):
        """
        Prepare client for building a new tree.
        In standard iForest, each tree uses a random subsample.
        """
        if self.subsample_size is not None and self.subsample_size < len(self.original_data):
            # Subsample data for this tree
            indices = self.rng.choice(len(self.original_data), 
                                     size=min(self.subsample_size, len(self.original_data)), 
                                     replace=False)
            self.data = self.original_data[indices]
            self.partition_indices = np.arange(len(self.data))
        else:
            self.data = self.original_data.copy()
            self.partition_indices = np.arange(len(self.data))
        
        self.current_partitions = [self.partition_indices.copy()]

    def compute_layer_split(self) -> float:
        """
        Compute client's split for the current layer.
        Strategy (following Isolation Forest):
         - For each currently open partition, select a random feature value 
           as split point uniformly between min and max of that partition
         - Average splits across partitions to produce client's split for that layer
        
        This follows the iForest random splitting strategy more closely.
        """
        splits = []
        new_partitions = []
        
        for part in self.current_partitions:
            if len(part) <= 1:
                # no further partitioning possible - skip this partition
                continue
                
            vals = self.data[part]
            minv, maxv = float(np.min(vals)), float(np.max(vals))
            
            if minv == maxv or np.isclose(minv, maxv):
                # no split possible - all values same
                continue
                
            # Random split between min and max (Isolation Forest strategy)
            # Not median-based, but uniform random
            split_point = self.random.uniform(minv, maxv)
            splits.append(split_point)
            
        if len(splits) == 0:
            # fallback: split in global data range
            if len(self.data) > 0:
                minv, maxv = float(np.min(self.data)), float(np.max(self.data))
                if minv == maxv:
                    return minv
                return self.random.uniform(minv, maxv)
            else:
                return 0.0
                
        # client's reported split is mean of partition splits
        return float(np.mean(splits))

    def apply_global_split(self, global_split: float):
        """
        Apply the aggregated global split to each current partition, replacing each partition with its two children.
        This mimics how tree nodes are expanded layer-by-layer.
        """
        new_partitions = []
        for part in self.current_partitions:
            if len(part) <= 1:
                continue
            vals = self.data[part]
            left_mask = vals < global_split
            right_mask = ~left_mask
            left_idx = part[left_mask]
            right_idx = part[right_mask]
            if len(left_idx) > 0:
                new_partitions.append(left_idx)
            if len(right_idx) > 0:
                new_partitions.append(right_idx)
        # If no partition produced, keep current partitions (prevents empty)
        if len(new_partitions) == 0:
            # mark as finished by keeping single-element partitions to prevent further splits
            self.current_partitions = [np.array([i]) for i in self.partition_indices]
        else:
            self.current_partitions = new_partitions

    def reset_for_next_tree(self):
        """Reset partition queue for next tree construction"""
        self.prepare_for_tree()

    def get_local_sample_size(self) -> int:
        return len(self.data)
