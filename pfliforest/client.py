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
    def __init__(self, data: np.ndarray, rng_seed: Optional[int] = None):
        # data should be 1D numpy array of feature values
        self.data = np.array(data).astype(float).flatten()
        self.partition_indices = np.arange(len(self.data))  # indices of current partition
        # We keep a stack/queue of partitions for the current tree to simulate splitting across layers:
        self.current_partitions = [self.partition_indices.copy()]
        self.random = random.Random(rng_seed)

    def compute_layer_split(self) -> float:
        """
        Compute client's split for the current layer.
        Strategy:
         - For simplicity: for every currently open partition pick random split in [min, max] of partition.
         - Then average splits across partitions to produce client's split for that layer.
        """
        splits = []
        new_partitions = []
        for part in self.current_partitions:
            if len(part) <= 1:
                # no further partitioning possible
                continue
            vals = self.data[part]
            minv, maxv = float(np.min(vals)), float(np.max(vals))
            if minv == maxv:
                # no split possible
                continue
            median = np.median(vals)
            jitter = self.random.uniform(-0.2, 0.2) * (maxv - minv)
            s = float(np.clip(median + jitter, minv, maxv))
            splits.append(s)
        if len(splits) == 0:
            # fallback: split in global data min/max
            minv, maxv = float(np.min(self.data)), float(np.max(self.data))
            if minv == maxv:
                return minv
            return self.random.uniform(minv, maxv)
        # client's reported split is mean of partition splits (simplifying assumption)
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
        self.current_partitions = [self.partition_indices.copy()]

    def get_local_sample_size(self) -> int:
        return len(self.data)
