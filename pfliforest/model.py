"""
PFLiForest model primitives.

Implements:
- A layer-by-layer isolation tree with proper binary tree structure
- Forest building helpers for federated simulation (server + clients)
- Inference: anomaly score calculation following IsolationForest behavior.
"""

import numpy as np
import math
from typing import List, Any, Tuple, Optional


def c_n(n: int) -> float:
    """
    Normalization factor c(n) used in Isolation Forest scoring.
    Approximated from the paper:
      c(n) = 2 * H(n-1) - 2*(n-1)/n
    where H(m) is harmonic number approximated by ln(m) + gamma.
    """
    if n <= 1:
        return 1.0
    H = math.log(n - 1) + 0.5772156649  # Euler's constant approx
    return 2.0 * H - (2.0 * (n - 1) / n)


class TreeNode:
    """A node in the isolation tree"""
    def __init__(self, split_value: Optional[float] = None, size: int = 0):
        self.split_value = split_value
        self.size = size  # number of samples in this node
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None
        self.is_leaf = split_value is None


class FederatedIsolationTree:
    """
    Isolation tree with proper binary structure, built layer-by-layer in federated setting.
    """

    def __init__(self, splits: List[float], sample_data: Optional[np.ndarray] = None):
        """
        Build tree structure from layer-by-layer splits.
        
        Args:
            splits: Split values at each depth
            sample_data: Representative sample data to build tree structure
        """
        self.splits = splits
        self.max_depth = len(splits)
        self.root = None
        
        # Build tree structure if sample data provided
        if sample_data is not None and len(sample_data) > 0:
            self.root = self._build_tree_structure(sample_data, 0)
    
    def _build_tree_structure(self, data: np.ndarray, depth: int) -> TreeNode:
        """
        Recursively build tree structure using splits.
        """
        # Base cases
        if depth >= self.max_depth or len(data) <= 1:
            return TreeNode(split_value=None, size=len(data))
        
        split_val = self.splits[depth]
        node = TreeNode(split_value=split_val, size=len(data))
        
        # Partition data
        left_mask = data < split_val
        right_mask = ~left_mask
        
        left_data = data[left_mask]
        right_data = data[right_mask]
        
        # Build children
        if len(left_data) > 0:
            node.left = self._build_tree_structure(left_data, depth + 1)
        else:
            node.left = TreeNode(split_value=None, size=0)
            
        if len(right_data) > 0:
            node.right = self._build_tree_structure(right_data, depth + 1)
        else:
            node.right = TreeNode(split_value=None, size=0)
        
        return node

    def path_length(self, x: float) -> float:
        """
        Compute path length for value x by traversing tree.
        """
        if self.root is None:
            # Fallback: simple depth-based calculation
            return self._fallback_path_length(x)
        
        return self._traverse(x, self.root, 0)
    
    def _traverse(self, x: float, node: TreeNode, depth: int) -> float:
        """
        Recursively traverse tree to find path length.
        """
        # Reached leaf or max depth
        if node.is_leaf or depth >= self.max_depth:
            # Add adjustment c(n) for unsplit nodes
            adjustment = c_n(max(1, node.size)) if node.size > 1 else 0
            return depth + adjustment
        
        # Continue traversal
        if x < node.split_value:
            if node.left is not None:
                return self._traverse(x, node.left, depth + 1)
            else:
                return depth + 1 + c_n(1)
        else:
            if node.right is not None:
                return self._traverse(x, node.right, depth + 1)
            else:
                return depth + 1 + c_n(1)
    
    def _fallback_path_length(self, x: float) -> float:
        """
        Fallback path length calculation when tree structure not available.
        """
        depth = 0
        for split in self.splits:
            depth += 1
        return depth + c_n(2)


class FederatedIsolationForest:
    def __init__(self, trees: List[FederatedIsolationTree], train_sample_size: int, subsample_size: int = 256):
        self.trees = trees
        # n used in normalization c(n) should be subsample size per tree (default 256 in iForest)
        # NOT the total training set size
        self.subsample_size = min(subsample_size, train_sample_size) if train_sample_size > 0 else 256
        self.train_sample_size = max(2, int(train_sample_size))

    def anomaly_score(self, x: float) -> float:
        """
        Compute the anomaly score s(x, n) = 2^{-E(h(x))/c(n)}
        where E(h(x)) is average path length across forest.
        
        Higher scores indicate anomalies.
        Score close to 1 = anomaly (short average path)
        Score close to 0 = normal (long average path)
        """
        if len(self.trees) == 0:
            return 0.0
            
        path_lengths = [t.path_length(x) for t in self.trees]
        E_h = float(np.mean(path_lengths))
        
        # Use subsample size for normalization, not total training size
        cn = c_n(self.subsample_size)
        
        # Avoid division by zero
        if cn == 0 or cn < 1e-10:
            cn = 1.0
            
        # Correct formula: s(x,n) = 2^(-E(h(x))/c(n))
        score = 2.0 ** (-(E_h / cn))
        
        return score

    def scores(self, X: List[float]) -> np.ndarray:
        return np.array([self.anomaly_score(float(x)) for x in X])

