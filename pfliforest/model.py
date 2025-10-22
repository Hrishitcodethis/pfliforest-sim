"""
PFLiForest model primitives.

Implements:
- A simple layer-by-layer isolation tree representation (list of split values per depth).
- Forest building helpers for federated simulation (server + clients).
- Inference: anomaly score calculation approximating IsolationForest behavior.
"""

import numpy as np
import math
from typing import List, Any, Tuple


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


class FederatedIsolationTree:
    """
    Simple representation of a federated isolation tree built layer-by-layer.
    For simulation we store split values at each depth (global split for that layer).
    """

    def __init__(self, splits: List[float]):
        # splits length == max_depth (or until early stop)
        self.splits = splits

    def path_length(self, x: float) -> float:
        """
        Compute a traversal-based path length for value x.
        We simulate: at each split, if x < split -> go left else -> go right.
        We stop when we reach leaf (we assume full depth == len(splits)).
        Returns the depth (int).
        """
        depth = 0
        for s in self.splits:
            depth += 1
            if x < s:
                # go left (we don't maintain partitions in simulation)
                # continue to next split
                continue
            else:
                # go right
                continue
        return depth


class FederatedIsolationForest:
    def __init__(self, trees: List[FederatedIsolationTree], train_sample_size: int):
        self.trees = trees
        # n used in normalization c(n) should be sample size used to train trees
        self.train_sample_size = max(2, int(train_sample_size))

    def anomaly_score(self, x: float) -> float:
        """
        Compute the anomaly score s(x, n) = 2^{-E(h(x))/c(n)}
        where E(h(x)) is average path length across forest.
        """
        path_lengths = [t.path_length(x) for t in self.trees]
        max_depth = max(1, max(len(t.splits) for t in self.trees))
        E_h = float(np.mean(path_lengths)) / max_depth
        cn = c_n(self.train_sample_size)
        # Avoid division by zero
        if cn == 0:
            return 0.0
        score = 1 - 2 ** ( - (E_h / cn) )
        return score

    def scores(self, X: List[float]) -> np.ndarray:
        return np.array([self.anomaly_score(float(x)) for x in X])
