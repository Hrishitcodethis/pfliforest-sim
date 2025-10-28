"""
Utilities:
- data generation (temperature + anomalies)
- evaluation helpers (AUC, PR, confusion matrix, threshold selection)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_recall_fscore_support


def generate_temperature_dataset(n_samples=1000, anomaly_fraction=0.1,
                                 normal_mu=25.0, normal_sigma=2.0,
                                 anomaly_low=35.0, anomaly_high=50.0,
                                 rng_seed: int = 42):
    """
    Generate 1D temperature readings with injected anomalies.
    
    Improved version with better separation:
    - Normal data: Gaussian around 25°C with sigma=2
    - Anomalies: Uniform between 35-50°C (clearly separated)
    
    Returns: (X (n,1) array, labels (n,) {0 normal,1 anomaly})
    """
    rng = np.random.RandomState(rng_seed)
    n_anom = int(n_samples * anomaly_fraction)
    n_norm = n_samples - n_anom
    
    # Normal temperatures clustered around normal_mu
    normal = rng.normal(normal_mu, normal_sigma, n_norm)
    
    # Anomalies are clearly separated from normal range
    # Using wider range for better detectability
    anomalies = rng.uniform(anomaly_low, anomaly_high, n_anom)
    
    X = np.concatenate([normal, anomalies]).astype(float)
    labels = np.concatenate([np.zeros(n_norm, dtype=int), np.ones(n_anom, dtype=int)])
    
    # shuffle to mix normal and anomalies
    perm = rng.permutation(n_samples)
    X = X[perm]
    labels = labels[perm]
    
    return X.reshape(-1, 1), labels


def split_to_clients(X: np.ndarray, labels: np.ndarray, n_clients: int, non_iid: bool = False, rng_seed=0):
    """
    Partition data into n_clients.
    If non_iid=True, give each client a shifted mean to simulate heterogeneity.
    Returns list of client data arrays (1D) and corresponding label arrays (1D).
    """
    rng = np.random.RandomState(rng_seed)
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    parts = np.array_split(indices, n_clients)
    client_datas = []
    client_labels = []
    for i, p in enumerate(parts):
        xi = X[p].flatten()
        yi = labels[p].flatten()
        if non_iid:
            # add small shift to simulate local bias
            shift = (i - n_clients / 2) * 0.2 * rng.randn()  # small shift noise
            xi = xi + shift
        client_datas.append(xi)
        client_labels.append(yi)
    return client_datas, client_labels


def choose_threshold_by_f1(y_true, scores):
    """
    Find best threshold on scores by maximizing F1 score.
    Returns best_thresh, best_f1.
    """
    prec, recall, thr = precision_recall_curve(y_true, scores)
    # compute f1 for each threshold
    f1s = (2 * prec * recall) / (prec + recall + 1e-12)
    # precision_recall_curve returns arrays where thr length = len(prec)-1
    if len(f1s) == 0:
        return 0.5, 0.0
    # find max f1 ignoring last entry (which corresponds to thr beyond)
    idx = np.nanargmax(f1s)
    if idx >= len(thr):
        best_t = thr[-1]
    else:
        best_t = thr[idx]
    best_f1 = f1s[idx]
    return float(best_t), float(best_f1)


def evaluate_scores(y_true, scores, threshold=None):
    """
    Evaluate metrics: AUC-ROC, AUC-PR, precision/recall/f1 @ threshold.
    If threshold is None, choose threshold by F1.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
    auc_roc = roc_auc_score(y_true, scores)
    auc_pr = average_precision_score(y_true, scores)
    if threshold is None:
        threshold, _ = choose_threshold_by_f1(y_true, scores)
    preds = (scores >= threshold).astype(int)
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
    return {"auc_roc": auc_roc, "auc_pr": auc_pr, "prec": float(prec), "recall": float(recall), "f1": float(f1), "threshold": float(threshold)}
