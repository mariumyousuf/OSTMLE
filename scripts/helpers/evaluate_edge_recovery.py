import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, matthews_corrcoef
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr, pearsonr

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

def compute_metrics(W_gt, W_est):
    # Flatten the matrices
    gt_flat = W_gt.flatten()
    est_flat = W_est.flatten()

    # Mask negative or zero values before log transformation
    # Replace non-positive values with a small positive number (e.g., 1e-6)
    gt_flat_safe = np.where(gt_flat > 0, gt_flat, 1e-6)
    est_flat_safe = np.where(est_flat > 0, est_flat, 1e-6)

    # Apply log(1+x) to ground truth and estimate
    gt_log = np.log1p(gt_flat_safe)  # log(1 + gt_flat_safe)
    est_log = np.log1p(est_flat_safe)  # log(1 + est_flat_safe)

    # Remove any remaining NaN or Inf values (e.g., due to extreme values)
    valid_indices = np.isfinite(gt_log) & np.isfinite(est_log)
    gt_log = gt_log[valid_indices]
    est_log = est_log[valid_indices]

    # AUPRC: Convert the ground truth into binary format
    gt_binary = (gt_flat > 0).astype(int)  # Binary ground truth (0 or 1)
    
    # Use the predicted values directly as scores
    auprc = average_precision_score(gt_binary, est_flat)

    # Spearman Rank Correlation: Measures rank-order correlation between the matrices
    # Check if the ground truth or the estimated values are constant
    if np.std(gt_flat) == 0 or np.std(est_flat) == 0:
        spearman_corr = np.nan  # Set as NaN if there's no variability
    else:
        spearman_corr, _ = spearmanr(gt_flat, est_flat)

    # Log-space Pearson Correlation: Measures Pearson correlation on log-transformed values
    # Check if the log-transformed values are constant
    if np.std(gt_log) == 0 or np.std(est_log) == 0:
        pearson_corr = np.nan  # Set as NaN if there's no variability
    else:
        pearson_corr, _ = pearsonr(gt_log, est_log)

    # Return the metrics
    metrics = {
        'AUPRC': auprc,
        'Spearman Rank Correlation': spearman_corr,
        'Log-space Pearson Correlation': pearson_corr
    }

    return metrics


# def compute_metrics(W_gt, W_est):
#     # Flatten the matrices
#     gt_flat = W_gt.flatten()
#     est_flat = W_est.flatten()

#     # Mask negative or zero values before log transformation
#     # Replace non-positive values with a small positive number (e.g., 1e-6)
#     gt_flat_safe = np.where(gt_flat > 0, gt_flat, 1e-6)
#     est_flat_safe = np.where(est_flat > 0, est_flat, 1e-6)

#     # Apply log(1+x) to ground truth and estimate
#     gt_log = np.log1p(gt_flat_safe)  # log(1 + gt_flat_safe)
#     est_log = np.log1p(est_flat_safe)  # log(1 + est_flat_safe)

#     # Remove any remaining NaN or Inf values (e.g., due to extreme values)
#     valid_indices = np.isfinite(gt_log) & np.isfinite(est_log)
#     gt_log = gt_log[valid_indices]
#     est_log = est_log[valid_indices]

#     # AUPRC: Convert the ground truth into binary format
#     # Treat any non-zero value as a connection (label = 1), otherwise label = 0
#     gt_binary = (gt_flat > 0).astype(int)  # Binary ground truth (0 or 1)
    
#     # Use the predicted values directly as scores
#     auprc = average_precision_score(gt_binary, est_flat)

#     # Spearman Rank Correlation: Measures rank-order correlation between the matrices
#     spearman_corr, _ = spearmanr(gt_flat, est_flat)

#     # Log-space Pearson Correlation: Measures Pearson correlation on log-transformed values
#     pearson_corr, _ = pearsonr(gt_log, est_log)

#     # Return the metrics
#     metrics = {
#         'AUPRC': auprc,
#         'Spearman Rank Correlation': spearman_corr,
#         'Log-space Pearson Correlation': pearson_corr
#     }

#     return metrics

# def compute_metrics(W_gt, W_est):
#     """
#     Compute the following metrics between the ground truth matrix (W_gt) and the estimated matrix (W_est):
#     - AUPRC (Area Under Precision-Recall Curve)
#     - Spearman Rank Correlation
#     - Log-space Pearson Correlation
    
#     Parameters:
#     - W_gt (numpy array): Ground truth matrix (0 or positive values)
#     - W_est (numpy array): Estimated matrix (continuous values)
    
#     Returns:
#     - dict: Dictionary containing AUPRC, Spearman correlation, and Log-space Pearson correlation
#     """
    
#     # Ensure W_gt and W_est are numpy arrays
#     W_gt = np.array(W_gt)
#     W_est = np.array(W_est)

#     # Flatten matrices to 1D arrays for comparison
#     gt_flat = W_gt.flatten()
#     est_flat = W_est.flatten()
    
#     # 1. AUPRC (Area Under Precision-Recall Curve) on non-zero elements
#     # Consider only the non-zero elements (i.e., actual connections)
#     gt_binary = (gt_flat > 0).astype(int)  # Convert to binary labels: 0 for no connection, 1 for connection
#     est_nonzero = est_flat[gt_binary > 0]  # Only take the estimated values where ground truth is non-zero
    
#     # AUPRC: We use `average_precision_score`, which is equivalent to AUPRC for binary data
#     auprc = average_precision_score(gt_binary, est_flat)
    
#     # 2. Spearman Rank Correlation
#     spearman_corr, _ = spearmanr(gt_flat, est_flat)
    
#     # 3. Log-space Pearson Correlation (log(1+x) transformation)
#     # Handle zeros (log(1+x) is safe even when x=0)
#     gt_log = np.log1p(gt_flat)  # log(1+x)
#     est_log = np.log1p(est_flat)
    
#     pearson_corr, _ = pearsonr(gt_log, est_log)
    
#     metrics = {
#         'AUPRC': auprc,
#         'Spearman Rank Correlation': spearman_corr,
#         'Log-space Pearson Correlation': pearson_corr
#     }

#     return metrics


def evaluate_edge_recovery(A_true, A_est, exclude_diagonal=True, threshold_gt=0.0):
    """
    Evaluate network structure recovery between ground-truth and estimated adjacency matrices.
    
    Parameters
    ----------
    A_true : np.ndarray (p, p)
        Ground-truth adjacency matrix (can be binary or weighted).
    A_est : np.ndarray (p, p)
        Estimated adjacency matrix (scores or weights).
    exclude_diagonal : bool, default True
        Whether to ignore self-loops.
    threshold_gt : float, default 0.0
        Threshold to binarize ground-truth edges (> threshold_gt counts as edge).
    threshold_est : float or None
        Threshold to binarize estimated edges. If None, uses ranking/top_k.
    top_k : int or None
        If specified, only the top_k highest scoring estimated edges are considered positive.
    
    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'AUPRC', 'precision', 'recall', 'pr_thresholds'
        - 'Precision', 'Recall', 'F1', 'MCC', 'Jaccard', 'Hamming'
    """
    p = A_true.shape[0]
    assert A_true.shape == A_est.shape, "Shape mismatch"

    # Mask to exclude diagonal
    mask = np.ones((p, p), dtype=bool)
    if exclude_diagonal:
        np.fill_diagonal(mask, False)

    # Flatten and mask matrices
    y_true = (A_true[mask] > threshold_gt).astype(int)
    y_score = A_est[mask]

    # ---------- Ranking metrics ----------
    auprc = average_precision_score(y_true, y_score)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)

    # ---------- Thresholding for classification ----------
    y_est = (y_score > 0).astype(int)

    # True positives / false positives / false negatives / true negatives
    TP = np.sum((y_true == 1) & (y_est == 1))
    FP = np.sum((y_true == 0) & (y_est == 1))
    FN = np.sum((y_true == 1) & (y_est == 0))
    TN = np.sum((y_true == 0) & (y_est == 0))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    MCC = matthews_corrcoef(y_true, y_est) if len(np.unique(y_true)) > 1 else np.nan
    Jaccard = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    Hamming = (FP + FN) / len(y_true)

    metrics = {
        'AUPRC': auprc,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'pr_thresholds': pr_thresholds,
        'Precision': precision,
        'Recall': recall,
        'F1': F1,
        'MCC': MCC,
        'Jaccard': Jaccard,
        'Hamming': Hamming,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN
    }

    return metrics
