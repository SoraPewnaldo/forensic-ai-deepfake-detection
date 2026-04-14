"""
Forensic Evaluation Metrics for Forensic-v2.
Implements: AUC, EER, HTER, CLLR, Average Precision.
All functions are pure numpy — no torch dependency at runtime.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


# --------------------------------------------------------------------
# AUC
# --------------------------------------------------------------------
def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC. scores = raw probabilities or LRs (monotone transform)."""
    return float(roc_auc_score(labels, scores))


# --------------------------------------------------------------------
# EER
# --------------------------------------------------------------------
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Equal Error Rate: threshold where FAR == FRR.
    Returns EER in [0, 1].
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    # Interpolate where FPR == FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer


# --------------------------------------------------------------------
# HTER  (Half Total Error Rate)
# --------------------------------------------------------------------
def compute_hter(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
) -> float:
    """
    HTER = (FAR + FRR) / 2 at the given threshold.
    If threshold is None, uses the EER threshold (threshold-free equivalent).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    if threshold is None:
        # Use EER operating point
        idx = np.nanargmin(np.abs(fpr - fnr))
    else:
        idx = np.nanargmin(np.abs(thresholds - threshold))

    far = float(fpr[idx])
    frr = float(fnr[idx])
    return (far + frr) / 2.0


# --------------------------------------------------------------------
# CLLR  (Cost of Log-Likelihood Ratio)
# --------------------------------------------------------------------
def compute_cllr(lr_values: np.ndarray, labels: np.ndarray) -> float:
    """
    CLLR = (C_llr_real + C_llr_fake) / 2

    Where:
        C_llr_real = (1/N_r) * sum log2(1 + LR_i)      for real samples
        C_llr_fake = (1/N_f) * sum log2(1 + 1/LR_i)    for fake samples

    LR values < 1 → evidence for real.
    LR values > 1 → evidence for fake.
    Target: CLLR < 0.4 for forensic admissibility.
    """
    lr = np.clip(lr_values.astype(np.float64), 1e-9, 1e9)
    real_mask = (labels == 0)
    fake_mask = (labels == 1)

    if real_mask.sum() == 0 or fake_mask.sum() == 0:
        raise ValueError("CLLR requires both real and fake samples.")

    c_real = np.mean(np.log2(1.0 + lr[real_mask]))
    c_fake = np.mean(np.log2(1.0 + 1.0 / lr[fake_mask]))

    return float((c_real + c_fake) / 2.0)


# --------------------------------------------------------------------
# Average Precision
# --------------------------------------------------------------------
def compute_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(average_precision_score(labels, scores))


# --------------------------------------------------------------------
# All-in-one summary
# --------------------------------------------------------------------
def compute_all_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    lr_values: np.ndarray | None = None,
) -> dict:
    """
    Returns a dict with AUC, EER, HTER, AP, and optionally CLLR.
    lr_values must be supplied for CLLR (output of IsotonicCalibrator.predict_lr).
    """
    result = {
        "auc":  compute_auc(scores, labels),
        "eer":  compute_eer(scores, labels),
        "hter": compute_hter(scores, labels),
        "ap":   compute_ap(scores, labels),
    }
    if lr_values is not None:
        result["cllr"] = compute_cllr(lr_values, labels)

    return result
