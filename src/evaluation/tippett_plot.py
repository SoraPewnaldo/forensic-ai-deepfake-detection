"""
Tippett Plot generator for Forensic-v2.
Mandatory output for court-admissible LR reporting.

A Tippett plot shows:
  - CDF of log10(LR) for real samples (should be left-shifted)
  - CDF of log10(LR) for fake samples (should be right-shifted)
  - Ideal discrimination  separation around log10(LR)=0
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless/Windows-safe
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.logging_utils import logger


def plot_tippett(
    lr_values: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str = "Tippett Plot",
) -> None:
    """
    Generates and saves a Tippett plot.

    Args:
        lr_values:   LR from IsotonicCalibrator.predict_lr(), shape [N]
        labels:      Binary ground-truth (0=real, 1=fake), shape [N]
        output_path: Where to save the .png
        title:       Plot title (include dataset name for reports)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_lr = np.log10(np.clip(lr_values, 1e-9, 1e9))
    real_log_lr = np.sort(log_lr[labels == 0])
    fake_log_lr = np.sort(log_lr[labels == 1])

    # CDF
    def _cdf(arr):
        x = np.sort(arr)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    rx, ry = _cdf(real_log_lr)
    fx, fy = _cdf(fake_log_lr)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rx, ry, color="#3a7ebf", lw=2, label="Real (H0)")
    ax.plot(fx, fy, color="#d94f3d", lw=2, label="Fake (H1)")
    ax.axvline(x=0, color="grey", linestyle="--", lw=1, label="LR=1 (chance)")

    ax.set_xlabel("log10(LR)", fontsize=12)
    ax.set_ylabel("Cumulative Proportion", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Tippett plot saved: {output_path}")
