"""
Full Calibration Pipeline for Forensic-v2 — Multi-Domain Hybrid Calibration.

Strategy:
  Calibration is fitted on a HYBRID set spanning three domains:
    1. FF++ CAL partition      — studio-quality, controlled compression
    2. Celeb-DF CAL partition  — celebrity videos, diverse encodings
    3. WildDeepfake CAL subset — in-the-wild, noisy internet video

  This ensures the calibrator produces reliable Likelihood Ratios (LR)
  across all deployment domains, not just the training domain.

Data Integrity:
  All three CAL partitions are STRICTLY DISJOINT from training, validation,
  proxy validation, and the final test sets. No leakage is possible.

Steps:
  1. Collect raw logits from all three CAL loaders.
  2. Z-score normalize each domain's logits separately (aligns score distributions).
  3. Concatenate + shuffle.
  4. Fit Temperature Scaling (minimise NLL via LBFGS).
  5. Fit Isotonic Regression (PAVA) on T-scaled probabilities.
  6. Save artefacts. Report per-domain diagnostics + CLLR.
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.config import config
from src.utils.seed import set_global_seed
from src.utils.logging_utils import logger
from src.utils.checkpoint import load_checkpoint
from src.models import ForensicModel
from src.datasets import FFPPDataset
from src.datasets.celebdf_dataset import CelebDFDataset
from src.datasets.wild_dataset import WildDeepfakeDataset
from src.calibration.temperature_scaling import fit_temperature, save_temperature
from src.calibration.isotonic_calibrator import IsotonicCalibrator


_CAL_DIR  = Path(config.paths.project_root) / "calibration_artefacts"
_TEMP_PATH = _CAL_DIR / "temperature.pt"
_ISO_PATH  = _CAL_DIR / "isotonic.json"


@torch.no_grad()
def _collect_logits(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model over dataset, return (logits_np, labels_np) and log diagnostics."""
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).squeeze(1).cpu()
        all_logits.append(logits)
        all_labels.append(labels)

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()

    # Per-domain diagnostic — reveals domain shift to calibration reviewers
    logger.info(
        f"  [{tag}] n={len(logits_np):5d} | "
        f"logit μ={logits_np.mean():.3f} σ={logits_np.std():.3f} | "
        f"real={int((labels_np==0).sum())} fake={int((labels_np==1).sum())}"
    )
    return logits_np, labels_np


def run_calibration(stage2_ckpt: Path | None = None) -> None:
    set_global_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load best Stage 2 model ─────────────────────────────────────────
    model = ForensicModel().to(device)
    ckpt_path = stage2_ckpt or (
        Path(config.paths.project_root) / "checkpoints" / "best_Stage2_Hybrid.pt"
    )
    ckpt = load_checkpoint(model, ckpt_path, strict=True)
    logger.info(f"Loaded model from {ckpt_path.name} (epoch={ckpt['epoch']})")

    # ── 2. Collect logits from all three CAL domains ───────────────────────
    logger.info("=== Collecting CAL logits (3 domains) ===")

    ff_logits,   ff_labels   = _collect_logits(model, FFPPDataset(mode="cal"),          device, "FF++")
    cel_logits,  cel_labels  = _collect_logits(model, CelebDFDataset(mode="cal"),       device, "CelebDF")
    wild_logits, wild_labels = _collect_logits(model, WildDeepfakeDataset(mode="cal"),  device, "Wild")

    # ── 3. Concatenate + shuffle (raw logits — consistent with evaluation pipeline) ──
    logger.info("Combining domains (raw logits, no normalization)...")
    all_logits_np = np.concatenate([ff_logits, cel_logits, wild_logits])
    all_labels_np = np.concatenate([ff_labels, cel_labels, wild_labels])

    perm = np.random.RandomState(42).permutation(len(all_logits_np))
    all_logits_np = all_logits_np[perm]
    all_labels_np = all_labels_np[perm]

    logger.info(
        f"Hybrid CAL set: {len(all_logits_np)} samples | "
        f"real={int((all_labels_np==0).sum())} fake={int((all_labels_np==1).sum())}"
    )

    logits_t = torch.from_numpy(all_logits_np).float()
    labels_t = torch.from_numpy(all_labels_np).float()

    # ── 5. Temperature Scaling ─────────────────────────────────────────────
    logger.info("Step 1/2: Fitting Temperature Scaling (hybrid)...")
    T = fit_temperature(logits_t, labels_t)
    save_temperature(T, _TEMP_PATH)

    # ── 6. Isotonic Regression (PAVA) ──────────────────────────────────────
    logger.info("Step 2/2: Fitting Isotonic Regression (PAVA) on hybrid CAL...")
    probs_np = torch.sigmoid(logits_t / T).numpy()

    iso = IsotonicCalibrator()
    iso.fit(probs_np, all_labels_np)
    iso.save(_ISO_PATH)

    # ── 7. Sanity check — CLLR per domain ──────────────────────────────────
    from src.evaluation.metrics import compute_cllr

    def _domain_cllr(logits_raw: np.ndarray, labels: np.ndarray, tag: str) -> None:
        # Raw logits → same pipeline as run_evaluation.py
        probs   = torch.sigmoid(torch.from_numpy(logits_raw).float() / T).numpy()
        lr_vals = iso.predict_lr(probs)
        cllr    = compute_cllr(lr_vals, labels)
        flag    = "✅" if cllr < 0.4 else ("⚠️" if cllr < 1.0 else "❌")
        logger.info(f"  SANITY [{tag:12s}] CLLR = {cllr:.4f} {flag}")

    logger.info("=== CLLR Sanity (per domain) ===")
    _domain_cllr(ff_logits,   ff_labels,   "FF++")
    _domain_cllr(cel_logits,  cel_labels,  "CelebDF")
    _domain_cllr(wild_logits, wild_labels, "Wild")

    logger.info("=== Calibration Pipeline COMPLETE ===")


if __name__ == "__main__":
    run_calibration()
