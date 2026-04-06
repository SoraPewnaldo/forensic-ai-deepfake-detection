"""
Full Calibration Pipeline for Forensic-v2.
Step 1: Collect logits on FF++ CAL partition.
Step 2: Fit Temperature Scaling (T > 1.0 enforced).
Step 3: Fit Isotonic Regression (PAVA) on T-scaled probs.
Step 4: Save both artefacts for evaluation and LR reporting.
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
from src.calibration.temperature_scaling import fit_temperature, save_temperature
from src.calibration.isotonic_calibrator import IsotonicCalibrator


_CAL_DIR = Path(config.paths.project_root) / "calibration_artefacts"
_TEMP_PATH = _CAL_DIR / "temperature.pt"
_ISO_PATH  = _CAL_DIR / "isotonic.json"


@torch.no_grad()
def _collect_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Run model on loader, return raw logits and labels as numpy arrays."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).squeeze(1).cpu()
        all_logits.append(logits)
        all_labels.append(labels)
    return (
        torch.cat(all_logits).numpy(),
        torch.cat(all_labels).numpy(),
    )


def run_calibration(stage2_ckpt: Path | None = None) -> None:
    set_global_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Load best Stage 2 model
    # ------------------------------------------------------------------
    model = ForensicModel().to(device)
    ckpt_path = stage2_ckpt or (
        Path(config.paths.project_root) / "checkpoints" / "best_Stage2_Hybrid.pt"
    )
    ckpt = load_checkpoint(model, ckpt_path, strict=True)
    logger.info(f"Loaded model from {ckpt_path.name} (epoch={ckpt['epoch']})")

    # ------------------------------------------------------------------
    # 2. Collect logits on FF++ CAL partition (subject-disjoint, never seen in training)
    # ------------------------------------------------------------------
    cal_dataset = FFPPDataset(mode="cal")
    cal_loader = DataLoader(
        cal_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    logger.info(f"Collecting logits on CAL partition ({len(cal_dataset)} samples)...")
    logits_np, labels_np = _collect_logits(model, cal_loader, device)

    logits_t = torch.from_numpy(logits_np)
    labels_t = torch.from_numpy(labels_np)

    # ------------------------------------------------------------------
    # 3. Temperature Scaling
    # ------------------------------------------------------------------
    logger.info("Step 1/2: Fitting Temperature Scaling...")
    T = fit_temperature(logits_t, labels_t)
    save_temperature(T, _TEMP_PATH)

    # ------------------------------------------------------------------
    # 4. Isotonic Regression (PAVA) on T-scaled probabilities
    # ------------------------------------------------------------------
    logger.info("Step 2/2: Fitting Isotonic Regression (PAVA)...")
    probs_np = torch.sigmoid(logits_t / T).numpy()

    iso = IsotonicCalibrator()
    iso.fit(probs_np, labels_np)
    iso.save(_ISO_PATH)

    # ------------------------------------------------------------------
    # 5. Sanity check — compute CLLR on CAL set
    # ------------------------------------------------------------------
    from src.evaluation.metrics import compute_cllr
    lr_vals = iso.predict_lr(probs_np)
    cllr = compute_cllr(lr_vals, labels_np)
    logger.info(f"CALIBRATION SANITY: CLLR on CAL partition = {cllr:.4f}")

    if cllr < 0.4:
        logger.info("  >> TARGET MET: CLLR < 0.4")
    else:
        logger.warning(f"  >> TARGET MISSED: CLLR={cllr:.4f} >= 0.4. Review calibration strategy.")

    logger.info("=== Calibration Pipeline COMPLETE ===")


if __name__ == "__main__":
    run_calibration()
