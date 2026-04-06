"""
Ablation Runner for Forensic-v2.
Evaluates three configurations on the FF++ TEST + CelebDF TEST sets:
  A) CLIP-only          (no freq, no temporal)
  B) CLIP + Temporal    (no freq)
  C) CLIP + Temporal + Frequency  (full model, baseline for paper)

Saves results to evaluation_results/ablation.json.
"""
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.config import config
from src.utils.seed import set_global_seed
from src.utils.logging_utils import logger
from src.utils.checkpoint import load_checkpoint
from src.models import ForensicModel
from src.datasets import FFPPDataset, CelebDFDataset
from src.calibration.temperature_scaling import load_temperature
from src.calibration.isotonic_calibrator import IsotonicCalibrator
from src.evaluation.metrics import compute_auc, compute_eer, compute_hter


_CAL_DIR  = Path(config.paths.project_root) / "calibration_artefacts"
_EVAL_DIR = Path(config.paths.project_root) / "evaluation_results"


@torch.no_grad()
def _infer(model, loader, device, disable_freq=False, disable_temporal=False):
    """
    Inference with optional branch disabling via config-level monkey-patching.
    We temporarily set module flags rather than rebuilding the model.
    """
    model.eval()
    if disable_freq:
        model.frequency.enabled = False
    if disable_temporal:
        model.temporal.enabled = False

    logits_all, labels_all = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).squeeze(1).cpu().numpy()
        logits_all.append(logits)
        labels_all.append(labels.numpy())

    # Restore
    model.frequency.enabled = True
    model.temporal.enabled = True

    return np.concatenate(logits_all), np.concatenate(labels_all)


def run_ablation(stage2_ckpt: Path | None = None) -> dict:
    set_global_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = ForensicModel().to(device)
    ckpt_path = stage2_ckpt or (
        Path(config.paths.project_root) / "checkpoints" / "best_Stage2_Hybrid.pt"
    )
    load_checkpoint(model, ckpt_path, strict=True)

    # Load calibration
    T = load_temperature(_CAL_DIR / "temperature.pt")
    iso = IsotonicCalibrator()
    iso.load(_CAL_DIR / "isotonic.json")

    bs = config.training.batch_size
    nw = config.training.num_workers

    dataset_map = {
        "FFPP_TEST":    FFPPDataset(mode="test"),
        "CelebDF_TEST": CelebDFDataset(mode="test"),
    }

    configurations = [
        ("CLIP_only",        dict(disable_freq=True,  disable_temporal=True)),
        ("CLIP+Temporal",    dict(disable_freq=True,  disable_temporal=False)),
        ("CLIP+Temp+Freq",   dict(disable_freq=False, disable_temporal=False)),
    ]

    results = {}

    for ds_name, ds in dataset_map.items():
        if len(ds) == 0:
            logger.warning(f"Ablation: Skipping {ds_name} — empty dataset.")
            continue
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
        results[ds_name] = {}

        for cfg_name, flags in configurations:
            logits, labels = _infer(model, loader, device, **flags)
            probs = 1.0 / (1.0 + np.exp(-logits / T))
            lr_vals = iso.predict_lr(probs)

            row = {
                "auc":  compute_auc(probs, labels),
                "eer":  compute_eer(probs, labels),
                "hter": compute_hter(probs, labels),
            }
            results[ds_name][cfg_name] = row

            logger.info(
                f"Ablation | {ds_name} | {cfg_name:<18} | "
                f"AUC={row['auc']:.4f} | EER={row['eer']:.4f} | HTER={row['hter']:.4f}"
            )

    # Save
    out_path = _EVAL_DIR / "ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Ablation results saved: {out_path}")

    return results


if __name__ == "__main__":
    run_ablation()
