"""
Ablation Evaluation for Forensic-v2 Stage 5.

Loads all 3 ablation checkpoints and evaluates them on the same test sets:
  - FFPP_TEST
  - CelebDF_TEST (zero-shot)
  - WildDeepfake_TEST

Produces a comparison table showing the incremental contribution of each branch.

Command:
  python -m src.evaluation.run_ablation_eval
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
from src.models.ablation_model import AblationForensicModel
from src.calibration.isotonic_calibrator import IsotonicCalibrator
from src.calibration.temperature_scaling import load_temperature
from src.datasets import FFPPDataset
from src.datasets.celebdf_dataset import CelebDFDataset
from src.datasets.wild_dataset import WildDeepfakeDataset
from src.evaluation.metrics import compute_auc, compute_eer, compute_cllr, compute_ap


_CKPT_DIR = Path(config.paths.project_root) / "checkpoints"
_CAL_DIR  = Path(config.paths.project_root) / "calibration_artefacts"
_OUT_DIR  = Path(config.paths.project_root) / "evaluation_results"
_OUT_DIR.mkdir(parents=True, exist_ok=True)


VARIANTS = [
    {
        "name":     "Baseline",
        "ckpt":     "Ablation_Baseline",
        "model_fn": lambda: AblationForensicModel(use_frequency=False, use_temporal=False),
    },
    {
        "name":     "Beta",
        "ckpt":     "Ablation_Beta",
        "model_fn": lambda: AblationForensicModel(use_frequency=False, use_temporal=True),
    },
    {
        "name":     "Gamma (Full)",
        "ckpt":     "Stage2_Hybrid",
        "model_fn": lambda: ForensicModel(),
    },
]


@torch.no_grad()
def evaluate_dataset(model, dataset, device, tag: str, iso: IsotonicCalibrator, T: float):
    loader = DataLoader(dataset, batch_size=config.training.batch_size,
                        shuffle=False, num_workers=config.training.num_workers)
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).squeeze(1).cpu().numpy()
        all_logits.append(logits)
        all_labels.append(labels.numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    probs_np  = torch.sigmoid(torch.from_numpy(logits_np).float() / T).numpy()
    lr_vals   = iso.predict_lr(probs_np)

    return {
        "auc":  compute_auc(probs_np, labels_np),
        "eer":  compute_eer(probs_np, labels_np),
        "cllr": compute_cllr(lr_vals, labels_np),
        "ap":   compute_ap(probs_np, labels_np),
    }


def run_ablation_eval():
    set_global_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load shared calibration artifacts
    T = load_temperature(_CAL_DIR / "temperature.pt")
    iso = IsotonicCalibrator()
    iso.load(_CAL_DIR / "isotonic.json")
    logger.info(f"Calibration loaded: T={T:.4f}")

    # Build test datasets once
    datasets = {
        "FFPP":     FFPPDataset(mode="test"),
        "CelebDF":  CelebDFDataset(mode="test"),
        "Wild":     WildDeepfakeDataset(mode="test"),
    }

    all_results = {}

    for variant in VARIANTS:
        ckpt_path = _CKPT_DIR / f"best_{variant['ckpt']}.pt"
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path} - skipping {variant['name']}")
            continue

        logger.info(f"\n[ABLATION EVAL] Variant: {variant['name']}")
        model = variant["model_fn"]().to(device)
        load_checkpoint(model, ckpt_path, strict=False)
        model.eval()

        variant_results = {}
        for ds_name, ds in datasets.items():
            metrics = evaluate_dataset(model, ds, device, ds_name, iso, T)
            variant_results[ds_name] = metrics
            logger.info(
                f"  {ds_name:12s} | AUC={metrics['auc']:.4f} | "
                f"EER={metrics['eer']:.4f} | CLLR={metrics['cllr']:.4f}"
            )

        all_results[variant["name"]] = variant_results

    #  Save JSON 
    out_path = _OUT_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAblation results saved: {out_path}")

    #  Print comparison table 
    datasets_order = ["FFPP", "CelebDF", "Wild"]
    header = f"{'Variant':<22} | " + " | ".join(
        f"{d} AUC  CLLR" for d in datasets_order
    )
    logger.info("\n" + "="*80)
    logger.info(header)
    logger.info("="*80)
    for vname, vres in all_results.items():
        row = f"{vname:<22} | "
        cols = []
        for d in datasets_order:
            m = vres.get(d, {})
            cols.append(f"{m.get('auc', 0):.4f}  {m.get('cllr', 0):.4f}")
        row += " | ".join(cols)
        logger.info(row)
    logger.info("="*80)


if __name__ == "__main__":
    run_ablation_eval()
