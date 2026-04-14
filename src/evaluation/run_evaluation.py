"""
Cross-Dataset Evaluation Runner for Forensic-v2.
Produces the full 3-dataset evaluation matrix + Tippett plots.

Datasets evaluated:
  - FF++ TEST partition (in-distribution)
  - WildDeepfake TEST  (held-out domain)
  - Celeb-DF TEST      (zero-shot, never seen in any training stage)
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
from src.datasets import FFPPDataset, WildDeepfakeDataset, CelebDFDataset
from src.calibration.temperature_scaling import load_temperature
from src.calibration.isotonic_calibrator import IsotonicCalibrator
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.tippett_plot import plot_tippett


_CAL_DIR   = Path(config.paths.project_root) / "calibration_artefacts"
_EVAL_DIR  = Path(config.paths.project_root) / "evaluation_results"
_TEMP_PATH = _CAL_DIR / "temperature.pt"
_ISO_PATH  = _CAL_DIR / "isotonic.json"


@torch.no_grad()
def _infer(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).squeeze(1).cpu().numpy()
        logits_all.append(logits)
        labels_all.append(labels.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def run_evaluation(stage2_ckpt: Path | None = None) -> dict:
    set_global_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    model = ForensicModel().to(device)
    ckpt_path = stage2_ckpt or (
        Path(config.paths.project_root) / "checkpoints" / "best_Stage2_Hybrid.pt"
    )
    load_checkpoint(model, ckpt_path, strict=True)
    logger.info(f"Eval model loaded: {ckpt_path.name}")

    # ------------------------------------------------------------------
    # 2. Load calibration artefacts
    # ------------------------------------------------------------------
    T = load_temperature(_TEMP_PATH)
    iso = IsotonicCalibrator()
    iso.load(_ISO_PATH)
    logger.info(f"Calibration loaded: T={T:.4f}")

    # ------------------------------------------------------------------
    # 3. Dataset list
    # ------------------------------------------------------------------
    bs = config.training.batch_size
    nw = config.training.num_workers

    eval_datasets = {
        "FFPP_TEST":        FFPPDataset(mode="test"),
        "WildDeepfake_TEST":WildDeepfakeDataset(mode="test"),
        "CelebDF_TEST":     CelebDFDataset(mode="test"),
    }

    results = {}

    for name, ds in eval_datasets.items():
        if len(ds) == 0:
            logger.warning(f"Skipping {name} — dataset is empty.")
            continue

        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
        logger.info(f"Evaluating on {name} ({len(ds)} samples)...")

        logits, labels = _infer(model, loader, device)

        # Raw probability from temperature-scaled logits
        probs = 1.0 / (1.0 + np.exp(-logits / T))      # sigmoid(logit/T)

        # Isotonic LR
        lr_values = iso.predict_lr(probs)

        # Metrics
        metrics = compute_all_metrics(probs, labels, lr_values=lr_values)
        results[name] = metrics

        # Tippett plot
        plot_tippett(
            lr_values, labels,
            output_path=_EVAL_DIR / f"tippett_{name}.png",
            title=f"Tippett Plot — {name}",
        )

        logger.info(
            f"  {name} | AUC={metrics['auc']:.4f} | "
            f"EER={metrics['eer']:.4f} | HTER={metrics['hter']:.4f} | "
            f"CLLR={metrics['cllr']:.4f} | AP={metrics['ap']:.4f}"
        )

    # ------------------------------------------------------------------
    # 4. Save results JSON
    # ------------------------------------------------------------------
    results_path = _EVAL_DIR / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation results saved: {results_path}")

    # ------------------------------------------------------------------
    # 5. Print summary table
    # ------------------------------------------------------------------
    _print_table(results)

    return results


def _print_table(results: dict) -> None:
    header = f"{'Dataset':<22} {'AUC':>6} {'EER':>6} {'HTER':>6} {'CLLR':>6} {'AP':>6}"
    sep = "-" * len(header)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for ds_name, m in results.items():
        row = (
            f"{ds_name:<22} "
            f"{m.get('auc', float('nan')):>6.4f} "
            f"{m.get('eer', float('nan')):>6.4f} "
            f"{m.get('hter', float('nan')):>6.4f} "
            f"{m.get('cllr', float('nan')):>6.4f} "
            f"{m.get('ap', float('nan')):>6.4f}"
        )
        logger.info(row)
    logger.info(sep)


if __name__ == "__main__":
    run_evaluation()
