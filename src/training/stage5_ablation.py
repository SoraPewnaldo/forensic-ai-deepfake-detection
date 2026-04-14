"""
Stage 5: Scientific Ablation Training for Forensic-v2.

Trains 3 ablation variants on FF++ (Stage 1 protocol) to isolate
the contribution of each architectural branch.

Variant         | Branches Active
----------------|--------------------------------------
Baseline (A)    | Spatial only
Beta (B)        | Spatial + Temporal
Gamma (Ours)    | Spatial + Frequency + Temporal (full model, already trained)

All variants use identical:
  - Training data:   FF++ (Stage 1, frozen backbone blocks 0-7)
  - Early stopping:  CelebDF Proxy AUC (patience=5)
  - Optimiser:       AdamW, dual-LR, same schedule
  - Head capacity:   1024 → 512 → 1

After training, run src.evaluation.run_ablation_eval to produce comparison tables.

Command:
  python -m src.training.stage5_ablation
"""
import torch
from src.config import config
from src.utils.seed import set_global_seed
from src.utils.logging_utils import logger
from src.models.ablation_model import AblationForensicModel
from src.datasets import get_dataloaders
from src.training.trainer import ForensicTrainer


ABLATION_VARIANTS = [
    {
        "name":          "Ablation_Baseline",
        "use_frequency": False,
        "use_temporal":  False,
        "description":   "Spatial only (CLIP ViT + mean-pool)",
    },
    {
        "name":          "Ablation_Beta",
        "use_frequency": False,
        "use_temporal":  True,
        "description":   "Spatial + Temporal SA",
    },
    # Gamma = our trained model (best_Stage2_Hybrid.pt). No retraining needed.
]


def run_ablation(variant: dict, device: torch.device) -> float:
    logger.info(f"\n{'='*60}")
    logger.info(f"[ABLATION] Variant: {variant['name']}")
    logger.info(f"[ABLATION] Config:  {variant['description']}")
    logger.info(f"{'='*60}")

    model = AblationForensicModel(
        use_frequency=variant["use_frequency"],
        use_temporal=variant["use_temporal"],
    )

    # Stage 1 dataloaders (FF++ only, same as production Stage 1)
    train_loader, val_loader, celeb_loader = get_dataloaders(stage=1)

    trainer = ForensicTrainer(
        model=model,
        device=device,
        stage_name=variant["name"],
    )

    best_auc = trainer.run_training(train_loader, val_loader, celeb_loader)
    logger.info(f"[ABLATION] {variant['name']} DONE | Best CelebDF-AUC={best_auc:.4f}")
    return best_auc


def main():
    set_global_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("INIT: Forensic-AI V2.1 Stage 5 — Scientific Ablation Training")
    logger.info(f"Running {len(ABLATION_VARIANTS)} variants (Baseline, Beta).")
    logger.info("Gamma (full model) = best_Stage2_Hybrid.pt (already trained).")

    results = {}
    for variant in ABLATION_VARIANTS:
        auc = run_ablation(variant, device)
        results[variant["name"]] = auc

    logger.info("\n=== ABLATION SUMMARY ===")
    for name, auc in results.items():
        logger.info(f"  {name:30s} | CelebDF Proxy AUC = {auc:.4f}")
    logger.info(
        "  Ablation_Gamma (Full)         | CelebDF Proxy AUC = 0.9435 "
        "(best_Stage2_Hybrid.pt, Ep4)"
    )
    logger.info("\nNext: python -m src.evaluation.run_ablation_eval")


if __name__ == "__main__":
    main()
