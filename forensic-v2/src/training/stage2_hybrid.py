"""
Stage 2 Training: Hybrid Domain Adaptation.
Loads Stage 1 best checkpoint, unfreezes ViT blocks 9-12, trains on FF++ + WildDeepfake.
"""
import torch
from pathlib import Path

from src.config import config
from src.utils.seed import set_global_seed
from src.utils.logging_utils import logger
from src.utils.checkpoint import load_checkpoint
from src.models import ForensicModel
from src.datasets import get_dataloaders
from src.training.trainer import ForensicTrainer


def _unfreeze_top_blocks(model: ForensicModel, n_unfreeze: int = 4):
    """
    Stage 2 policy: unfreeze the top N ViT blocks (blocks 8-11 for freeze_blocks=8).
    All other previously-frozen params remain frozen.
    """
    blocks = model.spatial.model.blocks
    total = len(blocks)
    for i, block in enumerate(blocks):
        if i >= (total - n_unfreeze):
            for param in block.parameters():
                param.requires_grad = True
    logger.info(f"Stage 2: Unfrozen top {n_unfreeze} ViT blocks (blocks {total - n_unfreeze}-{total - 1}).")


def main():
    set_global_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("INIT: Forensic-AI V2.1 Stage 2 (Hybrid Domain Adaptation)")

    # 1. Model
    model = ForensicModel()

    # 2. Load Stage 1 checkpoint
    stage1_ckpt = Path(config.paths.project_root) / "checkpoints" / "best_Stage1_FFPP.pt"
    if stage1_ckpt.exists():
        ckpt = load_checkpoint(model, stage1_ckpt, strict=True)
        logger.info(f"Loaded Stage 1 checkpoint (epoch={ckpt['epoch']}, val_auc={ckpt['metric_val']:.4f})")
    else:
        logger.warning(f"Stage 1 checkpoint NOT FOUND at {stage1_ckpt}. Training from scratch.")

    # 3. Unfreeze top 4 ViT blocks for Stage 2 fine-tuning
    _unfreeze_top_blocks(model, n_unfreeze=4)

    # 4. Hybrid data (FF++ + WildDeepfake)
    train_loader, val_loader, test_loader = get_dataloaders(stage=2)

    # 5. Trainer — stage name controls checkpoint filename
    trainer = ForensicTrainer(model, device, stage_name="Stage2_Hybrid")
    trainer.run_training(train_loader, val_loader, test_loader)

    logger.info("Stage 2 COMPLETE.")


if __name__ == "__main__":
    main()
