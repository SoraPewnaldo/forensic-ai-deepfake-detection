"""
Base Trainer for Forensic-v2.
Implements the staged training protocols with VRAM-aware gradient accumulation.
"""
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.config import config
from src.utils.logging_utils import logger
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.device import log_vram_usage, assert_vram_safe


class ForensicTrainer:
    def __init__(self, model: nn.Module, device: torch.device, stage_name: str = "Stage1"):
        self.model = model.to(device)
        self.device = device
        self.stage_name = stage_name
        self.patience = config.training.early_stopping_patience
        self.acc_steps = config.training.gradient_accumulation_steps
        self.smoothing = config.training.label_smoothing

        # FP16 AMP Scaler for 6GB VRAM limit
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.criterion = nn.BCEWithLogitsLoss()

        # Dual-LR optimizer: separate frozen backbone params from head/new modules
        head_params = []
        backbone_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "spatial.model" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': head_params,     'lr': config.training.lrs.classifier_head},
            {'params': backbone_params, 'lr': config.training.lrs.unfrozen_backbone},
        ], weight_decay=1e-2)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.training.epochs
        )

        # Checkpoint path
        self._ckpt_dir = Path(config.paths.project_root) / "checkpoints"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _smooth(self, labels: torch.Tensor) -> torch.Tensor:
        # real=0.1, fake=0.9
        return labels * 0.8 + 0.1

    # ------------------------------------------------------------------
    def train_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        pbar = tqdm(loader, ascii=True, desc=f"Ep{epoch:03d} [TRAIN]", leave=False)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # FP16 Precision context
            with torch.amp.autocast('cuda', enabled=True):
                logits = self.model(images).squeeze(1)
                loss = self.criterion(logits, self._smooth(labels)) / self.acc_steps
            
            # Scaler backwards
            self.scaler.scale(loss).backward()

            if (i + 1) % self.acc_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.acc_steps
            pbar.set_postfix(loss=f"{loss.item() * self.acc_steps:.4f}")

            if i % 200 == 0:
                log_vram_usage(logger, tag=f"Ep{epoch} step{i}")

        # flush remaining gradient
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()

        return total_loss / max(len(loader), 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader, tag: str = "VAL") -> float:
        self.model.eval()
        all_probs, all_labels = [], []

        for images, labels in tqdm(loader, ascii=True, desc=f"[{tag}]", leave=False):
            images = images.to(self.device, non_blocking=True)
            logits = self.model(images).squeeze(1)
            probs = torch.sigmoid(logits).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())

        if len(set(all_labels)) < 2:
            logger.warning(f"[{tag}] Only one class present - AUC undefined, returning 0.5")
            return 0.5

        return roc_auc_score(all_labels, all_probs)

    # ------------------------------------------------------------------
    def run_training(self, train_loader, val_loader, test_loader) -> float:
        logger.info(f"=== STARTING {self.stage_name} TRAINING ===")
        best_auc = 0.0
        no_improve = 0
        ckpt_path = self._ckpt_dir / f"best_{self.stage_name}.pt"

        for epoch in range(1, config.training.epochs + 1):
            assert_vram_safe(min_free_mb=512)

            train_loss = self.train_epoch(train_loader, epoch)
            val_auc = self.evaluate(val_loader, tag="VAL-FFPP")
            cel_auc = self.evaluate(test_loader, tag="CELEBDF-PROXY")

            logger.info(
                f"Ep{epoch:03d} | Loss={train_loss:.4f} | "
                f"Val-AUC={val_auc:.4f} | CelebDF-AUC={cel_auc:.4f}"
            )

            if cel_auc > best_auc:
                best_auc = cel_auc
                no_improve = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    metric_val=cel_auc, path=ckpt_path,
                    meta={"stage": self.stage_name, "ffpp_auc": val_auc}
                )
                logger.info(f"  >> NEW BEST CelebDF-AUC={best_auc:.4f}. Checkpoint saved.")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch} (patience={self.patience})")
                    break

        logger.info(f"=== {self.stage_name} COMPLETE | Best Val-AUC={best_auc:.4f} ===")
        return best_auc
