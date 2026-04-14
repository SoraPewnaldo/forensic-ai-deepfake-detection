"""
Stage 1 Training: FF++ Foundation.
Trains the full model (Frozen Backbone blocks 0-8) on FF++ only.
"""
import torch
from src.config import config
from src.utils.seed import set_global_seed
from src.utils.logging_utils import logger
from src.models import ForensicModel
from src.datasets import get_dataloaders
from src.training.trainer import ForensicTrainer

def main():
    set_global_seed(config.training.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("INIT: Forensic-AI V2.1 Stage 1 (FF++ Only)")
    
    # 1. Model
    model = ForensicModel()
    
    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(stage=1)
    
    # 3. Trainer
    trainer = ForensicTrainer(model, device, stage_name="Stage1_FFPP")
    
    # 4. Run loop
    trainer.run_training(
        train_loader, val_loader, test_loader
    )
    
    logger.info("Stage 1 COMPLETE.")

if __name__ == "__main__":
    main()
