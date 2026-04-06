"""
Windows-safe ASCII-only structured logger.
Log format: [LEVEL] YYYY-MM-DD HH:MM:SS | module | message
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        fmt='[%(levelname)s] %(asctime)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def log_config(cfg, logger: logging.Logger):
    """Recursively log dataclass config values."""
    logger.info("--- SYSTEM CONFIGURATION ---")
    from dataclasses import is_dataclass, asdict
    
    def _log_dict(d, indent=0):
        for k, v in d.items():
            if isinstance(v, dict):
                logger.info(f"{'  ' * indent}{k}:")
                _log_dict(v, indent + 1)
            else:
                logger.info(f"{'  ' * indent}{k}: {v}")
                
    if is_dataclass(cfg):
        _log_dict(asdict(cfg))
    else:
        logger.info(str(cfg))
    logger.info("----------------------------")


# ---------------------------------------------------------------------------
# Module-level singleton: `from src.utils.logging_utils import logger`
# ---------------------------------------------------------------------------
logger = get_logger("forensic-v2")
