"""
Temperature Scaling for Forensic-v2 Calibration.
Fits a single temperature scalar T on the calibration partition logits.
Produces calibrated probabilities P(fake | x) used as input to isotonic regression.
"""
import torch
import torch.nn as nn
from pathlib import Path
from src.config import config
from src.utils.logging_utils import logger


class TemperatureScaler(nn.Module):
    """Wraps any model and scales its logits by a learnable temperature."""

    def __init__(self):
        super().__init__()
        init_val = float(config.calibration.temperature_init)
        self.temperature = nn.Parameter(torch.tensor(init_val))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.clamp_temperature()

    def clamp_temperature(self) -> torch.Tensor:
        # Let optimizer decide, clamping only extreme numeric blowouts
        return torch.clamp(self.temperature, min=0.5, max=10.0)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 200,
    lr: float = 0.05,
) -> float:
    """
    Optimise temperature T by minimising NLL on calibration partition logits.

    Args:
        logits:  Raw model logits [N] — collected on 'cal' split ONLY.
        labels:  Binary ground-truth [N].
        max_iter: LBFGS iterations.
        lr:       Learning rate for LBFGS.

    Returns:
        Fitted temperature value (float).
    """
    scaler = TemperatureScaler()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    def _closure():
        optimizer.zero_grad()
        scaled = scaler(logits)
        loss = criterion(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(_closure)

    fitted_t = scaler.clamp_temperature().item()

    if fitted_t <= 0.5 or fitted_t >= 10.0:
        logger.warning(
            f"CALIBRATION: Temperature converged to {fitted_t:.4f} (hit clamp boundary). "
            "Model is either heavily underconfident or overconfident."
        )
    else:
        logger.info(f"CALIBRATION: Temperature fitted to T={fitted_t:.4f}")

    return fitted_t


def save_temperature(t: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"temperature": t}, path)
    logger.info(f"CALIBRATION: Temperature saved to {path}")


def load_temperature(path: Path) -> float:
    data = torch.load(path, map_location="cpu")
    return float(data["temperature"])
