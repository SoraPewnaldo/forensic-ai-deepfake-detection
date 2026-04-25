"""
Isotonic Regression Calibrator for Forensic-v2.
Implements PAVA (Pool Adjacent Violators Algorithm) via sklearn.
Converts calibrated probabilities  Likelihood Ratios (LR).

Court-admissible output formula:
    LR = P(score | fake) / P(score | real)
        P_calibrated / (1 - P_calibrated)      [log-odds form]
"""
import json
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from src.utils.logging_utils import logger


class IsotonicCalibrator:
    """
    Two-step: fits isotonic regression on calibrated probabilities,
    then converts output to Likelihood Ratios.
    """

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self._fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit PAVA on temperature-scaled probabilities from the CAL partition.

        Args:
            probs:  Sigmoid(logits / T)  - shape [N]
            labels: Binary ground truth  - shape [N]
        """
        self.iso.fit(probs, labels)
        self._fitted = True
        logger.info("CALIBRATION: Isotonic Regression (PAVA) fitted.")

    def predict_prob(self, probs: np.ndarray) -> np.ndarray:
        """
        Return calibrated posterior P(fake | x).
        Implemented as a manual step-function interpolation to be picklable/serializable.
        """
        self._check_fitted()
        # Manual interpolation using thresholds
        # np.searchsorted finds segments, np.take gets the calibrated y values
        idx = np.searchsorted(self.iso.X_thresholds_, probs, side="right") - 1
        idx = np.clip(idx, 0, len(self.iso.y_thresholds_) - 1)
        return self.iso.y_thresholds_[idx]

    def predict_lr(self, probs: np.ndarray) -> np.ndarray:
        """
        Return Likelihood Ratio: LR = p_cal / (1 - p_cal).
        Values < 1  evidence for real.
        Values > 1  evidence for fake.
        """
        p_cal = self.predict_prob(probs)
        # Clip to prevent division by zero
        p_cal = np.clip(p_cal, 1e-7, 1 - 1e-7)
        return p_cal / (1.0 - p_cal)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "X_thresholds": self.iso.X_thresholds_.tolist(),
            "y_thresholds": self.iso.y_thresholds_.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"CALIBRATION: Isotonic calibrator saved to {path}")

    def load(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.iso.X_thresholds_ = np.array(data["X_thresholds"])
        self.iso.y_thresholds_ = np.array(data["y_thresholds"])
        
        # sklearn's predict(out_of_bounds='clip') requires these internal attributes
        self.iso.X_min_ = self.iso.X_thresholds_[0]
        self.iso.X_max_ = self.iso.X_thresholds_[-1]
        
        self._fitted = True
        logger.info(f"CALIBRATION: Isotonic calibrator loaded from {path}")

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator is not fitted. Call .fit() first.")
