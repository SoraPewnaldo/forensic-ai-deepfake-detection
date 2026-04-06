"""
Full Assemblage Forensic Model for Forensic-v2.
Integrates Spatial (CLIP), Frequency (DCT), and Temporal branches.
"""
import torch
import torch.nn as nn
from src.config import config
from src.models.backbone import CLIPViTBackbone
from src.models.frequency_branch import FrequencyBranch
from src.models.temporal_module import TemporalModule

class ForensicModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Branch Initializations
        self.spatial = CLIPViTBackbone()
        self.frequency = FrequencyBranch()
        self.temporal = TemporalModule()
        
        # 2. Dimensions from SSOT config
        self.spatial_dim = config.architecture.spatial.clip_hidden_dim # 768
        self.freq_dim = config.architecture.frequency.channels # 256
        self.d_model = config.architecture.temporal.d_model # 1024
        
        # 2b. Fusion Balance Projections
        self.spatial_proj = nn.Linear(self.spatial_dim, 512)
        self.freq_proj = nn.Linear(self.freq_dim, 512)
        
        # 3. Model Head (ForensicHead)
        # We output a single logit for Binary Classification
        self.head = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
        # 4. Temperature Scalar for Calibration (initially fixed, fittable in T5.1)
        self.temperature = nn.Parameter(torch.tensor(config.calibration.temperature_init))
        
        print(f"[INFO] ForensicModel assembled. Fusion Dim: {self.d_model}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, T, 3, H, W]
        Output: logit (single scalar per batch element if averaged) or [B, 1]
        """
        # 1. Feature Extraction (Frame-level)
        z_spatial = self.spatial_proj(self.spatial(x))  # [B, T, 512]
        z_freq = self.freq_proj(self.frequency(x))      # [B, T, 512]
        
        # 2. Balanced Multimodal Fusion
        # [B, T, 1024]
        z_fused = torch.cat([z_spatial, z_freq], dim=-1)
        
        # 3. Temporal Modeling (Sequence Aggregation)
        # [B, 1024]
        z_temporal = self.temporal(z_fused)
        
        # 4. Classification Head
        # [B, 1]
        logits = self.head(z_temporal)
        
        # 5. Temperature Scaling (Applied during training/eval, 
        # but fitted separately in calibration phase)
        return logits / self.temperature

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Helper for explicit logit retrieval without T-scaling if needed."""
        with torch.no_grad():
             z_spatial = self.spatial_proj(self.spatial(x))
             z_freq = self.freq_proj(self.frequency(x))
             z_fused = torch.cat([z_spatial, z_freq], dim=-1)
             z_temporal = self.temporal(z_fused)
             return self.head(z_temporal)
