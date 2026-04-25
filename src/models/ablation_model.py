"""
Ablation Model for Forensic-v2 Stage 5.

Supports 3 architecturally distinct configurations to isolate the contribution
of each branch, used in the Stage 5 scientific ablation study.

Config          | use_frequency | use_temporal | Description
----------------|---------------|--------------|---------------------------
Baseline (A)    | False         | False        | Spatial only (CLIP ViT)
Beta (B)        | False         | True         | Spatial + Temporal SA
Gamma (Full)    | True          | True         | Spatial + Frequency + Temporal (our model)

Pipeline invariant: fusion dim = 1024, head always takes 1024.
This ensures the head capacity is identical across all ablation variants.
"""
import torch
import torch.nn as nn
from src.config import config
from src.models.backbone import CLIPViTBackbone
from src.models.frequency_branch import FrequencyBranch
from src.models.temporal_module import TemporalModule


class AblationForensicModel(nn.Module):
    """
    Configurable forensic model for ablation experiments.

    All variants share:
    - Same CLIP-ViT spatial backbone (same freeze policy).
    - Same classification head (1024  1).
    - Same fusion dimensionality (1024).

    Only the presence/absence of modality branches varies.
    """

    VARIANT_NAMES = {
        (False, False): "Baseline",
        (False, True):  "Beta",
        (True,  True):  "Gamma",
    }

    def __init__(self, use_frequency: bool = True, use_temporal: bool = True):
        super().__init__()
        self.use_frequency = use_frequency
        self.use_temporal  = use_temporal

        fusion_dim = config.architecture.temporal.d_model  # 1024

        #  Spatial branch (always present) 
        self.spatial = CLIPViTBackbone()
        spatial_hidden = config.architecture.spatial.clip_hidden_dim  # 768

        if use_frequency:
            # Project each branch to 512 then concatenate  1024
            self.spatial_proj = nn.Linear(spatial_hidden, 512)
            self.frequency     = FrequencyBranch()
            self.freq_proj     = nn.Linear(config.architecture.frequency.channels, 512)
        else:
            # Project spatial directly to fusion_dim  1024
            self.spatial_proj = nn.Linear(spatial_hidden, fusion_dim)

        #  Temporal branch (optional) 
        if use_temporal:
            self.temporal = TemporalModule()  # expects [B, T, fusion_dim]
        else:
            self.temporal = None  # mean-pool over T instead

        #  Classification head (shared capacity across all variants) 
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

        # Temperature scalar (same as production model)
        self.temperature = nn.Parameter(
            torch.tensor(config.calibration.temperature_init)
        )

        variant = self.VARIANT_NAMES.get((use_frequency, use_temporal), "Custom")
        print(
            f"[ABLATION] Model variant: {variant} "
            f"(freq={use_frequency}, temporal={use_temporal})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: [B, T, 3, H, W]  Output: logit [B, 1]"""
        z = self.spatial_proj(self.spatial(x))   # [B, T, 512 or 1024]

        if self.use_frequency:
            z_freq = self.freq_proj(self.frequency(x))   # [B, T, 512]
            z = torch.cat([z, z_freq], dim=-1)            # [B, T, 1024]

        if self.use_temporal:
            z = self.temporal(z)   # [B, 1024]
        else:
            z = z.mean(dim=1)      # mean-pool over T  [B, 1024]

        logits = self.head(z)      # [B, 1]
        return logits / self.temperature
