"""
Model architectures for Forensic-v2.
CLIP-ViT + Frequency DCT + Temporal Self-Attention Fusion.
"""
from .backbone import CLIPViTBackbone
from .frequency_branch import FrequencyBranch
from .temporal_module import TemporalModule
from .forensic_model import ForensicModel

__all__ = [
    'CLIPViTBackbone',
    'FrequencyBranch',
    'TemporalModule',
    'ForensicModel'
]
