"""
Frequency Branch for Forensic-v2.
Processes magnitude spectra using 2D Discrete Cosine Transform (DCT).
Using Option 1: best practical approximate DCT via matrix multiplication.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import config

def get_dct_matrix(N: int, device: torch.device) -> torch.Tensor:
    """Calculates the N x N DCT-II transform matrix."""
    matrix = torch.zeros((N, N), device=device)
    for i in range(N):
        alpha = math.sqrt(1/N) if i == 0 else math.sqrt(2/N)
        for j in range(N):
            matrix[i, j] = alpha * math.cos((math.pi * i * (2 * j + 1)) / (2 * N))
    return matrix

class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.enabled = config.architecture.frequency.enabled
        self.input_size = config.architecture.frequency.input_size # 256
        self.out_channels = config.architecture.frequency.channels # 256
        
        # Pre-calculated DCT matrix (not a parameter, but stored as buffer for device management)
        # We'll initialize it in forward or use a lazy buffer if needed.
        self.register_buffer("dct_mat", None)
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 128
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4), # 32
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1) # 1x1
        )
        
    def _apply_dct2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D DCT using the matrix property Y = C X C^T.
        Input: [B*T, 3, N, N]
        """
        N = x.shape[-1]
        
        if self.dct_mat is None or self.dct_mat.shape[0] != N:
            self.dct_mat = get_dct_matrix(N, x.device)
            
        # x is [B*T, 3, N, N]
        # We want to multiply on last two dims: (C @ X) @ C.T
        # Batch matrix multiplication: (Mat @ Vec)
        
        # Perform C @ X (matrix multiply last two dimensions)
        # Using torch.matmul(C, x) works if C is prepended or we use einsum
        # Y = C @ X @ C.T
        # We'll use einsum for clarity across channels and batch
        # 'ik' is C, 'bc kj' is X, 'jl' is C.T
        # But we have [B*T, 3, N, N]
        res = torch.einsum('ij, bckj -> bcik', self.dct_mat, x)
        res = torch.einsum('bcik, lk -> bcil', res, self.dct_mat)
        
        # Magnitude spectrum
        mag = torch.abs(res)
        
        # Suppress pure DC bias (first coefficient)
        mag[..., 0, 0] = 0.0
        
        # Normalize spectrum per frame and channel
        mag = mag / (mag.mean(dim=(-2, -1), keepdim=True) + 1e-6)
        
        # Log scaling for visual artifacts
        return torch.log(mag + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, T, 3, H, W]
        Output: [B, T, out_channels]
        """
        if not self.enabled:
            B, T = x.shape[:2]
            return torch.zeros(B, T, self.out_channels, device=x.device)
            
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        
        # 1. Resize to efficient spectral input size (112 avoids extreme O(N^2) DCT overhead while preserving macro frequency artifacts)
        downsample_size = 112
        if H != downsample_size or W != downsample_size:
            x_flat = F.interpolate(x_flat, size=(downsample_size, downsample_size), mode='bilinear', align_corners=False)
            
        # 2. Extract Spectral features via DCT
        spectrum = self._apply_dct2d(x_flat)
        
        # 3. Conv processing
        feats = self.conv(spectrum) # [B*T, 256, 1, 1]
        
        return feats.view(B, T, self.out_channels)
