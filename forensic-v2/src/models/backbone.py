"""
CLIP-ViT Backbone for Forensic-v2.
Using timm with weight interpolation for 256x256 resolution.
"""
import torch
import torch.nn as nn
import timm
from src.config import config

class CLIPViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 'vit_base_patch16_clip_224.openai' weights 
        # timm automatically interpolates positional embeddings to config.training.img_size (256)
        self.model = timm.create_model(
            'vit_base_patch16_clip_224.openai',
            pretrained=config.architecture.spatial.pretrained,
            num_classes=0, # Returns pooled CLS token [B, 768]
            img_size=config.training.img_size
        )
        
        self.freeze_blocks = config.architecture.spatial.freeze_blocks
        self.use_checkpointing = config.architecture.spatial.gradient_checkpointing
        self.hidden_dim = config.architecture.spatial.clip_hidden_dim # 768
        
        self._apply_freeze_policy()
        if self.use_checkpointing:
            self.model.set_grad_checkpointing(enable=True)
            print("[INFO] CLIP Backbone: Gradient Checkpointing ENABLED.")

    def _apply_freeze_policy(self):
        """
        FREEZE POLICY: Freeze patch embeddings + first N transformer blocks.
        V2.1 standard: freeze_blocks = 8.
        """
        # 1. Freeze patch embeddings
        for param in self.model.patch_embed.parameters():
            param.requires_grad = False
            
        # 2. Freeze positional embeddings
        if hasattr(self.model, 'pos_embed'):
            self.model.pos_embed.requires_grad = False
            
        # 3. Freeze first N blocks
        for i, block in enumerate(self.model.blocks):
            if i < self.freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False
        
        print(f"[INFO] CLIP Backbone: Frozen PatchEmbed + first {self.freeze_blocks} blocks.")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, T, 3, H, W]
        Output: [B, T, hidden_dim]
        """
        B, T, C, H, W = x.shape
        # Flatten B*T for ViT processing
        x_flat = x.view(B * T, C, H, W)
        
        # timm returns the pooled CLS token directly due to num_classes=0
        cls_embedding = self.model(x_flat) # [B*T, 768]
        
        return cls_embedding.view(B, T, self.hidden_dim)
