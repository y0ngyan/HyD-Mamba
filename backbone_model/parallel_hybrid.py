import torch
import torch.nn as nn
from functools import partial
from .vmamba_layer import SS2Dv2, LayerNorm2d

class ParallelHybridBlock(nn.Module):
    """
    Hybrid Block with Parallel Geometry (CNN) and Context (Mamba) branches.
    Ref: HCT-Net idea of parallel local-global processing.
    Structure:
       Input
       /   \
    [Conv] [Mamba]
       \   /
      Concat -> Fusion
    """
    def __init__(self, dim, d_state=16, conv_k=3):
        super().__init__()
        
        # Branch 1: Local Detail (CNN)
        # Depthwise Conv to keep it lightweight
        self.local_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=conv_k, padding=conv_k//2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
        
        # Branch 2: Global Context (Mamba / SS2D)
        self.norm = LayerNorm2d(dim)
        self.global_branch = SS2Dv2(d_model=dim, d_state=d_state)
        
        # Fusion
        # If we concat, dim becomes 2*dim. We need to project back.
        # Or we can just Add them?
        # HCT-Net usually does Concat -> Projection.
        self.fusion = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        
        # Local
        x_local = self.local_branch(x)
        
        # Global
        # SS2D expects (B, H, W, C), LayerNorm2d expects (B, C, H, W)
        x_norm = self.norm(x) # (B, C, H, W)
        x_norm_cl = x_norm.permute(0, 2, 3, 1) # (B, H, W, C)
        x_global = self.global_branch(x_norm_cl) 
        x_global = x_global.permute(0, 3, 1, 2) # (B, C, H, W)
        
        # Concat
        # (B, 2C, H, W)
        x_cat = torch.cat([x_local, x_global], dim=1)
        
        # Fuse
        x_cat = x_cat.permute(0, 2, 3, 1) # Channel last for Linear
        x_out = self.fusion(x_cat)
        x_out = x_out.permute(0, 3, 1, 2)
        
        # Residual connection if dims match?
        # Usually Block = X + Module(X)
        return x + x_out
