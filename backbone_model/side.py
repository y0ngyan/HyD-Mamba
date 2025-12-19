import torch
import torch.nn as nn

class SIDE(nn.Module):
    """
    Scale-Invariant Depth Encoder (SIDE)
    Paper: Real-time RGB-D Semantic Segmentation for Autonomous Systems
    Action: Log transform -> Instance Normalization
    """
    def __init__(self):
        super().__init__()
        # Instance Norm for 1 channel depth
        self.inst_norm = nn.InstanceNorm2d(1, affine=True)

    def forward(self, depth):
        # depth: (B, 1, H, W)
        
        # 1. Log Transform: ln(D + 1)
        # Add epsilon to avoid log(0) if not already handled, though D>=0 usually.
        depth_log = torch.log(depth + 1.0)
        
        # 2. Instance Normalization
        # Remove absolute scale info, keep relative geometry
        out = self.inst_norm(depth_log)
        
        return out

class NRGM(nn.Module):
    """
    Noise Robust Guiding Module (NRGM)
    Paper: Real-time RGB-D Semantic Segmentation...
    Action: Generate mask from raw depth (valid > 0), suppress noise in features.
    """
    def __init__(self):
        super().__init__()
        # No learnable params strictly needed for the mask generation itself,
        # but usually we might want to refine the mask?
        # The prompt says: "Generate mask M_valid = (Depth > 0). Depth_Feat = Depth_Feat * M_valid"
        pass

    def forward(self, depth_feat, raw_depth):
        # depth_feat: (B, C, H, W)
        # raw_depth: (B, 1, H_raw, W_raw) typically same resolution or needs sizing
        
        # Resize raw_depth to feature size if needed
        if raw_depth.shape[-2:] != depth_feat.shape[-2:]:
            raw_depth = nn.functional.interpolate(raw_depth, size=depth_feat.shape[-2:], mode='nearest')
            
        # Generate Mask: 1 if depth > 0 (valid), 0 otherwise (noise/black hole)
        # RealSense holes are exactly 0.
        mask = (raw_depth > 0).float()
        
        # Suppress noise
        return depth_feat * mask
