import torch
import torch.nn as nn
import torch.nn.functional as F
from .dusa_attention import DuSABlock

class HyDHead(nn.Module):
    """
    Lightweight Decoder Head with Detail Refinement (DuSA).
    Steps:
    1. MLP to unify channel dimensions of encoder outputs.
    2. Upsample and Concat (Fuse).
    3. DuSA Refinement (using a learned or provided importance map).
    4. Classifier.
    """
    def __init__(self, in_channels_list, embedding_dim=128, num_classes=19):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Channel unification
        self.linear_layers = nn.ModuleList([
            nn.Conv2d(in_ch, embedding_dim, 1)
            for in_ch in in_channels_list
        ])
        
        # 2. Fusion
        # Fused dimension = embedding_dim * len(inputs)
        self.fuse_dim = embedding_dim * len(in_channels_list)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(self.fuse_dim, embedding_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU()
        )
        
        # 3. Refinement (DuSA)
        self.dusa = DuSABlock(embedding_dim, topk_ratio=0.1)
        
        # 4. Classifier
        self.cls_seg = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, inputs):
        # inputs: list of features [c1, c2, c3, c4] at different scales
        
        # Unify channels
        projected = [layer(x) for layer, x in zip(self.linear_layers, inputs)]
        
        # Upsample all to the size of the first (largest) feature map (usually 1/4)
        target_size = projected[0].shape[-2:]
        upsampled = []
        for p in projected:
            if p.shape[-2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
            upsampled.append(p)
            
        # Concat
        fused = torch.cat(upsampled, dim=1) # (B, 4*emb, H/4, W/4)
        fused = self.fuse_conv(fused) # (B, emb, H/4, W/4)
        
        # Apply DuSA
        # We can use the fused feature itself to compute importance
        fused = self.dusa(fused) # Residual connection inside
        
        # Logic output
        out = self.cls_seg(fused)
        return out
