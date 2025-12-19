import torch
import torch.nn as nn
import torch.nn.functional as F
from .dusa_attention import DuSABlock

class HyDHead(nn.Module):
    """
    Upgraded Decoder Head: Top-Down FPN structure.
    Resolves edge blurriness by fusing High-Level semantics with Low-Level details.
    Steps:
    1. Unify channels of all stages.
    2. Top-down fusion: P4 -> P3 -> P2 -> P1.
    3. DuSA Refinement at Stage 1 (1/4 scale).
    """
    def __init__(self, in_channels_list, embedding_dim=64, num_classes=19):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Channel unification for all stages (Stage 1 to 4)
        self.proj = nn.ModuleList([
            nn.Conv2d(in_ch, embedding_dim, 1)
            for in_ch in in_channels_list
        ])
        
        # 2. Top-down fusion layers (3 fusions for 4 stages)
        # Each fusion: Upsample(Higher) + Lower -> Conv
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embedding_dim * 2, embedding_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(embedding_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(len(in_channels_list) - 1)
        ])
        
        # 3. Final Refinement (DuSA Moved to Stage 1)
        self.dusa = DuSABlock(embedding_dim, topk_ratio=0.1)
        
        # 4. Classifier
        self.cls_seg = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, inputs, depth=None):
        # inputs: [P1(1/4), P2(1/8), P3(1/16), P4(1/32)]
        
        # Project all to embedding_dim
        p1, p2, p3, p4 = [layer(x) for layer, x in zip(self.proj, inputs)]
        
        # Top-down fusion
        # P4 -> P3
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode='bilinear', align_corners=False)
        f3 = self.fusion_layers[2](torch.cat([p3, p4_up], dim=1))
        
        # P3' -> P2
        f3_up = F.interpolate(f3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        f2 = self.fusion_layers[1](torch.cat([p2, f3_up], dim=1))
        
        # P2' -> P1
        f2_up = F.interpolate(f2, size=p1.shape[-2:], mode='bilinear', align_corners=False)
        f1 = self.fusion_layers[0](torch.cat([p1, f2_up], dim=1))
        
        # Apply DuSA at the finest scale (1/4) with Depth-guided edges
        refined = self.dusa(f1, depth=depth)
        
        # Classifier
        out = self.cls_seg(refined)
        return out
