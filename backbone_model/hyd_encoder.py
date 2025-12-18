import torch
import torch.nn as nn
import torch.nn.functional as F
from .vmamba_layer import Permute, LayerNorm2d
from .parallel_hybrid import ParallelHybridBlock

class ConvBNAct(nn.Module):
    """Simple Convolution -> BN -> Activation block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, act_layer=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DepthStream(nn.Module):
    """
    Shallow Geometry Extractor (Strictly 3 Conv layers).
    Generates multi-scale edge features from just 3 layers by using strides.
    Structure:
    Input -> Conv1 (s2, /2) -> Conv2 (s2, /4) -> Conv3 (s2, /8)
    For deeper requires (/16, /32), we pool the output of Conv3.
    This satisfies "only contains 2-3 layers".
    """
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        # Layer 1: /2
        self.conv1 = ConvBNAct(in_channels, base_channels, stride=2)
        # Layer 2: /4
        self.conv2 = ConvBNAct(base_channels, base_channels*2, stride=2)
        # Layer 3: /8
        self.conv3 = ConvBNAct(base_channels*2, base_channels*4, stride=2)
        
    def forward(self, x):
        # x: H, W
        c1 = self.conv1(x) # /2 (Not utilized in fusion usually, or for stage 0?)
        c2 = self.conv2(c1) # /4 -> Fuse Stage 1
        c3 = self.conv3(c2) # /8 -> Fuse Stage 2
        
        # For Stage 3 (/16) and Stage 4 (/32), we don't have layers (limit 3).
        # We use simple pooling on c3.
        c4 = F.avg_pool2d(c3, kernel_size=2, stride=2) # /16 -> Fuse Stage 3
        c5 = F.avg_pool2d(c4, kernel_size=2, stride=2) # /32 -> Fuse Stage 4
        
        return [c2, c3, c4, c5] # Matching [Stage1, Stage2, Stage3, Stage4]

class GGFM(nn.Module):
    """
    Geometry-Gated Fusion Module.
    Uses depth features to gate RGB features.
    """
    def __init__(self, rgb_dim, depth_dim):
        super().__init__()
        self.depth_proj = nn.Sequential(
            nn.Conv2d(depth_dim, rgb_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_dim),
            nn.Sigmoid()
        )
        
    def forward(self, rgb, depth):
        # rgb: (B, C_rgb, H, W)
        # depth: (B, C_depth, H, W)
        
        gate = self.depth_proj(depth)
        # Enhance RGB where depth indicates importance (edges)
        # F_out = F_rgb + F_rgb * Gate
        return rgb + rgb * gate

class HyDEncoder(nn.Module):
    """
    Asymmetric Encoder with Parallel CNN-Mamba and Shallow Depth Stream.
    """
    def __init__(self, in_chans_rgb=3, in_chans_depth=1, 
                 dims=[32, 64, 128, 256], 
                 # Depth channels determined by DepthStream structure: 16*2=32, 16*4=64...
                 # Fixed to [32, 64, 64, 64] for simplicity below
                 vss_depths=[2, 2], 
                 d_state=16):
        super().__init__()
        
        # --- Depth Stream (Shallow 3-Layer) ---
        base_d = 16
        self.depth_stream = DepthStream(in_channels=in_chans_depth, base_channels=base_d)
        # Output channels:
        # 0: c2 (/4) -> 32
        # 1: c3 (/8) -> 64
        # 2: c4 (/16) -> 64 (pooled)
        # 3: c5 (/32) -> 64 (pooled)
        depth_chans = [32, 64, 64, 64]
        
        # --- RGB Stream ---
        self.fusions = nn.ModuleList()
        
        # Stem
        self.rgb_stem = nn.Sequential(
            ConvBNAct(in_chans_rgb, dims[0]//2, stride=2),
            ConvBNAct(dims[0]//2, dims[0])
        ) 
        
        # Stage 1 (CNN, /4)
        self.stage1 = nn.Sequential(
             ConvBNAct(dims[0], dims[0], stride=2),
             ConvBNAct(dims[0], dims[0])
        )
        self.fusions.append(GGFM(dims[0], depth_chans[0]))

        # Stage 2 (CNN, /8)
        self.stage2 = nn.Sequential(
             ConvBNAct(dims[0], dims[1], stride=2),
             ConvBNAct(dims[1], dims[1])
        )
        self.fusions.append(GGFM(dims[1], depth_chans[1]))

        # Stage 3 (Parallel Hybrid, /16)
        self.stage3_down = ConvBNAct(dims[1], dims[2], stride=2)
        # Parallel Hybrid Blocks
        self.stage3_blocks = nn.Sequential(*[
            ParallelHybridBlock(dim=dims[2], d_state=d_state)
            for _ in range(vss_depths[0])
        ])
        self.fusions.append(GGFM(dims[2], depth_chans[2]))

        # Stage 4 (Parallel Hybrid, /32)
        self.stage4_down = ConvBNAct(dims[2], dims[3], stride=2)
        self.stage4_blocks = nn.Sequential(*[
            ParallelHybridBlock(dim=dims[3], d_state=d_state)
            for _ in range(vss_depths[1])
        ])
        self.fusions.append(GGFM(dims[3], depth_chans[3]))
        
    def forward(self, rgb, depth):
        # 1. Forward Depth Stream
        # Returns list of features at [1/4, 1/8, 1/16, 1/32]
        d_feats = self.depth_stream(depth)
        
        outs = []
        
        # 2. Forward RGB Stream + Fusion
        
        # Stem (/2)
        x = self.rgb_stem(rgb)
        
        # Stage 1 (/4)
        x = self.stage1(x)
        x = self.fusions[0](x, d_feats[0])
        outs.append(x)
        
        # Stage 2 (/8)
        x = self.stage2(x)
        x = self.fusions[1](x, d_feats[1])
        outs.append(x)
        
        # Stage 3 (/16) - Parallel Hybrid
        x = self.stage3_down(x)
        x = self.stage3_blocks(x)
        x = self.fusions[2](x, d_feats[2])
        outs.append(x)
        
        # Stage 4 (/32) - Parallel Hybrid
        x = self.stage4_down(x)
        x = self.stage4_blocks(x)
        x = self.fusions[3](x, d_feats[3])
        outs.append(x)
        
        return outs

if __name__ == "__main__":
    print("Testing HyDEncoder...")
    encoder = HyDEncoder().cuda()
    rgb = torch.randn(1, 3, 512, 512).cuda()
    depth = torch.randn(1, 1, 512, 512).cuda()
    outs = encoder(rgb, depth)
    for i, o in enumerate(outs):
        print(f"Stage {i}: {o.shape}")
