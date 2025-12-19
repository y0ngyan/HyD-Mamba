import torch
import torch.nn as nn
from .vmamba_layer import VSSBlock, LayerNorm2d
from .parallel_hybrid import ParallelHybridBlock
from .side import SIDE, NRGM
from .large_kernel import LargeKernelBlock

class MambaFusion(nn.Module):
    """
    Geometry-Aware Mamba Fusion
    Structure: Concat(RGB, Depth) -> Linear -> VSSBlock
    """
    def __init__(self, rgb_dim, depth_dim):
        super().__init__()
        self.proj = nn.Conv2d(rgb_dim + depth_dim, rgb_dim, kernel_size=1, bias=False)
        self.norm = LayerNorm2d(rgb_dim)
        # Use VSSBlock for selective scanning fusion
        self.mamba = VSSBlock(hidden_dim=rgb_dim, drop_path=0.0)
        
    def forward(self, rgb, depth):
        # rgb: (B, C, H, W)
        # depth: (B, C_d, H, W)
        x = torch.cat([rgb, depth], dim=1)
        x = self.proj(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # N H W C
        x = self.mamba(x)
        x = x.permute(0, 3, 1, 2) # N C H W
        return x

class HyDEncoder(nn.Module):
    def __init__(self, 
                 in_chans=3, 
                 embed_dims=[64, 128, 256, 512], 
                 drop_path_rate=0.2):
        super().__init__()
        
        # --- Depth Preprocessing (SIDE) ---
        self.side = SIDE()
        
        # --- RGB Stream (Mix of CNN and Mamba) ---
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims[0], embed_dims[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
        )
        
        # Stage 1 & 2: CNN (MobileNet-style or Simple Blocks)
        # Here keeping it simple matching previous HyD-Mamba logic but cleaner
        self.stage1_blocks = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[1], 3, 2, 1, bias=False),
            nn.BatchNorm2d(embed_dims[1]),
            nn.ReLU(inplace=True)
        )
        self.stage2_blocks = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[2], 3, 2, 1, bias=False),
            nn.BatchNorm2d(embed_dims[2]),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3 & 4: Parallel CNN-Mamba (Deeper: 2 blocks each for better global context)
        self.stage3_down = nn.Conv2d(embed_dims[2], embed_dims[3], 3, 2, 1)
        self.stage3_blocks = nn.Sequential(
            ParallelHybridBlock(embed_dims[3]),
            ParallelHybridBlock(embed_dims[3])
        )
        
        # Stage 4
        self.stage4_down = nn.Conv2d(embed_dims[3], embed_dims[3], 3, 2, 1) # Downsample
        self.stage4_blocks = nn.Sequential(
            ParallelHybridBlock(embed_dims[3]),
            ParallelHybridBlock(embed_dims[3])
        )
        
        # --- Depth Stream (Large Kernel) ---
        # Channels 1/4 of RGB
        d_chans = [c // 4 for c in embed_dims]
        
        # Depth Stem: SIDE outputs 1 channel, map to d_chans[0]
        # Input depth resolution is H,W. RGB stem downsamples /2. 
        # Depth Stream needs to match resolution? 
        # HDBFormer suggests maintaining resolution or matching stages.
        # We will match RGB stages.
        self.depth_stem = nn.Sequential(
            nn.Conv2d(1, d_chans[0], 3, 2, 1, bias=False), # /2
            nn.BatchNorm2d(d_chans[0]),
            nn.ReLU(inplace=True)
        )
        
        self.depth_stage1 = LargeKernelBlock(d_chans[0], d_chans[1], stride=2) # /4
        self.depth_stage2 = LargeKernelBlock(d_chans[1], d_chans[2], stride=2) # /8
        self.depth_stage3 = LargeKernelBlock(d_chans[2], d_chans[3], stride=2) # /16
        self.depth_stage4 = LargeKernelBlock(d_chans[3], d_chans[3], stride=2) # /32
        
        # --- Fusion Modules (NRGM + MambaFusion) ---
        self.nrgm = NRGM()
        
        # Fusion at Stage 1 (1/4 scale)
        self.fuse1 = MambaFusion(embed_dims[1], d_chans[1])
        # Fusion at Stage 2 (1/8 scale)
        self.fuse2 = MambaFusion(embed_dims[2], d_chans[2])
        # Fusion at Stage 3 (1/16 scale)
        self.fuse3 = MambaFusion(embed_dims[3], d_chans[3])
        # Fusion at Stage 4 (1/32 scale)
        self.fuse4 = MambaFusion(embed_dims[3], d_chans[3])
        
    def forward(self, rgb, depth):
        # rgb: (B, 3, H, W)
        # depth: (B, 1, H, W) - Raw Depth
        outputs = []
        
        # --- Depth Preprocessing ---
        # 1. SIDE: Log + InstNorm
        depth_side = self.side(depth)
        
        # --- Depth Stream Forward ---
        d0 = self.depth_stem(depth_side) # /2
        d1 = self.depth_stage1(d0)       # /4
        d2 = self.depth_stage2(d1)       # /8
        d3 = self.depth_stage3(d2)       # /16
        d4 = self.depth_stage4(d3)       # /32
        
        # --- Apply NRGM ---
        # Suppress noise using raw depth mask
        # Note: raw depth needs to be downsampled to match d1, d2...
        # We can pass raw_depth to NRGM and it will interpolate.
        d1 = self.nrgm(d1, depth)
        d2 = self.nrgm(d2, depth)
        d3 = self.nrgm(d3, depth)
        d4 = self.nrgm(d4, depth)
        
        # --- RGB Stream & Fusion ---
        x = self.rgb_stem(rgb) # /2
        
        # Stage 1
        x = self.stage1_blocks(x) # /4
        x = self.fuse1(x, d1)
        outputs.append(x)
        
        # Stage 2
        x = self.stage2_blocks(x) # /8
        x = self.fuse2(x, d2)
        outputs.append(x)
        
        # Stage 3
        x = self.stage3_down(x) # /16
        x = self.stage3_blocks(x)
        x = self.fuse3(x, d3)
        outputs.append(x)
        
        # Stage 4
        x = self.stage4_down(x) # /32
        x = self.stage4_blocks(x)
        x = self.fuse4(x, d4)
        outputs.append(x)
        
        return outputs

if __name__ == "__main__":
    print("Testing HyDEncoder...")
    encoder = HyDEncoder().cuda()
    rgb = torch.randn(1, 3, 512, 512).cuda()
    depth = torch.randn(1, 1, 512, 512).cuda()
    outs = encoder(rgb, depth)
    for i, o in enumerate(outs):
        print(f"Stage {i}: {o.shape}")
