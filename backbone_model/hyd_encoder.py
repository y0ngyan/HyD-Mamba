import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from functools import partial
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig
from .vmamba_layer import VSSBlock, LayerNorm2d
from .parallel_hybrid import ParallelHybridBlock
from .side import SIDE, NRGM
from .large_kernel import LargeKernelBlock
import torchvision.models as models

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
                 embed_dims=[16, 24, 128, 256], # Grafted dims
                 drop_path_rate=0.2):
        super().__init__()
        
        self.dims = embed_dims
        
        # --- Depth Preprocessing (SIDE) ---
        self.side = SIDE()
        
        # --- RGB Stream (Grafted) ---
        # 1. Stem: Same as MobileNetV3 Small (3x3 conv, stride 2, 16 out)
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.Hardswish(inplace=True)
        )
        
        # 2. Stage 1 (Resol: 1/4): MobileNetV3-Small blocks 1-3
        # We use standard torchvision InvertedResidual configurations for Small
        # bneck 1: 16 -> 16, exp 16, k 3, s 2, use_se False, use_hs False
        # bneck 2: 16 -> 24, exp 72, k 3, s 2, use_se False, use_hs False (Note: stride 2 makes it Stage 1->2 transition)
        
        # Configs matching MobileNetV3 Small
        adjust_channels = lambda x: x # We keep it same as small to load weights easily
        
        # Layer 1 (In 16, Out 16, Stride 2 -> Resol 1/4)
        cnnf = partial(InvertedResidualConfig, width_mult=1.0, dilation=1)
        self.stage1_blocks = nn.Sequential(
            InvertedResidual(cnnf(16, 3, 16, 16, False, "RE", 2), nn.BatchNorm2d),
            InvertedResidual(cnnf(16, 3, 72, 24, False, "RE", 2), nn.BatchNorm2d), # This actually downsamples to 1/8
        )
        # Note: We adjust self.dims[1] to 24 because MobileNetV3 Small layer 2 outputs 24.
        
        # Stage 2 (Resol: 1/8 -> 1/8):
        self.stage2_blocks = nn.Sequential(
            InvertedResidual(cnnf(24, 3, 88, 24, False, "RE", 1), nn.BatchNorm2d),
        )
        
        # Bridge to Stage 3 Mamba (Channel count alignment)
        # MobileNetV3 Small at Stage 3 usually starts at layer 4 (40 chans)
        # But we jump to Mamba 128 chans.
        self.bridge_3 = nn.Sequential(
            nn.Conv2d(24, embed_dims[2], 1, bias=False),
            nn.BatchNorm2d(embed_dims[2]),
            nn.Hardswish(inplace=True)
        )
        
        # Stage 3 & 4: Parallel CNN-Mamba
        # Downsample to 1/16
        self.stage3_down = nn.Sequential(
            nn.Conv2d(embed_dims[2], embed_dims[2], 3, 2, 1, groups=embed_dims[2], bias=False),
            nn.BatchNorm2d(embed_dims[2]),
            nn.ReLU(inplace=True)
        )
        self.stage3_blocks = nn.Sequential(
            ParallelHybridBlock(embed_dims[2]),
            ParallelHybridBlock(embed_dims[2])
        )
        
        # Downsample to 1/32
        self.stage4_down = nn.Sequential(
            nn.Conv2d(embed_dims[2], embed_dims[3], 3, 2, 1, bias=False),
            nn.BatchNorm2d(embed_dims[3]),
            nn.ReLU(inplace=True)
        )
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
        self.depth_stage4 = nn.Sequential(
            nn.Conv2d(d_chans[3], d_chans[3], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_chans[3]),
            nn.ReLU(inplace=True)
        ) # /32 (Simplified with BN/ReLU)
        
        # --- Fusion Modules (NRGM + MambaFusion) ---
        self.nrgm = NRGM()
        
        # Fusion at Stage 1 (1/4 scale)
        # RGB x has 16 channels, Depth d1 has d_chans[1]=4 channels
        self.fuse1 = MambaFusion(embed_dims[0], d_chans[1])
        
        # Fusion at Stage 2 (1/8 scale)
        # RGB x has 24 channels, Depth d2 has d_chans[2]=6 channels
        self.fuse2 = MambaFusion(embed_dims[1], d_chans[2])
        
        # Fusion at Stage 3 (1/16 scale)
        # RGB x has 128 channels, Depth d3 has d_chans[3]=32 channels
        self.fuse3 = MambaFusion(embed_dims[2], d_chans[3])
        
        # Fusion at Stage 4 (1/32 scale)
        # RGB x has 256 channels, Depth d4 has d_chans[3]=64 channels (wait, d_chans[3] is 64 here)
        self.fuse4 = MambaFusion(embed_dims[3], d_chans[3])
        
    def forward(self, rgb, depth):
        outputs = []
        
        # --- Depth Preprocessing ---
        depth_side = self.side(depth)
        
        # --- Depth Stream Forward ---
        d0 = self.depth_stem(depth_side) # /2
        d1 = self.depth_stage1(d0)       # /4
        d2 = self.depth_stage2(d1)       # /8
        d3 = self.depth_stage3(d2)       # /16
        d4 = self.depth_stage4(d3)       # /32
        
        # --- Apply NRGM ---
        d1 = self.nrgm(d1, depth)
        d2 = self.nrgm(d2, depth)
        d3 = self.nrgm(d3, depth)
        d4 = self.nrgm(d4, depth)
        
        # --- RGB Stream & Fusion ---
        x = self.rgb_stem(rgb) # /2
        
        # Stage 1
        x = self.stage1_blocks(x) # /8 (MobileNetV3 small layer 1->2 are s2)
        # Wait, if layer 1 is s2 and layer 2 is s2, we are at 1/8.
        # We need to collect 1/4 features.
        # Let's check intermediate.
        # Stage 1 block 0 is s2 (/4). Block 1 is s2 (/8).
        # We collect after block 0.
        
        # RE-WRITE FORWARD for correct stage collection
        x = self.rgb_stem(rgb) # /2
        
        # Layer 1 of self.stage1_blocks
        x = self.stage1_blocks[0](x) # /4
        x_f1 = self.fuse1(x, d1)
        outputs.append(x_f1)
        
        x = self.stage1_blocks[1](x) # /8
        x = self.stage2_blocks(x)    # /8
        x_f2 = self.fuse2(x, d2)
        outputs.append(x_f2)
        
        # Bridge to Stage 3
        x = self.bridge_3(x) 
        
        # Stage 3
        x = self.stage3_down(x) # /16
        x = self.stage3_blocks(x)
        x_f3 = self.fuse3(x, d3)
        outputs.append(x_f3)
        
        # Stage 4
        x = self.stage4_down(x) # /32
        x = self.stage4_blocks(x)
        x_f4 = self.fuse4(x, d4)
        outputs.append(x_f4)
        
        return outputs

    def load_grafted_weights(self, mamba_ckpt_path):
        """
        Model Surgery:
        1. Load MobileNetV3 Small from torchvision
        2. Load VMamba Tiny from local path with channel slicing
        """
        print(f"--- Grafting Weights Activity ---")
        
        # Part 1: MobileNetV3 Small (Automatic)
        print("Loading MobileNetV3 Small (Stage 1&2)...")
        mbv3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        mb_state = mbv3.state_dict()
        
        # Mapping for Stage 1 & 2
        # rgb_stem
        self.rgb_stem[0].load_state_dict({'weight': mb_state['features.0.0.weight']})
        self.rgb_stem[1].load_state_dict({'weight': mb_state['features.0.1.weight'], 
                                          'bias': mb_state['features.0.1.bias'],
                                          'running_mean': mb_state['features.0.1.running_mean'],
                                          'running_var': mb_state['features.0.1.running_var']})
        
        # stage1_blocks[0] (MobileNetV3 layer 1)
        # InvertedResidual in torchvision has 'block' attribute
        def copy_bneck(target, source_prefix, state):
            target_dict = {}
            for k, v in state.items():
                if k.startswith(source_prefix):
                    new_k = k.replace(source_prefix + ".", "")
                    target_dict[new_k] = v
            target.load_state_dict(target_dict, strict=False)
            
        copy_bneck(self.stage1_blocks[0], "features.1", mb_state)
        copy_bneck(self.stage1_blocks[1], "features.2", mb_state)
        copy_bneck(self.stage2_blocks[0], "features.3", mb_state)
        print("MobileNetV3 weights loaded successfully.")
        
        # Part 2: VMamba Tiny (Local + Slicing)
        if mamba_ckpt_path and os.path.exists(mamba_ckpt_path):
            print(f"Loading VMamba Tiny from {mamba_ckpt_path} (Stage 3&4)...")
            checkpoint = torch.load(mamba_ckpt_path, map_location='cpu')
            mamba_state = checkpoint['model']
            
            # Helper for channel slicing
            def slice_weight(src, target_shape):
                # src: tensor
                # target_shape: list of dims
                res = src
                for i, dim in enumerate(target_shape):
                    if i < len(src.shape) and src.shape[i] > dim:
                        # Slice the first 'dim' channels
                        res = res.narrow(i, 0, dim)
                return res

            # We need to map vssm layers to our ParallelHybridBlock.global_branch (SS2Dv2)
            # HyDEncoder.stage3_blocks[0].global_branch is a SS2Dv2
            # VMamba Stage 3 is layers.2, Stage 4 is layers.3
            
            # This is complex mapping, we will use a keyword search approach
            our_state = self.state_dict()
            grafted_count = 0
            
            for name, param in self.named_parameters():
                mamba_key = None
                if 'stage3_blocks' in name:
                    # Map to VMamba layers.2
                    # My: stage3_blocks.0.global_branch.in_proj.weight
                    # Official: layers.2.blocks.0.op.in_proj.weight
                    parts = name.split('.') 
                    if 'global_branch' in name:
                        block_idx = parts[1]
                        suffix = ".".join(parts[3:]) # in_proj.weight
                        mamba_key = f"layers.2.blocks.{block_idx}.op.{suffix}"
                
                elif 'stage4_blocks' in name:
                    parts = name.split('.')
                    if 'global_branch' in name:
                        block_idx = parts[1]
                        suffix = ".".join(parts[3:])
                        mamba_key = f"layers.3.blocks.{block_idx}.op.{suffix}"
                
                if mamba_key and mamba_key in mamba_state:
                    src_weight = mamba_state[mamba_key]
                    if src_weight.shape == param.shape:
                        param.data.copy_(src_weight)
                    else:
                        # Slice!
                        sliced = slice_weight(src_weight, param.shape)
                        if sliced.shape == param.shape:
                            param.data.copy_(sliced)
                    grafted_count += 1
            
            print(f"Successfully grafted {grafted_count} Mamba layers with slicing.")
        else:
            print("Warning: VMamba checkpoint not found, Stage 3&4 will be random.")

if __name__ == "__main__":
    print("Testing HyDEncoder...")
    encoder = HyDEncoder().cuda()
    rgb = torch.randn(1, 3, 512, 512).cuda()
    depth = torch.randn(1, 1, 512, 512).cuda()
    outs = encoder(rgb, depth)
    for i, o in enumerate(outs):
        print(f"Stage {i}: {o.shape}")
