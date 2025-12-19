import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone_model.hyd_encoder import HyDEncoder
from decoder.hyd_head import HyDHead

class HyDNet(nn.Module):
    """
    HyD-Mamba Network: Hybrid Depth-aware Mamba Network for Lightweight RGB-D Segmentation.
    Structure:
    - Encoder: Asymmetric Dual Stream (RGB: CNN+Mamba, Depth: CNN) + Fusion
    - Decoder: MLP Head + DuSA Refinement
    """
    def __init__(self, num_classes=19, pretrained=None):
        super().__init__()
        
        # 1. Encoder (V2 Lightweight)
        # Reduced channels: [32, 64, 128, 256] for ~5M params
        self.dims = [32, 64, 128, 256]
        self.encoder = HyDEncoder(
            in_chans=3, 
            embed_dims=self.dims,
            drop_path_rate=0.2
        )
        
        # 2. Decoder
        # Encoder outputs: Stage1 (dim[1]=64), Stage2 (dim[2]=128), Stage3 (dim[3]=256), Stage4 (dim[3]=256)
        encoder_out_channels = [self.dims[1], self.dims[2], self.dims[3], self.dims[3]]
        self.decoder = HyDHead(
            in_channels_list=encoder_out_channels,
            embedding_dim=64,  # Reduced from 128
            num_classes=num_classes
        )
        
    def forward(self, rgb, depth):
        # 1. Encode
        features = self.encoder(rgb, depth)
        
        # 2. Decode with Depth-guided DuSA
        logits = self.decoder(features, depth=depth)
        
        # 3. Upsample to input size
        out = F.interpolate(logits, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        
        return out

if __name__ == "__main__":
    print("Testing HyDNet (Full Model)...")
    # Simulation
    model = HyDNet(num_classes=19).cuda()
    rgb = torch.randn(1, 3, 512, 512).cuda()
    depth = torch.randn(1, 1, 512, 512).cuda()
    
    output = model(rgb, depth)
    print(f"Input RGB: {rgb.shape}, Depth: {depth.shape}")
    print(f"Output: {output.shape}")
    
    # Check parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params / 1e6:.2f} M")
    
    if output.shape == (1, 19, 512, 512):
        print("Test passed!")
    else:
        print("Test FAILED (Shape mismatch)")
