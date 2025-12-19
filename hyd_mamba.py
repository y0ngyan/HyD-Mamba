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
        
        # 1. Encoder (V2.2 Grafted Architecture)
        self.dims = [16, 24, 128, 256]
        self.encoder = HyDEncoder(
            in_chans=3, 
            embed_dims=self.dims,
            drop_path_rate=0.2
        )
        
        # 2. Decoder (FPN Upgrade)
        self.decoder = HyDHead(
            in_channels_list=self.dims,
            embedding_dim=64,
            num_classes=num_classes
        )
        
        # 3. Auxiliary Head (Stage 3 - 1/16 scale)
        if num_classes > 0:
            self.aux_head = nn.Sequential(
                nn.Conv2d(self.dims[2], self.dims[2], 3, padding=1, bias=False),
                nn.BatchNorm2d(self.dims[2]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(self.dims[2], num_classes, 1)
            )
        else:
            self.aux_head = None
            
        # 4. Initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb, depth):
        # 1. Encode
        features = self.encoder(rgb, depth) # [P1, P2, P3, P4]
        
        # 2. Decode with FPN + DuSA
        logits = self.decoder(features, depth=depth)
        
        # 3. Upsample main output
        out = F.interpolate(logits, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.training and self.aux_head is not None:
            # Auxiliary output from Stage 3 (1/16 scale)
            aux_logits = self.aux_head(features[2])
            aux_out = F.interpolate(aux_logits, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
            return out, aux_out
            
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
