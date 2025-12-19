import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from hyd_mamba import HyDNet

def verify():
    print("--- Verifying HyD-Mamba V2.2 (Grafted) ---")
    
    # 1. Initialize Model
    model = HyDNet(num_classes=41)
    
    # 2. Trigger Grafting
    mamba_ckpt = "/home/yy/deepsemanticseg-test/pre-trained/vmamba-tiny/vssm1_tiny_0230s_ckpt_epoch_264.pth"
    if hasattr(model.encoder, 'load_grafted_weights'):
        model.encoder.load_grafted_weights(mamba_ckpt)
    
    # 3. Parameter Count
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total Parameters: {total_params:.2f}M")
    
    # 4. Forward Pass
    model = model.cuda().eval()
    rgb = torch.randn(1, 3, 480, 640).cuda()
    depth = torch.randn(1, 1, 480, 640).cuda()
    
    with torch.no_grad():
        output = model(rgb, depth)
        
    print(f"Input Shape: {rgb.shape}")
    print(f"Output Shape: {output.shape}")
    
    assert output.shape == (1, 41, 480, 640), "Output shape mismatch!"
    print("Verification Successful!")

if __name__ == "__main__":
    verify()
