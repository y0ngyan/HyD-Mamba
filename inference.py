import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

# Paths
sys.path.append(os.path.dirname(__file__))

from hyd_mamba import HyDNet
from datasets.nyuv2_adapter import NYUv2Hyd

def colorize_mask(mask, num_classes=41):
    # Create a random color map
    # 0 is strictly black
    np.random.seed(42)
    cmap = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
    cmap[0] = [0, 0, 0]
    
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for c in range(num_classes):
        color_mask[mask == c] = cmap[c]
        
    return color_mask

def inference(cfg_path, model_path, output_dir="inference_results"):
    # 1. Load Config
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Dataset (Test Mode)
    print("Initializing Dataset...")
    val_set = NYUv2Hyd(cfg, mode='test', do_aug=False)
    # Pick 5 random samples
    indices = np.random.choice(len(val_set), 5, replace=False)
    
    # 3. Model
    print(f"Initializing HyDNet (Classes: {cfg['n_classes']})...")
    model = HyDNet(num_classes=cfg['n_classes']).to(device)
    
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Running Inference...")
    with torch.no_grad():
        for idx in tqdm(indices):
            sample = val_set[idx]
            image_check = sample['image'] # Tensor
            depth_check = sample['depth'] # Tensor
            label_check = sample['label'] # Tensor
            
            # Helper to get original image back for viz
            # Un-normalize: (x * std) + mean
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            viz_img = image_check * std + mean
            viz_img = viz_img.numpy().transpose(1, 2, 0) # HWC
            viz_img = (viz_img * 255).clip(0, 255).astype(np.uint8)
            
            # Depth viz
            viz_depth = depth_check.squeeze().numpy()
            viz_depth = (viz_depth / viz_depth.max() * 255).astype(np.uint8)
            
            # Label
            viz_label = colorize_mask(label_check.numpy(), cfg['n_classes'])
            
            # Inference
            inp_img = image_check.unsqueeze(0).to(device)
            inp_depth = depth_check.unsqueeze(0).to(device)
            
            output = model(inp_img, inp_depth) # (1, C, H, W)
            
            # Upsample if needed
            if output.shape[-2:] != label_check.shape[-2:]:
                output = nn.functional.interpolate(output, size=label_check.shape[-2:], mode='bilinear', align_corners=False)
                
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            viz_pred = colorize_mask(pred, cfg['n_classes'])
            
            # Save Combined Image
            # Grid: 2x2 [RGB, Depth]
            #           [GT,  Pred ]
            
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            ax[0, 0].imshow(viz_img)
            ax[0, 0].set_title("Input RGB")
            ax[0, 0].axis('off')
            
            ax[0, 1].imshow(viz_depth, cmap='jet')
            ax[0, 1].set_title("Input Depth")
            ax[0, 1].axis('off')
            
            ax[1, 0].imshow(viz_label)
            ax[1, 0].set_title("Ground Truth")
            ax[1, 0].axis('off')
            
            ax[1, 1].imshow(viz_pred)
            ax[1, 1].set_title("Prediction")
            ax[1, 1].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"result_{idx}.png")
            plt.savefig(save_path)
            plt.close()
            
    print(f"Done! Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/nyuv2_hyd.json')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/hyd_mamba_nyuv2/best_model.pth')
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        # Try finding it automatically?
        # For now assume user follows instruction
        sys.exit(1)
        
    inference(args.config, args.checkpoint)
