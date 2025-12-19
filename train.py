import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import argparse
from tqdm import tqdm
import sys

# Paths
sys.path.append(os.path.dirname(__file__))

from hyd_mamba import HyDNet
from datasets.nyuv2_adapter import NYUv2Hyd
from tools.losses import DetailAggregateLoss, OhemCrossEntropyLoss

def train(cfg_path):
    # 1. Load Config
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Dataset
    print("Initializing Dataset...")
    train_set = NYUv2Hyd(cfg, mode='train', do_aug=True)
    val_set = NYUv2Hyd(cfg, mode='test', do_aug=False)
    
    train_loader = DataLoader(train_set, batch_size=cfg['ims_per_gpu'], shuffle=True, 
                              num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, 
                            num_workers=cfg['num_workers'], pin_memory=True)
    
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    
    # 3. Model
    print(f"Initializing HyDNet (Classes: {cfg['n_classes']})...")
    model = HyDNet(num_classes=cfg['n_classes']).to(device)
    
    # 4. Probable Pretraining loading? (Skip for now)
    
    # 5. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=cfg['epochs'], power=cfg['lr_power'])
    
    # 6. Loss
    # Use standard CE for start, or OHEM
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 0 is void/unlabeled
    # Alternatively use our custom loss if needed
    # criterion = OhemCrossEntropyLoss(ignore_label=0)
    
    # 7. Training Loop
    best_iou = 0.0
    output_dir = os.path.join('checkpoints', cfg['model_name'])
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}", dynamic_ncols=True, file=sys.stdout)
        
        for batch in pbar:
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images, depths)
            
            # Outputs might be smaller? usually HyDNet upsamples to input size at end of forward()
            # Check shape
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = nn.functional.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
        
        # Validation (Every 5 epochs)
        if (epoch + 1) % 5 == 0:
            val_iou = validate(model, val_loader, device, cfg['n_classes'])
            print(f"Validation mIoU: {val_iou:.4f}")
            
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print("Saved Best Model!")
                
        # Save latest
        torch.save(model.state_dict(), os.path.join(output_dir, 'latest.pth'))

def validate(model, loader, device, num_classes):
    model.eval()
    intersection = torch.zeros(num_classes).to(device)
    union = torch.zeros(num_classes).to(device)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", dynamic_ncols=True, file=sys.stdout):
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, depths)
            preds = torch.argmax(outputs, dim=1)
            
            # Ignore index 0
            valid = (labels != 0)
            preds = preds[valid]
            labels = labels[valid]
            
            for i in range(1, num_classes): # Skip 0
                pred_i = (preds == i)
                label_i = (labels == i)
                intersection[i] += (pred_i & label_i).sum()
                union[i] += (pred_i | label_i).sum()
                
    iou = intersection / (union + 1e-6)
    miou = iou[1:].mean().item() # Mean over classes 1..N
    return miou

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
    else:
        cfg_path = "configs/nyuv2_hyd.json"
        
    train(cfg_path)
