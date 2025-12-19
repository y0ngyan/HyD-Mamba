import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) # Add root

# Since folder has hyphen, we add the folder itself to path so we can import 'hyd_mamba' directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hyd_mamba
from tools.losses import DetailAggregateLoss, OhemCrossEntropyLoss

class SyntheticRGBDDataset(Dataset):
    """
    Generates synthetic RGB-D images with simple geometric shapes.
    - RGB: Colored shapes on background.
    - Depth: Shapes have different 'depth' values than background.
    - Mask: Class ID of the shape.
    """
    def __init__(self, size=256, num_samples=100):
        self.size = size
        self.num_samples = num_samples
        self.num_classes = 3 # Background, Circle, Square

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Background
        rgb = np.zeros((3, self.size, self.size), dtype=np.float32)
        depth = np.zeros((1, self.size, self.size), dtype=np.float32) + 0.1
        mask = np.zeros((self.size, self.size), dtype=np.longlong)

        # 2. Add a geometric shape (Square) - Class 1
        x, y = random.randint(0, self.size-50), random.randint(0, self.size-50)
        h, w = random.randint(20, 50), random.randint(20, 50)
        
        rgb[:, y:y+h, x:x+w] = 1.0 # White square
        depth[:, y:y+h, x:x+w] = 0.5 # Mid depth
        mask[y:y+h, x:x+w] = 1

        # 3. Add a geometric shape (Circle) - Class 2
        cx, cy = random.randint(20, self.size-20), random.randint(20, self.size-20)
        r = random.randint(10, 30)
        y_grid, x_grid = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        circle_mask = dist <= r
        
        rgb[0, circle_mask] = 1.0 # Red
        rgb[1, circle_mask] = 0.0
        rgb[2, circle_mask] = 0.0
        depth[0, circle_mask] = 0.9 # Close depth
        mask[circle_mask] = 2

        return torch.tensor(rgb), torch.tensor(depth), torch.tensor(mask)

def train_synthetic():
    print("="*50)
    print("Starting Feasibility Verification with Synthetic Data")
    print("Goal: Verify finding convergence (Loss going down)")
    print("="*50)

    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    dataset = SyntheticRGBDDataset(size=256, num_samples=50) # Small sample
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. Model
    model = hyd_mamba.HyDNet(num_classes=3).to(device)
    model.train()
    
    # 3. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = DetailAggregateLoss(ignore_index=255)
    
    # 4. Loop
    print("Test 1: OHEM Loss Robustness")
    ohem_criterion = OhemCrossEntropyLoss(ignore_index=255)
    epochs = 2
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (rgb, depth, mask) in enumerate(dataloader):
            rgb, depth, mask = rgb.to(device), depth.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(rgb, depth)
            
            # Handle dual outputs
            if isinstance(outputs, tuple):
                main_out, aux_out = outputs
                loss = ohem_criterion(main_out, mask) + 0.4 * ohem_criterion(aux_out, mask)
            else:
                loss = ohem_criterion(outputs, mask)
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] OHEM Loss: {epoch_loss / len(dataloader):.4f}")

    print("Test 2: DetailAggregateLoss Convergence")
    detail_criterion = DetailAggregateLoss(ignore_index=255)
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (rgb, depth, mask) in enumerate(dataloader):
            rgb, depth, mask = rgb.to(device), depth.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(rgb, depth)
            
            # Handle dual outputs
            if isinstance(outputs, tuple):
                main_out, aux_out = outputs
                loss = detail_criterion(main_out, mask, depth) + 0.4 * detail_criterion(aux_out, mask, depth)
            else:
                loss = detail_criterion(outputs, mask, depth)
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Detail Loss: {avg_loss:.4f}")
        
    print("="*50)
    if avg_loss < 0.5:
        print("VERIFICATION SUCCESS: Model is learning effectively!")
    else:
        print("VERIFICATION WARNING: Loss is high. Check hyperparameters.")

if __name__ == "__main__":
    train_synthetic()
