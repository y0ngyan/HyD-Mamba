import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast  # AMP
import json
import os
import argparse
from tqdm import tqdm
import sys
import time
from datetime import datetime
import numpy as np
import copy

# Paths
sys.path.append(os.path.dirname(__file__))

from hyd_mamba import HyDNet
from datasets.nyuv2_adapter import NYUv2Hyd
from tools.losses import DetailAggregateLoss, OhemCrossEntropyLoss

# NYUv2 40-class names (for TensorBoard)
NYU40_CLASSES = [
    'void', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser',
    'pillow', 'mirror', 'floor_mat', 'clothes', 'ceiling', 'books', 'fridge', 'tv',
    'paper', 'towel', 'shower_curtain', 'box', 'whiteboard', 'person', 'night_stand',
    'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop'
]

def get_grad_norm(model):
    """Calculate total gradient norm of model parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay
        
    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
            
    def state_dict(self):
        return self.model.state_dict()

def log_predictions(writer, images, depths, labels, preds, epoch, num_samples=2):
    """Log prediction visualizations to TensorBoard."""
    import torchvision.utils as vutils
    
    # Denormalize RGB
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images_denorm = images * std + mean
    
    for i in range(min(num_samples, images.shape[0])):
        writer.add_image(f'Sample_{i}/RGB', images_denorm[i], epoch)
        writer.add_image(f'Sample_{i}/Depth', depths[i] / depths[i].max(), epoch)
        
        # Color-code labels and preds (simple grayscale for now)
        label_viz = labels[i].float() / 40.0
        pred_viz = preds[i].float() / 40.0
        writer.add_image(f'Sample_{i}/GT', label_viz.unsqueeze(0), epoch)
        writer.add_image(f'Sample_{i}/Pred', pred_viz.unsqueeze(0), epoch)

def train(cfg_path):
    # 1. Load Config
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Setup TensorBoard
    run_name = f"{cfg['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join('runs', run_name)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # Log hyperparameters
    writer.add_text('Config', json.dumps(cfg, indent=2))
    
    # 3. Dataset
    print("Initializing Dataset...")
    train_set = NYUv2Hyd(cfg, mode='train', do_aug=True)
    val_set = NYUv2Hyd(cfg, mode='test', do_aug=False)
    
    train_loader = DataLoader(train_set, batch_size=cfg['ims_per_gpu'], shuffle=True, 
                              num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, 
                            num_workers=cfg['num_workers'], pin_memory=True)
    
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    
    # 4. Model
    print(f"Initializing HyDNet (Classes: {cfg['n_classes']})...")
    model = HyDNet(num_classes=cfg['n_classes'])
    
    # --- Weight Grafting (v2.2) ---
    mamba_ckpt = cfg.get('mamba_ckpt')
    if mamba_ckpt and hasattr(model.encoder, 'load_grafted_weights'):
        model.encoder.load_grafted_weights(mamba_ckpt)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M")
    writer.add_text('Model', f'Total: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M')
    
    # 5. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    
    # Warmup + Polynomial LR Scheduler
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linear warmup
        else:
            # Polynomial decay after warmup
            progress = (epoch - warmup_epochs) / (cfg['epochs'] - warmup_epochs)
            return (1 - progress) ** cfg['lr_power']
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 6. Loss Selection (From Config)
    if cfg.get('loss') == 'detail_aggregate':
        criterion = DetailAggregateLoss(ignore_index=0)
        print("Using DetailAggregateLoss (Depth-guided)")
    else:
        criterion = OhemCrossEntropyLoss(thres=0.7, min_kept=10000, ignore_index=0)
        print("Using OHEM Loss (thresh=0.7)")
    
    # 7. AMP Scaler for mixed precision
    scaler = GradScaler()
    print("Mixed Precision (AMP) enabled")
    
    # 8. EMA Model
    ema = EMA(model, decay=0.999)
    print("EMA model enabled (decay=0.999)")
    
    # 9. Training Loop
    best_iou = 0.0
    output_dir = os.path.join('checkpoints', cfg['model_name'])
    os.makedirs(output_dir, exist_ok=True)
    
    global_step = 0
    training_history = {'train_loss': [], 'val_miou': [], 'lr': [], 'grad_norm': []}
    
    # 10. Gradient Accumulation
    accumulation_steps = cfg.get('accumulation_steps', 1)  # Default no accumulation
    effective_batch_size = cfg['ims_per_gpu'] * accumulation_steps
    print(f"Gradient Accumulation: {accumulation_steps} steps (effective batch size: {effective_batch_size})")
    
    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0
        epoch_grad_norms = []
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        
        # --- Handle Staged Fine-tuning (V2.2 Freeze Logic) ---
        freeze_epochs = cfg.get('freeze_backbone_epochs', 0)
        if epoch < freeze_epochs:
            # Stage 1: Freeze PRE-TRAINED RGB Encoder parts
            # Only freeze parameters that were actually grafted
            # We keep DepthStream, MambaFusion, and and Bridge/Downsample layers trainable
            frozen_count = 0
            for name, param in model.encoder.named_parameters():
                # Selective freeze: stage blocks are pre-trained (except local branches maybe? 
                # but we usually freeze the whole pre-trained block).
                # However, MambaFusion and depth_stream are DEFINITELY not pre-trained.
                if 'depth_' in name or 'fuse' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    frozen_count += 1
            
            pbar.set_description(f"Epoch {epoch+1}/{cfg['epochs']} [FREEZE {frozen_count}]")
        else:
            # Stage 2: Full Fine-tuning
            for param in model.parameters():
                param.requires_grad = True
            pbar.set_description(f"Epoch {epoch+1}/{cfg['epochs']} [FULL]")

        optimizer.zero_grad()  # Zero gradients at epoch start
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            labels = batch['label'].to(device)
            
            # Mixed Precision Forward
            with autocast():
                # Model now returns (main, aux) during training
                model_outs = model(images, depths)
                
                if isinstance(model_outs, tuple):
                    main_outs, aux_outs = model_outs
                else:
                    main_outs, aux_outs = model_outs, None
                
                # Check shapes for main
                if main_outs.shape[-2:] != labels.shape[-2:]:
                    main_outs = nn.functional.interpolate(main_outs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                
                # Main Loss
                if cfg.get('loss') == 'detail_aggregate':
                    main_loss = criterion(main_outs, labels, depths)
                else:
                    main_loss = criterion(main_outs, labels)
                
                # Auxiliary Loss (if exists)
                if aux_outs is not None:
                    if aux_outs.shape[-2:] != labels.shape[-2:]:
                        aux_outs = nn.functional.interpolate(aux_outs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                    
                    if cfg.get('loss') == 'detail_aggregate':
                        aux_loss = criterion(aux_outs, labels, depths)
                    else:
                        aux_loss = criterion(aux_outs, labels)
                    
                    loss = main_loss + 0.4 * aux_loss
                else:
                    loss = main_loss
                
                loss = loss / accumulation_steps  # Scale loss for accumulation
            
            # Mixed Precision Backward (accumulate gradients)
            scaler.scale(loss).backward()
            
            # Only update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                # Record gradient norm
                grad_norm = get_grad_norm(model)
                epoch_grad_norms.append(grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Reset gradients after update
                
                # EMA update
                ema.update(model)
                
                global_step += 1
                
                # Log per-step metrics (every 10 actual updates)
                if global_step % 10 == 0:
                    writer.add_scalar('Train/Loss_step', loss.item() * accumulation_steps, global_step)
                    writer.add_scalar('Train/GradNorm_step', grad_norm, global_step)
            
            epoch_loss += loss.item() * accumulation_steps  # Unscale for logging
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}', 'acc': f'{(batch_idx+1) % accumulation_steps + 1}/{accumulation_steps}'})
        
        # Epoch metrics
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)
        avg_grad = np.mean(epoch_grad_norms)
        current_lr = scheduler.get_last_lr()[0]
        
        # Log epoch metrics
        writer.add_scalar('Train/Loss_epoch', avg_loss, epoch)
        writer.add_scalar('Train/GradNorm_epoch', avg_grad, epoch)
        writer.add_scalar('Train/LR', current_lr, epoch)
        writer.add_scalar('Train/EpochTime_sec', epoch_time, epoch)
        
        training_history['train_loss'].append(avg_loss)
        training_history['lr'].append(current_lr)
        training_history['grad_norm'].append(avg_grad)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Grad: {avg_grad:.2f} | Time: {epoch_time:.1f}s")
        
        scheduler.step()
        
        # Validation (Every 5 epochs or last epoch)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == cfg['epochs']:
            val_miou, class_ious = validate(model, val_loader, device, cfg['n_classes'])
            
            # Log validation metrics
            writer.add_scalar('Val/mIoU', val_miou, epoch)
            training_history['val_miou'].append(val_miou)
            
            # Log per-class IoU
            for i, iou in enumerate(class_ious):
                if i > 0 and i < len(NYU40_CLASSES):  # Skip void
                    writer.add_scalar(f'Val_Class/{NYU40_CLASSES[i]}', iou, epoch)
            
            # Log prediction visualizations
            model.eval()
            with torch.no_grad():
                sample = next(iter(val_loader))
                imgs = sample['image'].to(device)
                deps = sample['depth'].to(device)
                lbls = sample['label'].to(device)
                outs = model(imgs, deps)
                preds = torch.argmax(outs, dim=1)
                log_predictions(writer, imgs, deps, lbls, preds, epoch)
            
            print(f"Validation mIoU: {val_miou:.4f}")
            
            # Log worst classes
            valid_ious = [(NYU40_CLASSES[i], class_ious[i]) for i in range(1, min(len(class_ious), len(NYU40_CLASSES)))]
            valid_ious.sort(key=lambda x: x[1])
            print(f"Worst 5 classes: {valid_ious[:5]}")
            
            if val_miou > best_iou:
                best_iou = val_miou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_miou': best_iou,
                    'config': cfg
                }, os.path.join(output_dir, 'best_model.pth'))
                print("✓ Saved Best Model!")
                
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': cfg
        }, os.path.join(output_dir, 'latest.pth'))
    
    # Save training summary
    summary = {
        'config': cfg,
        'best_miou': best_iou,
        'total_epochs': cfg['epochs'],
        'total_params': total_params,
        'history': training_history
    }
    torch.save(summary, os.path.join(output_dir, 'training_summary.pth'))
    
    writer.close()
    print(f"\n{'='*50}")
    print(f"Training Complete! Best mIoU: {best_iou:.4f}")
    print(f"Checkpoints: {output_dir}")
    print(f"TensorBoard: tensorboard --logdir runs")
    print(f"{'='*50}")

def validate(model, loader, device, num_classes):
    """Validate and return mIoU + per-class IoU."""
    model.eval()
    intersection = torch.zeros(num_classes).to(device)
    union = torch.zeros(num_classes).to(device)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, depths)
            preds = torch.argmax(outputs, dim=1)
            
            valid = (labels != 0)
            preds_v = preds[valid]
            labels_v = labels[valid]
            
            for i in range(1, num_classes):
                pred_i = (preds_v == i)
                label_i = (labels_v == i)
                intersection[i] += (pred_i & label_i).sum()
                union[i] += (pred_i | label_i).sum()
                
    iou = intersection / (union + 1e-6)
    miou = iou[1:].mean().item()
    return miou, iou.cpu().numpy()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
    else:
        cfg_path = "configs/nyuv2_hyd.json"
        
    train(cfg_path)

