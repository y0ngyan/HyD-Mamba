import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms

# Local imports to ensure independence
from .augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, RandomRotation

class NYUv2Hyd(data.Dataset):
    def __init__(self, cfg, mode='train', do_aug=True):
        assert mode in ['train', 'test', 'val']
        
        # Preprocessing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # Depth is uint16 in my extraction (mm). 
        # But `dp_to_tensor` in TUNI expects some normalization.
        # TUNI `nyuv2.py` assumes depth is opened as RGB? (Line 84: .convert('RGB_T')?? typo in TUNI?)
        # TUNI line 84: `depth = Image.open(...).convert('RGB')` or similar.
        # Actually TUNI likely uses pre-encoded depth?
        # My depth is 1 channel uint16.
        # I should just normalize it.
        # Mean/Std for depth? 
        # DeepLab usually uses raw depth or HHA. 
        # HyD-Mamba DepthStream takes 1 channel.
        # So I will ToTensor() which scales uint16 [0, 65535] to [0, 1] float? 
        # Wait, ToTensor support mode I;16? No.
        # I need to convert to np.float32 manually.
        
        self.root = cfg['root']
        self.n_classes = cfg['n_classes']
        
        # Augmentation
        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))
        
        self.aug = Compose([
            ColorJitter(
                brightness=cfg.get('brightness', 0.5), # Add defaults
                contrast=cfg.get('contrast', 0.5),
                saturation=cfg.get('saturation', 0.5)),
            RandomHorizontalFlip(0.5),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])
        
        self.mode = mode
        self.do_aug = do_aug
        
        # Split
        all_ids = np.arange(1449)
        # 795 train, rest test
        self.train_ids, self.test_ids = train_test_split(all_ids, train_size=795, random_state=3)
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_ids)
        else:
            return len(self.test_ids)
            
    def __getitem__(self, index):
        if self.mode == 'train':
            idx = self.train_ids[index]
        else:
            idx = self.test_ids[index]
            
        # Paths
        # My extraction: all_data/image/{i}.jpg
        image_path = os.path.join(self.root, f'all_data/image/{idx}.jpg')
        # Depth: all_data/depth/{idx}.png
        depth_path = os.path.join(self.root, f'all_data/depth/{idx}.png')
        # Label: all_data/label_40/{idx}.png (Mapped)
        label_path = os.path.join(self.root, f'all_data/label_40/{idx}.png')
        
        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path) # I;16
        label = Image.open(label_path) # L
        
        sample = {
            'image': image,
            'depth': depth,
            'label': label
        }
        
        if self.mode == 'train' and self.do_aug:
            sample = self.aug(sample)
            
        # Convert to Tensor
        # Image -> Tensor normalized
        sample['image'] = self.im_to_tensor(np.array(sample['image']))
        
        # Depth: Convert to float metric (meters) or just normalized?
        # HyD-Mamba DepthStream uses ConvBNAct directly.
        # Just normalize to roughly 0-1 or standard.
        # Raw mm: 0-10000. 
        d_np = np.array(sample['depth']).astype(np.float32)
        # Normalize by max depth (10m = 10000mm)?
        d_np = d_np / 10000.0 
        sample['depth'] = torch.from_numpy(d_np).unsqueeze(0) # (1, H, W)
        
        # Label
        sample['label'] = torch.from_numpy(np.array(sample['label'], dtype=np.int64)).long()
        
        return sample
