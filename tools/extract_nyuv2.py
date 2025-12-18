import h5py
import numpy as np
import os
from PIL import Image
import sys
from tqdm import tqdm

def extract_nyu(path_to_mat, output_dir):
    print(f"Loading {path_to_mat} ...")
    # NYUv2 .mat is an HDF5 file
    f = h5py.File(path_to_mat, 'r')
    
    # Structure:
    # images: (1449, 3, 640, 480) uint8
    # depths: (1449, 640, 480) float32
    # labels: (1449, 640, 480) uint16 (1..? probably need mapping)
    
    images = f['images']
    depths = f['depths']
    labels = f['labels']
    
    num_samples = images.shape[0]
    print(f"Found {num_samples} samples.")
    
    # Create dirs
    img_dir = os.path.join(output_dir, 'image')
    depth_dir = os.path.join(output_dir, 'depth')
    label_dir = os.path.join(output_dir, 'label')
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    print("Extracting...")
    for i in tqdm(range(num_samples)):
        # 1. Image
        # HDF5 in Python for NYU matches (N, C, W, H) or (N, C, H, W)?
        # Usually NYU mat is (N, 3, W, H) in Matlab, so (N, 3, 640, 480).
        # We need to transpose to (H, W, C) = (480, 640, 3)
        img = f['images'][i] # (3, 640, 480)
        img = np.transpose(img, (2, 1, 0)) # (480, 640, 3)
        img = Image.fromarray(img)
        img.save(os.path.join(img_dir, f"{i}.jpg"))
        
        # 2. Depth
        depth = f['depths'][i] # (640, 480)
        depth = np.transpose(depth, (1, 0)) # (480, 640)
        # Normalize depth for visualization/saving? 
        # Usually saved as raw float (npy) or 16-bit png. 
        # TUNI nyuv2.py reads .png and converts to RGB_T? 
        # Let's check TUNI loader again: 
        # depth = Image.open(...).convert('RGB_T') ??
        # Wait, usually depth is 1 channel. 
        # Let's save as uint16 png (mm) for standard compat. max depth is ~10m.
        # Max val in nyu is around 10.0 meters.
        # stored as float. 
        # Let's multiply by 1000 -> mm -> uint16.
        depth_uint16 = (depth * 1000).astype(np.uint16)
        depth_img = Image.fromarray(depth_uint16, mode='I;16')
        depth_img.save(os.path.join(depth_dir, f"{i}.png"))
        
        # 3. Label
        label = f['labels'][i] # (640, 480)
        label = np.transpose(label, (1, 0)) # (480, 640)
        # Label is 1-based class ID (1..894).
        # We generally map to 13 classes or 40 classes.
        # For now, let's just save the raw label. TUNI might handle mapping.
        # Saving as uint8 might overflow if > 255 classes.
        # Saving as uint16 or simple png.
        label_img = Image.fromarray(label.astype(np.uint16)) 
        label_img.save(os.path.join(label_dir, f"{i}.png"))
        
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_nyuv2.py <path_to_mat> <output_dir>")
        # Default for the user setup
        mat_path = "/home/yy/deepsemanticseg-test/nyu_depth_v2_labeled.mat"
        out_dir = "/home/yy/deepsemanticseg-test/database/nyuv2/all_data"
        if os.path.exists(mat_path):
            print(f"Using default: {mat_path} -> {out_dir}")
            extract_nyu(mat_path, out_dir)
        else:
            sys.exit(1)
    else:
        extract_nyu(sys.argv[1], sys.argv[2])
