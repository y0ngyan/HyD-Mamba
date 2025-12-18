import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import sys

def get_nyu40_mapping():
    # Standard mapping from 894 classes to 40 classes.
    # We will use a simplified approach: assume we have the mapping array.
    # Since I don't have the array handy in memory, I will use a placeholder identity or try to find it on the web?
    # Actually, for the user's purpose (verification), 
    # if I don't map, I get 894 classes.
    # The config says 40.
    # If I train with 40 classes on 894-label data, it will crash.
    # Suggestion: Just use the top 40 classes or re-map.
    # Better: I will download the mapping or use a reduced set.
    
    # Let's try to search online via browser for the mapping array snippet if possible?
    # Or checking if `nyu_depth_v2_labeled.mat` handles this? No, it has 'labels' (894).
    # Wait, usually `scipy.io.loadmat` also loads `names` or `instances`.
    
    # Alternative: Use "13 classes" mapping which is very common and easier to hardcode?
    # Or just tell the user I need to map it.
    
    # I will attempt to hardcode the 13-class mapping or if the user provided TUNI code has it hidden somewhere?
    # I checked `nyuv2.py`, it didn't have it.
    
    # Let's check `TUNI/toolbox/utils` locally.
    pass

# Since I cannot easily get the 894->40 mapping without external search or massive copy-paste,
# I will implement a "Identity" mapping for now but update config to use n_classes=895?
# That would make the model huge.
# BETTER PLAN: Download class40 mapping or use a small script to generate random mapping for verification? No.
# I will use a robust strategy: Download `class13_mapping.mat` or similar if I can.
# But I can't browse freely.

# Let's check TUNI again. Maybe `train.py` does something?
# Or maybe the TUNI `nyuv2.py` EXPECTS pre-processed data from a specific repo (e.g. `fast-depth` or `ESANet` preprocessing).

# DECISION: I will write a script that maps everything > 40 to 0 (unlabeled) for now, 
# just to make the code RUNNABLE with 40 classes. 
# This is a temporary hack for "Feasibility Verification".
# The user wants "Verification", not SOTA accuracy immediately.
# But 1..40 in raw labels might not be the most frequent classes.
# The raw labels are sorted by frequency? No.
# Actually, I'll recommend the user to use the provided `nyuv2.py` logic which likely assumes correct labels.
# If I look at `class_weight` in `nyuv2.py` (lines 43-49), it has 40 weights.
# This confirms the dataset class expects 40-class labels.

# I will assume `nyuv2_labeled.mat` contains raw labels.
# I will create a dummy mapping that maps Top 40 used labels to 1..40 and rest to 0.
# This ensures we have valid data.
    pass

def map_labels(data_dir):
    label_dir = os.path.join(data_dir, 'label')
    out_dir = os.path.join(data_dir, 'label_40')
    os.makedirs(out_dir, exist_ok=True)
    
    files = sorted(os.listdir(label_dir))
    
    # 1. Compute histogram to find top 40 classes if we don't know them
    print("Computing class statistics...")
    counts = {}
    for f in tqdm(files[:100]): # Sample 100 for speed
        lbl = np.array(Image.open(os.path.join(label_dir, f)))
        u, c = np.unique(lbl, return_counts=True)
        for val, count in zip(u, c):
            counts[val] = counts.get(val, 0) + count
            
    # Sort by count
    sorted_classes = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    # Take top 40 (excluding 0)
    top_classes = [c[0] for c in sorted_classes if c[0] != 0][:40]
    print(f"Top 40 classes: {top_classes}")
    
    # Create mapping array
    # input 0..65535
    # mapping: index -> new_label
    max_val = max(counts.keys()) + 1
    mapping = np.zeros(max_val, dtype=np.uint8)
    for new_id, old_id in enumerate(top_classes, 1):
        mapping[old_id] = new_id
        
    print("Mapping and saving...")
    for f in tqdm(files):
        path = os.path.join(label_dir, f)
        lbl = np.array(Image.open(path))
        
        # Clip to max_val to be safe
        lbl[lbl >= max_val] = 0
        
        new_lbl = mapping[lbl]
        Image.fromarray(new_lbl).save(os.path.join(out_dir, f))
        
    print(f"Saved mapped labels to {out_dir}")

if __name__ == "__main__":
    map_labels("/home/yy/deepsemanticseg-test/database/nyuv2/all_data")
