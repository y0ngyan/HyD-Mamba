import numpy as np
import os
import json
from PIL import Image
from tqdm import tqdm

def get_standard_mapping(names):
    """
    Manually defined semantic mapping from 894 NYUv2 names to 40 classes.
    Target classes (void=0, wall=1, ... otherprop=40)
    Mapping logic: fuzzy search for keywords in order of priority.
    """
    nyu40 = [
        'void', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
        'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser',
        'pillow', 'mirror', 'floor_mat', 'clothes', 'ceiling', 'books', 'fridge', 'tv',
        'paper', 'towel', 'shower_curtain', 'box', 'whiteboard', 'person', 'night_stand',
        'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop'
    ]
    
    mapping = np.zeros(len(names) + 1, dtype=np.uint8)
    
    for i, name in enumerate(names):
        old_id = i + 1  # 1-indexed in labels
        name = name.lower().strip()
        new_id = 40 # Default to otherprop
        
        # Priority mapping
        if any(k in name for k in ['wall', 'walls']): new_id = 1
        elif any(k in name for k in ['floor', 'flooring']): new_id = 2
        elif any(k in name for k in ['cabinet', 'cupboard']): new_id = 3
        elif 'bed' in name: new_id = 4
        elif any(k in name for k in ['chair', 'seat', 'stool']): new_id = 5
        elif any(k in name for k in ['sofa', 'couch']): new_id = 6
        elif 'table' in name: new_id = 7
        elif 'door' in name: new_id = 8
        elif 'window' in name: new_id = 9
        elif 'bookshelf' in name: new_id = 10
        elif any(k in name for k in ['picture', 'painting', 'poster', 'frame']): new_id = 11
        elif 'counter' in name: new_id = 12
        elif any(k in name for k in ['blinds', 'shade']): new_id = 13
        elif 'desk' in name: new_id = 14
        elif any(k in name for k in ['shelves', 'shelf']): new_id = 15
        elif 'curtain' in name and 'shower' not in name: new_id = 16
        elif any(k in name for k in ['dresser', 'wardrobe']): new_id = 17
        elif 'pillow' in name: new_id = 18
        elif 'mirror' in name: new_id = 19
        elif any(k in name for k in ['mat', 'rug', 'carpet']): new_id = 20
        elif any(k in name for k in ['clothes', 'clothing', 'garment']): new_id = 21
        elif 'ceiling' in name: new_id = 22
        elif 'book' in name and 'shelf' not in name: new_id = 23
        elif any(k in name for k in ['refrigerator', 'fridge', 'refridgerator']): new_id = 24
        elif any(k in name for k in ['television', 'tv']): new_id = 25
        elif 'paper' in name and 'towel' not in name: new_id = 26
        elif 'towel' in name and 'paper' not in name and 'shower' not in name: new_id = 27
        elif 'shower curtain' in name: new_id = 28
        elif 'box' in name: new_id = 29
        elif 'whiteboard' in name: new_id = 30
        elif 'person' in name: new_id = 31
        elif any(k in name for k in ['night stand', 'nightstand']): new_id = 32
        elif 'toilet' in name: new_id = 33
        elif 'sink' in name: new_id = 34
        elif 'lamp' in name: new_id = 35
        elif 'bathtub' in name: new_id = 36
        elif 'bag' in name: new_id = 37
        elif any(k in name for k in ['stairs', 'railing', 'column', 'pipe', 'beam', 'support']): new_id = 38
        elif any(k in name for k in ['furniture', 'stand', 'ottoman']): new_id = 39
        elif name in ['void', 'unknown', 'none', '']: new_id = 0
        
        mapping[old_id] = new_id
        
    return mapping

def map_labels(dataset_root, names_file):
    label_dir = os.path.join(dataset_root, 'all_data/label')
    out_dir = os.path.join(dataset_root, 'all_data/label_40')
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(names_file):
        raise FileNotFoundError(f"Names file {names_file} not found. Run extraction script first.")
        
    with open(names_file, 'r') as f:
        names = json.load(f)
    
    mapping = get_standard_mapping(names)
    
    files = sorted(os.listdir(label_dir))
    print(f"Mapping {len(files)} labels using semantic dictionary...")
    
    for f in tqdm(files):
        img_path = os.path.join(label_dir, f)
        lbl = np.array(Image.open(img_path))
        
        # Apply mapping
        # Label can be up to 894, mapping size is 895
        mapped_lbl = mapping[lbl.astype(np.int32)]
        
        Image.fromarray(mapped_lbl).save(os.path.join(out_dir, f))
        
    print(f"Success! Corrected labels saved to {out_dir}")

if __name__ == "__main__":
    # Correction: Use the correct dataset path provided by User
    dataset_dir = "/media/yy/YY2号/deepDatasets/database/nyuv2"
    names_json = "nyu894_names.json"
    map_labels(dataset_dir, names_json)
