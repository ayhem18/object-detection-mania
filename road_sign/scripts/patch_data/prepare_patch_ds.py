import os
import sys
import cv2
import numpy as np
import json
import yaml
import hashlib
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Path setup logic
current_dir = os.path.dirname(os.path.abspath(__file__))
while 'road_sign' not in os.listdir(current_dir):
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:
        break
    current_dir = parent_dir

road_sign_root = os.path.join(current_dir, 'road_sign')
sys.path.insert(0, os.path.join(road_sign_root, 'scripts', 'training', 'patch_based'))
sys.path.insert(0, os.path.dirname(road_sign_root)) # to import home_made_od

# Import the core logic
from mypt.code_utils.pytorch_utils import seed_everything
from prepare_patches import (
    generate_positive_patches, 
    generate_negative_patches, 
    update_labels_for_patch
)
from home_made_od.yolo_family.anchors.anchor_utils import generate_anchors

DEFAULT_CONFIG = {
    "scales": [512, 1024, 2048],
    "min_scale_ratio_threshold": 0.5,
    "min_visibility_threshold": 0.4,
    "K2": 2,
    "target_size": [512, 512],
    "num_anchors": 5
}

def get_config_hash(config_dict):
    """Generates a stable MD5 hash of the configuration dictionary."""
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def process_single_image(args):
    img_path, label_path, target_img_dir, target_label_dir, config = args
    
    img = cv2.imread(img_path)
    if img is None:
        return False
        
    h, w = img.shape[:2]
    
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = [float(x) for x in line.split()]
                if len(parts) >= 5:
                    cls, ncx, ncy, nw, nh = parts[:5]
                    gt_boxes.append([
                        cls, ncx * w, ncy * h, nw * w, nh * h
                    ])
    gt_boxes = np.array(gt_boxes)

    pos_coords = generate_positive_patches((h, w), gt_boxes, config)
    neg_coords = generate_negative_patches(img, gt_boxes, config)
    
    all_patches = [('pos', c) for c in pos_coords] + [('neg', c) for c in neg_coords]
    
    base_name = Path(img_path).stem
    
    img_subdir = os.path.join(target_img_dir, base_name)
    lbl_subdir = os.path.join(target_label_dir, base_name)
    os.makedirs(img_subdir, exist_ok=True)
    os.makedirs(lbl_subdir, exist_ok=True)
    
    for i, (type_prefix, coords) in enumerate(all_patches):
        x1, y1, x2, y2 = coords
        patch_img = img[y1:y2, x1:x2]
        
        if patch_img.size == 0:
            continue
            
        patch_resized = cv2.resize(patch_img, tuple(config["target_size"]), interpolation=cv2.INTER_LINEAR)
        patch_filename = f"patch_{i}_{type_prefix}.png"
        out_img_path = os.path.join(img_subdir, patch_filename)
        cv2.imwrite(out_img_path, patch_resized)
        
        patch_labels = update_labels_for_patch(coords, gt_boxes, config)
        
        out_label_path = os.path.join(lbl_subdir, f"patch_{i}_{type_prefix}.txt")
        with open(out_label_path, 'w') as f_out:
            for lbl in patch_labels:
                cls_id, cx_norm, cy_norm, w_norm, h_norm = lbl
                f_out.write(f"{int(cls_id)} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                
    return True

def prepare_patch_dataset(source_data_dir, root_target_dir, config=None):
    if config is None:
        config = DEFAULT_CONFIG

    config_hash = get_config_hash(config)
    target_data_dir = os.path.join(root_target_dir, config_hash)
    
    print(f"--- Preparing Dataset Hash: {config_hash} ---")
    print(f"Target directory: {target_data_dir}")
    
    target_img_dir = os.path.join(target_data_dir, 'train', 'images')
    target_label_dir = os.path.join(target_data_dir, 'labels', 'annotations')
    
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)
    
    # Save config for reference
    with open(os.path.join(target_data_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    img_dir = os.path.join(source_data_dir, 'train', 'images')
    lbl_dir = os.path.join(source_data_dir, 'labels', 'annotations')
    
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"Source directories missing. Checked: {img_dir} and {lbl_dir}")
        return
        
    pairs = []
    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            stem = Path(img_file).stem
            label_file = os.path.join(lbl_dir, f"{stem}.txt")
            if os.path.exists(label_file):
                pairs.append((os.path.join(img_dir, img_file), label_file))
                
    if not pairs:
        print("No valid image/label pairs found.")
        return

    tasks = [(img_path, label_path, target_img_dir, target_label_dir, config) 
             for img_path, label_path in pairs]
    
    successful_images = 0
    with ProcessPoolExecutor() as executor:
        for success in tqdm(executor.map(process_single_image, tasks), total=len(tasks)):
            if success:
                successful_images += 1
                
    print(f"\nSuccessfully processed {successful_images}/{len(pairs)} base images.")
    
    num_imgs = sum(len(files) for _, _, files in os.walk(target_img_dir))
    num_lbls = sum(len(files) for _, _, files in os.walk(target_label_dir))
    print(f"Generated {num_imgs} images and {num_lbls} label files.")

    # Calculate and save anchors
    print("\nComputing anchors from generated patches...")
    wh_list = []
    for root, _, files in os.walk(target_label_dir):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 5:
                            w, h = float(parts[3]), float(parts[4])
                            wh_list.append((w, h))
    
    if wh_list:
        anchors = generate_anchors(wh_list, num_anchors=config["num_anchors"])
        anchors_list = anchors.tolist()
        print(f"Computed Anchors: {anchors_list}")
        
        anchors_path = os.path.join(target_data_dir, "anchors.json")
        with open(anchors_path, "w") as f:
            json.dump({"anchors": anchors_list}, f, indent=4)
        print(f"Saved anchors to {anchors_path}")
    else:
        print("Warning: No valid labels found to generate anchors.")

if __name__ == "__main__":
    seed_everything(42)
    SOURCE_DIR = os.path.join(road_sign_root, 'data')
    ROOT_TARGET_DIR = os.path.join(road_sign_root, 'data_patch_adaptive')
    
    prepare_patch_dataset(
        source_data_dir=SOURCE_DIR, 
        root_target_dir=ROOT_TARGET_DIR, 
        config=DEFAULT_CONFIG
    )
