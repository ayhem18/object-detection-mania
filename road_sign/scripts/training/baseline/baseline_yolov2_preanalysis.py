import os
import sys
from mypt.code_utils.pytorch_utils import seed_everything
import torch
import cv2
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from torchvision.transforms import v2

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
while True:
    if 'road_sign' in os.listdir(current_dir) and os.path.isdir(os.path.join(current_dir, 'road_sign')):
        break
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:
        raise RuntimeError("Could not find 'road_sign' directory in the parent path.")
    current_dir = parent_dir

workspace_root = current_dir
road_sign_root = os.path.join(workspace_root, 'road_sign')

sys.path.insert(0, workspace_root)
# Note: mypt is installed via uv workspaces, so it is naturally resolvable without sys.path hacks.

from road_sign.utils.data_utils import YoloFormatDataset

def get_data_pairs(data_dir):
    img_dir = os.path.join(data_dir, 'train', 'images')
    label_dir = os.path.join(data_dir, 'labels', 'annotations')
    
    pairs = []
    if not os.path.exists(img_dir):
        
        print(f"Warning: Image directory not found at {img_dir}")
        return pairs

    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            stem = Path(img_file).stem
            label_file = os.path.join(label_dir, f"{stem}.txt")
            if not os.path.exists(label_file):
                continue
            pairs.append((os.path.join(img_dir, img_file), label_file))
    return pairs

def analyze_visibility(data_pairs, img_size=(512, 512), stride=32):
    """
    Analyzes how many objects will effectively be 'visible' to the feature map.
    Metric: object_area / (stride * stride)
    """
    print(f"\n--- Pre-Analysis: Object Visibility (Input Size: {img_size}, Stride: {stride}) ---")
    
    target_w, target_h = img_size
    receptive_field_area = stride * stride
    
    visibility_bins = {
        "Sub-pixel (< 1.0)": 0,
        "Very Small (1.0 - 4.0)": 0,
        "Small (4.0 - 16.0)": 0,
        "Medium (16.0 - 64.0)": 0,
        "Large (> 64.0)": 0
    }
    
    total_boxes = 0
    
    for _, label_path in data_pairs:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    logging.warning(f"Invalid label file: {label_path}")
                    continue

                total_boxes += 1
                # YOLO format: cls, cx, cy, w_norm, h_norm
                w_norm, h_norm = float(parts[3]), float(parts[4])
                
                # Convert to absolute pixels based on target size
                abs_w = w_norm * target_w
                abs_h = h_norm * target_h
                abs_area = abs_w * abs_h
                
                # Calculate visibility score
                visibility_score = abs_area / receptive_field_area
                
                if visibility_score < 1.0:
                    visibility_bins["Sub-pixel (< 1.0)"] += 1
                elif visibility_score < 4.0:
                    visibility_bins["Very Small (1.0 - 4.0)"] += 1
                elif visibility_score < 16.0:
                    visibility_bins["Small (4.0 - 16.0)"] += 1
                elif visibility_score < 64.0:
                    visibility_bins["Medium (16.0 - 64.0)"] += 1
                else:
                    visibility_bins["Large (> 64.0)"] += 1
                        
    print(f"Total Objects Analyzed: {total_boxes}")
    print("Visibility Score = Object Pixel Area / Receptive Field Area (32x32 = 1024 px)")
    print("Interpretation: A score of 1.0 means the object is roughly the size of a single 1x1 cell on the final feature map.")
    for k, v in visibility_bins.items():
        percentage = (v / total_boxes) * 100 if total_boxes > 0 else 0
        print(f"  {k}: {v} objects ({percentage:.1f}%)")
    print("----------------------------------------------------------------------------------\n")

def visualize_dataset(data_pairs, img_size=(512, 512), max_per_class=5):
    out_dir = os.path.join(road_sign_root, 'artifacts', 'baseline', 'dataset_sa')
    os.makedirs(out_dir, exist_ok=True)
    
    # Simple transforms for visualization (no ImageNet norm)
    transforms = v2.Compose([
        v2.Resize(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    ds = YoloFormatDataset(data_pairs, img_size, transforms)
    saved_counts = defaultdict(int)
    
    print(f"--- Visualizing and Saving Sanity Check Images to {out_dir} ---")
    
    for i in range(len(ds)):
        img, targets = ds[i]
        
        if targets.shape[0] == 0:
            continue
            
        cls_ids = targets[:, 0].long().tolist()
        
        # Check if we should save this image
        should_save = False
        for c in set(cls_ids):
            if saved_counts[c] < max_per_class:
                should_save = True
                break
                
        if not should_save:
            # Assume 8 classes (0-7), stop if we have enough of all
            if len(saved_counts) == 8 and all(v >= max_per_class for v in saved_counts.values()):
                break
            continue
            
        for c in set(cls_ids):
            saved_counts[c] += 1
            
        # Convert tensor to BGR numpy array
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        h, w = img_size
        for t in targets:
            cls_id = int(t[0].item())
            cx, cy, bw, bh = t[1:]
            
            abs_cx, abs_cy = cx * w, cy * h
            abs_bw, abs_bh = bw * w, bh * h
            
            x1 = int(abs_cx - abs_bw / 2)
            y1 = int(abs_cy - abs_bh / 2)
            x2 = int(abs_cx + abs_bw / 2)
            y2 = int(abs_cy + abs_bh / 2)
            
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_np, f"cls:{cls_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # Extract original filename
        orig_img_path = ds.data_pairs[i][0]
        orig_name = Path(orig_img_path).stem
        
        out_path = os.path.join(out_dir, f"{orig_name}_classes_{'_'.join(map(str, set(cls_ids)))}.jpg")
        cv2.imwrite(out_path, img_np)
        
    print(f"Saved sanity check images. Class counts: {dict(saved_counts)}")

if __name__ == '__main__':
    seed_everything(42)
    data_dir = os.path.join(road_sign_root, 'data')
    pairs = get_data_pairs(data_dir)
    if not pairs:
        print("No data pairs found. Exiting.")
        sys.exit(1)
        
    analyze_visibility(pairs, img_size=(512, 512), stride=32)
    visualize_dataset(pairs, img_size=(512, 512), max_per_class=3)
