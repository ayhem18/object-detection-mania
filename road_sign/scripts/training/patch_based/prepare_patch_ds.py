import os
import sys
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Path setup logic
current_dir = os.path.dirname(os.path.abspath(__file__))
while 'road_sign' not in os.listdir(current_dir):
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:
        break
    current_dir = parent_dir

road_sign_root = os.path.join(current_dir, 'road_sign')
sys.path.insert(0, os.path.join(road_sign_root, 'scripts', 'training', 'patch_based'))

from mypt.code_utils.pytorch_utils import seed_everything

# Import the core logic we just built
from prepare_patches import (
    generate_positive_patches, 
    generate_negative_patches, 
    update_labels_for_patch
)

def process_single_image(args):
    """
    Worker function to process a single image.
    Extracts patches, resizes them, and saves images/labels in subdirectories.
    """
    img_path, label_path, target_img_dir, target_label_dir, target_size, K2 = args
    
    img = cv2.imread(img_path)
    if img is None:
        return False
        
    h, w = img.shape[:2]
    
    # 1. Load original labels and convert to absolute pixels
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

    # 2. Generate coordinates
    pos_coords = generate_positive_patches((h, w), gt_boxes)
    neg_coords = generate_negative_patches(img, gt_boxes, K2)
    
    all_patches = [('pos', c) for c in pos_coords] + [('neg', c) for c in neg_coords]
    
    base_name = Path(img_path).stem
    
    # Create subdirectories for this image
    img_subdir = os.path.join(target_img_dir, base_name)
    lbl_subdir = os.path.join(target_label_dir, base_name)
    os.makedirs(img_subdir, exist_ok=True)
    os.makedirs(lbl_subdir, exist_ok=True)
    
    # 3. Crop, Resize, and Save
    for i, (type_prefix, coords) in enumerate(all_patches):
        x1, y1, x2, y2 = coords
        
        # Crop the patch
        patch_img = img[y1:y2, x1:x2]
        
        # Ensure the patch is not empty before resizing
        if patch_img.size == 0:
            continue
            
        patch_resized = cv2.resize(patch_img, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Create filename: patch_{i}_{type}.png
        patch_filename = f"patch_{i}_{type_prefix}.png"
        
        out_img_path = os.path.join(img_subdir, patch_filename)
        cv2.imwrite(out_img_path, patch_resized)
        
        # Calculate new labels (normalized to the patch)
        patch_labels = update_labels_for_patch(coords, gt_boxes)
        
        out_label_path = os.path.join(lbl_subdir, f"patch_{i}_{type_prefix}.txt")
        with open(out_label_path, 'w') as f_out:
            for lbl in patch_labels:
                cls_id, cx_norm, cy_norm, w_norm, h_norm = lbl
                f_out.write(f"{int(cls_id)} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                
    return True

def prepare_patch_dataset(source_data_dir, target_data_dir, target_size=(512, 512), K2=2):
    print(f"Preparing adaptive patched dataset to {target_data_dir}")
    print(f"Final saved patch size: {target_size}")
    
    # Setup target directories
    target_img_dir = os.path.join(target_data_dir, 'train', 'images')
    target_label_dir = os.path.join(target_data_dir, 'labels', 'annotations')
    
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)
    
    # Gather pairs
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

    # Prepare multiprocessing tasks
    tasks = [(img_path, label_path, target_img_dir, target_label_dir, target_size, K2) 
             for img_path, label_path in pairs]
    
    successful_images = 0
    with ProcessPoolExecutor() as executor:
        for success in tqdm(executor.map(process_single_image, tasks), total=len(tasks)):
            if success:
                successful_images += 1
                
    print(f"\nSuccessfully processed {successful_images}/{len(pairs)} base images.")
    
    # Count generated files recursively
    num_imgs = sum(len(files) for _, _, files in os.walk(target_img_dir))
    num_lbls = sum(len(files) for _, _, files in os.walk(target_label_dir))
    print(f"Generated {num_imgs} images and {num_lbls} label files.")

if __name__ == "__main__":
    seed_everything(42)
    SOURCE_DIR = os.path.join(road_sign_root, 'data')
    TARGET_DIR = os.path.join(road_sign_root, 'data_patch_adaptive')
    
    prepare_patch_dataset(
        source_data_dir=SOURCE_DIR, 
        target_data_dir=TARGET_DIR, 
        target_size=(512, 512), 
        K2=2
    )

