import os
import sys
import shutil
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

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

from road_sign.scripts.training.baseline.baseline import get_data_pairs

def process_single_pair(args):
    img_path, label_path, target_img_dir, target_label_dir, target_size = args
    
    # 1. Read and resize image
    img = cv2.imread(img_path)
    if img is None:
        return False
        
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 2. Save resized image
    filename = Path(img_path).name
    out_img_path = os.path.join(target_img_dir, filename)
    cv2.imwrite(out_img_path, img_resized)
    
    # 3. Copy label file (no modifications needed as YOLO coordinates are relative)
    label_filename = Path(label_path).name
    out_label_path = os.path.join(target_label_dir, label_filename)
    shutil.copy2(label_path, out_label_path)
    
    return True

def resize_dataset(source_data_dir, target_data_dir, target_size=(512, 512)):
    print(f"Resizing dataset from {source_data_dir} to {target_data_dir}")
    print(f"Target size: {target_size}")
    
    pairs = get_data_pairs(source_data_dir)
    if not pairs:
        print("No data pairs found!")
        return

    target_img_dir = os.path.join(target_data_dir, 'train', 'images')
    target_label_dir = os.path.join(target_data_dir, 'labels', 'annotations')
    
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)
    
    # Prepare arguments for multiprocessing
    tasks = [(img_path, label_path, target_img_dir, target_label_dir, target_size) 
             for img_path, label_path in pairs]
    
    successful_conversions = 0
    # Use ProcessPoolExecutor for CPU-bound resizing
    with ProcessPoolExecutor() as executor:
        for success in tqdm(executor.map(process_single_pair, tasks), total=len(tasks)):
            if success:
                successful_conversions += 1
                
    print(f"Successfully resized {successful_conversions}/{len(pairs)} images.")

if __name__ == "__main__":
    SOURCE_DIR = os.path.join(road_sign_root, 'data')
    TARGET_DIR = os.path.join(road_sign_root, 'data_512')
    
    resize_dataset(SOURCE_DIR, TARGET_DIR, target_size=(512, 512))
