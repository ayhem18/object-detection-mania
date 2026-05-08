import os
import sys
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Path setup logic as requested
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up until 'road_sign' is in the current directory's list
while 'road_sign' not in os.listdir(current_dir):
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:
        break
    current_dir = parent_dir

road_sign_root = os.path.join(current_dir, 'road_sign')

# Add necessary paths to sys.path
sys.path.append(road_sign_root)
sys.path.append(os.path.join(road_sign_root, 'utils'))
sys.path.append(os.path.join(road_sign_root, 'scripts'))

import kagglehub

def download_dataset():
    # Load environment variables (KAGGLE_USERNAME, KAGGLE_KEY) from .env
    load_dotenv()
    
    print("Downloading dataset...")
    # Download latest version from Kaggle
    cache_path = kagglehub.competition_download('traffic-sign-object-detection-challenge')
    print(f"Path to competition files in cache: {cache_path}")
    
    # Destination directory
    data_dir = os.path.join(road_sign_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Copy files to our local data directory
    # Assuming standard structure, we might want to copy everything or specific folders
    for item in os.listdir(cache_path):
        src = os.path.join(cache_path, item)
        dst = os.path.join(data_dir, item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
            
    print(f"Dataset saved to: {data_dir}")
    return data_dir

def coco_to_yolo(data_dir):
    """
    Converts COCO format annotations to YOLO format.
    Expects annotations in data_dir/annotations or similar.
    """
    # This logic depends on the exact structure of the downloaded dataset.
    # Usually COCO has a JSON file. Let's find it.
    coco_files = list(Path(data_dir).rglob("*.json"))
    if not coco_files:
        print("No COCO JSON files found.")
        return

    for coco_file in coco_files:
        print(f"Converting {coco_file.name}...")
        with open(coco_file, 'r') as f:
            data = json.load(f)

        # Create output directory for labels
        # If the json is 'train.json', we might want labels/train/
        split_name = coco_file.stem
        labels_dir = os.path.join(data_dir, 'labels', split_name)
        os.makedirs(labels_dir, exist_ok=True)

        # Map image id to filename and dimensions
        images = {img['id']: img for img in data['images']}
        
        # Group annotations by image
        img_annots = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_annots:
                img_annots[img_id] = []
            img_annots[img_id].append(ann)

        for img_id, anns in img_annots.items():
            img_info = images[img_id]
            file_name = img_info['file_name']
            w = img_info['width']
            h = img_info['height']
            
            # YOLO file name (txt instead of jpg/png)
            label_file = os.path.join(labels_dir, Path(file_name).stem + ".txt")
            
            with open(label_file, 'w') as f_out:
                for ann in anns:
                    cat_id = ann['category_id']
                    # YOLO expects class_id (usually 0-indexed)
                    # COCO categories can be 1-indexed, might need mapping
                    class_id = cat_id - 1 
                    
                    # COCO bbox: [x_min, y_min, width, height]
                    bbox = ann['bbox']
                    x_min, y_min, bw, bh = bbox
                    
                    # YOLO: [x_center, y_center, width, height] (normalized)
                    x_center = (x_min + bw / 2) / w
                    y_center = (y_min + bh / 2) / h
                    nw = bw / w
                    nh = bh / h
                    
                    f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}\n")

    print("Conversion complete.")

if __name__ == "__main__":
    try:
        data_path = download_dataset()
        coco_to_yolo(data_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
