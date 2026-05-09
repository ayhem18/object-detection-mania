import os
import sys
import json
import shutil
import cv2
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

def visualize_original_data(data_dir, max_images=10):
    """
    Visualizes the raw annotations from the JSON file directly onto the original images.
    Saves the results to road_sign/artifacts/org_data_sa.
    """
    coco_files = list(Path(data_dir).rglob("*.json"))
    if not coco_files:
        print("No JSON files found.")
        return

    out_dir = os.path.join(road_sign_root, 'artifacts', 'org_data_sa')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Visualizing original data to {out_dir}")

    for coco_file in coco_files:
        with open(coco_file, 'r') as f:
            data = json.load(f)

        images = {img['id']: img for img in data['images']}
        categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        
        img_annots = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_annots:
                img_annots[img_id] = []
            img_annots[img_id].append(ann)

        count = 0
        for img_id, anns in img_annots.items():
            if count >= max_images:
                break
                
            img_info = images[img_id]
            file_name = img_info['file_name']
            
            # Assuming images are in data_dir/train/images or data_dir/images
            # We'll try a few common paths
            img_path = os.path.join(data_dir, 'train', 'images', file_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(data_dir, 'train', file_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(data_dir, 'images', file_name)
                
            if not os.path.exists(img_path):
                print(f"Could not find image: {file_name}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue

            for ann in anns:
                # The crucial check: how is the bbox actually formatted?
                # COCO standard is [x_min, y_min, width, height]
                bbox = ann['bbox']
                cat_id = ann['category_id']
                cat_name = categories.get(cat_id, str(cat_id))

                if len(bbox) == 4:
                    x, y, w, h = bbox
                    # Draw assuming standard COCO [x, y, w, h]
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.putText(img, f"{cat_name} (xywh)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    
                    # Draw assuming it might be [x_min, y_min, x_max, y_max] just in case
                    x1_alt, y1_alt = int(x), int(y)
                    x2_alt, y2_alt = int(w), int(h)
                    cv2.rectangle(img, (x1_alt, y1_alt), (x2_alt, y2_alt), (255, 0, 0), 2)
                    cv2.putText(img, "blue=[x1,y1,x2,y2]", (x1_alt, y1_alt + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            out_path = os.path.join(out_dir, f"org_{Path(file_name).name}")
            cv2.imwrite(out_path, img)
            count += 1
            
    print("Finished visualizing original data.")

def coco_to_yolo(data_dir):
    """
    Converts COCO format annotations to YOLO format.
    Expects annotations in data_dir/annotations or similar.
    """
    coco_files = list(Path(data_dir).rglob("*.json"))
    if not coco_files:
        print("No COCO JSON files found.")
        return

    for coco_file in coco_files:
        print(f"Converting {coco_file.name}...")
        with open(coco_file, 'r') as f:
            data = json.load(f)

        # Create output directory for labels
        split_name = coco_file.stem
        labels_dir = os.path.join(data_dir, 'labels', split_name)
        os.makedirs(labels_dir, exist_ok=True)

        # Build a robust class mapping
        categories = data.get('categories', [])
        # The COCO data is already 0-indexed: 0 to 7
        # We will map the original ID directly to our class_id, and save the mapping
        id_to_name = {cat['id']: cat['name'] for cat in categories}
        
        # Save class mapping to the data directory for future reference
        class_mapping_path = os.path.join(data_dir, 'class_mapping.json')
        with open(class_mapping_path, 'w') as f:
            json.dump(id_to_name, f, indent=4)
        print(f"Saved class mapping to {class_mapping_path}")

        images = {img['id']: img for img in data['images']}
        
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
            
            label_file = os.path.join(labels_dir, Path(file_name).stem + ".txt")
            
            with open(label_file, 'w') as f_out:
                for ann in anns:
                    # COCO dataset is already 0-indexed, no need to subtract 1
                    class_id = ann['category_id']
                    
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
        # data_path = download_dataset()
        # Assume data is already downloaded for this debugging step
        data_path = os.path.join(road_sign_root, 'data')
        
        visualize_original_data(data_path, max_images=5)
        # Re-run conversion to fix the label bug
        coco_to_yolo(data_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
