import os
import sys
import json
from pathlib import Path
from collections import Counter

# Path setup logic
current_dir = os.path.dirname(os.path.abspath(__file__))
while 'road_sign' not in os.listdir(current_dir):
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:
        break
    current_dir = parent_dir

road_sign_root = os.path.join(current_dir, 'road_sign')
sys.path.append(road_sign_root)
sys.path.append(os.path.join(road_sign_root, 'utils'))
sys.path.append(os.path.join(road_sign_root, 'scripts'))

def analyze_dataset(data_dir):
    coco_files = list(Path(data_dir).rglob("*.json"))
    if not coco_files:
        print("No COCO JSON files found for analysis.")
        return

    for coco_file in coco_files:
        print(f"\n--- Analysis for {coco_file.name} ---")
        with open(coco_file, 'r') as f:
            data = json.load(f)

        # 1. Number of images
        num_images = len(data['images'])
        print(f"Total images: {num_images}")

        # 2. Distribution of image sizes
        sizes = Counter([(img['width'], img['height']) for img in data['images']])
        print("\nImage size distribution (Width, Height):")
        for size, count in sizes.most_common():
            print(f"  {size}: {count} images")

        # 3. Class mapping
        categories = {cat['id']: cat['name'] for cat in data['categories']}
        
        # 4. Number of bounding boxes per class
        bbox_counts = Counter([ann['category_id'] for ann in data['annotations']])
        print("\nBounding boxes per class:")
        total_bboxes = 0
        for cat_id, count in sorted(bbox_counts.items()):
            class_name = categories.get(cat_id, f"Unknown({cat_id})")
            print(f"  {class_name}: {count}")
            total_bboxes += count
        
        print(f"\nTotal bounding boxes: {total_bboxes}")
        print(f"Average bboxes per image: {total_bboxes / num_images:.2f}")

if __name__ == "__main__":
    data_path = os.path.join(road_sign_root, 'data')
    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist. Please run prepare_data.py first.")
    else:
        analyze_dataset(data_path)
