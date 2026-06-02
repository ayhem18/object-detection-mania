import os
import sys
import time
import cv2
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
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

from road_sign.scripts.training.baseline.baseline import get_data_pairs

class PILMinimalDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = self.transforms(img)
        return img

class CV2MinimalDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)
        return img

def profile_dataset(dataset, name, batch_size=32, num_workers=0):
    print(f"\n--- Profiling {name} (num_workers={num_workers}) ---")
    
    dl = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    start_time = time.time()
    total_images = 0
    
    for i, imgs in enumerate(tqdm(dl, desc=f"{name} DataLoader")):
        total_images += imgs.shape[0]
            
    end_time = time.time()
    total_time = end_time - start_time
    
    fps = total_images / total_time
    print(f"Total time for {total_images} images: {total_time:.2f}s")
    print(f"Throughput: {fps:.2f} images/sec")
    
    return fps

def main():
    data_dir = os.path.join(road_sign_root, 'data_512')
    pairs = get_data_pairs(data_dir)
    
    if not pairs:
        print(f"No data pairs found in {data_dir}. Exiting.")
        return

    # Extract just the image paths
    image_paths = [img_path for img_path, _ in pairs]
    print(f"Testing DataLoader speed on {len(image_paths)} images (512x512)...")
    
    # 0 workers is the true test of pure decoding/tensor-conversion speed
    # without multiprocessing spawn overhead masking the results.
    workers = 0 
    batch_size = 64
    
    ds_pil = PILMinimalDataset(image_paths)
    pil_fps = profile_dataset(ds_pil, "PIL Dataset", batch_size, workers)
    
    ds_cv2 = CV2MinimalDataset(image_paths)
    cv2_fps = profile_dataset(ds_cv2, "CV2 Dataset", batch_size, workers)
    
    print("\n" + "="*30)
    print("      PERFORMANCE SUMMARY")
    print("="*30)
    print(f"PIL: {pil_fps:.2f} img/s")
    print(f"CV2: {cv2_fps:.2f} img/s")
    print("="*30)

if __name__ == "__main__":
    main()
