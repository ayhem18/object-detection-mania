import os
import sys
import time
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

class MinimalImageDataset(Dataset):
    """
    A minimal dataset that only loads and resizes images to isolate the
    image processing bottleneck from target transformations.
    """
    def __init__(self, image_paths, img_shape):
        self.image_paths = image_paths
        self.transforms = v2.Compose([
            v2.Resize(img_shape),
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

def profile_minimal_dataset(image_paths, resolution, batch_size=32, num_workers=4, max_batches=20):
    print(f"\n--- Profiling MINIMAL Dataset Resolution: {resolution} (num_workers={num_workers}) ---")
    
    ds = MinimalImageDataset(image_paths, resolution)
    dl = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    start_time = time.time()
    total_images = 0
    
    # Warmup
    it = iter(dl)
    try:
        for _ in range(2):
            next(it)
    except StopIteration:
        pass
    
    warmup_time = time.time() - start_time
    print(f"Warmup (2 batches) took: {warmup_time:.2f}s")
    
    start_time = time.time()
    batch_count = 0
    for i, imgs in enumerate(tqdm(dl, total=min(len(dl), max_batches), desc=f"Min Res {resolution}")):
        total_images += imgs.shape[0]
        batch_count += 1
        if batch_count >= max_batches:
            break
            
    end_time = time.time()
    total_time = end_time - start_time
    
    fps = total_images / total_time
    print(f"Total time for {batch_count} batches ({total_images} images): {total_time:.2f}s")
    print(f"Throughput: {fps:.2f} images/sec")
    
    return fps

def main():
    data_dir = os.path.join(road_sign_root, 'data')
    pairs = get_data_pairs(data_dir)
    if not pairs:
        print("No data pairs found. Exiting.")
        return

    # Extract just the image paths
    image_paths = [img_path for img_path, _ in pairs]
    
    # Test just 512x512 first to compare directly with previous result
    resolutions = [(512, 512)]
    results = {}
    workers = 4 
    
    for res in resolutions:
        fps = profile_minimal_dataset(image_paths, res, num_workers=workers)
        results[res] = fps

if __name__ == "__main__":
    main()
