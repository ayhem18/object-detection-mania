import os
import sys
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
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

from road_sign.utils.data_utils import YoloFormatDataset, yolov2_collate_fn
from road_sign.scripts.training.baseline.baseline import get_data_pairs

def profile_resized_data(data_pairs, target_size=(512, 512), batch_size=32, num_workers=4, max_batches=50):
    print(f"\n--- Profiling Pre-Resized Dataset: {target_size} (num_workers={num_workers}) ---")
    
    # Notice we removed the explicit Resize transform since images are already 512x512
    # The Dataset class will still enforce it, but it will be a no-op if sizes match
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    ds = YoloFormatDataset(data_pairs, target_size, transforms)
    dl = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=yolov2_collate_fn,
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
    for i, (imgs, _) in enumerate(tqdm(dl, total=min(len(dl), max_batches), desc="Loading Resized Data")):
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
    # Point directly to the new resized data directory
    data_dir = os.path.join(road_sign_root, 'data_512')
    pairs = get_data_pairs(data_dir)
    
    if not pairs:
        print(f"No data pairs found in {data_dir}. Exiting.")
        return

    print(f"Found {len(pairs)} pairs in the pre-resized directory.")
    
    # Test with standard parameters
    workers = 4 
    profile_resized_data(pairs, target_size=(512, 512), batch_size=64, num_workers=workers)

if __name__ == "__main__":
    main()
