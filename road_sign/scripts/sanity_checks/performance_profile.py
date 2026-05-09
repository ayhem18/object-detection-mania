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

def profile_resolution(data_pairs, resolution, batch_size=32, num_workers=4, max_batches=20):
    print(f"\n--- Profiling Resolution: {resolution} (num_workers={num_workers}) ---")
    
    transforms = v2.Compose([
        v2.Resize(resolution),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    ds = YoloFormatDataset(data_pairs, resolution, transforms)
    dl = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=yolov2_collate_fn,
        pin_memory=False
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
    for i, (imgs, _) in enumerate(tqdm(dl, total=min(len(dl), max_batches), desc=f"Res {resolution}")):
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

    resolutions = [(512, 512), (1800, 1800), (2400, 2400), (3600, 3600)]
    results = {}

    # Test with typical worker count
    workers = 4 # Adjust based on your CPU cores
    
    for res in resolutions:
        fps = profile_resolution(pairs, res, num_workers=workers)
        results[res] = fps

    print("\n" + "="*30)
    print("      PERFORMANCE SUMMARY")
    print("="*30)
    print(f"{'Resolution':<15} | {'Images/sec':<10}")
    print("-" * 30)
    for res, fps in results.items():
        print(f"{str(res):<15} | {fps:<10.2f}")
    print("="*30)

if __name__ == "__main__":
    main()
