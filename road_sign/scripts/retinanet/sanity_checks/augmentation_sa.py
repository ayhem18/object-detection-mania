import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
from torchvision.transforms import v2

# Project Imports
from dl_lib.common_tools.path_utils import get_data_dir
from dl_lib.etalon_object_detection.modules.ds_utils import WeldingDetectionDataset, DL_LIB_ETALON_FIXED_SIZE

def resolve_paths(dataset_hash: str) -> Dict[str, Path]:
    """Resolves all required project paths based on the dataset hash."""
    base_data = get_data_dir()
    obj_det_root = base_data / "labeling"
    dataset_cache_dir = obj_det_root / "cache" / dataset_hash
    
    return {
        "root": base_data.parent,
        "images": obj_det_root / "data_as_images",
        "cache": dataset_cache_dir,
        "artifacts": obj_det_root / "artifacts" / "sanity_checks" / "augmentation_sa" / dataset_hash
    }

def denormalize(tensor):
    """Simple denormalization for visualization assuming [0, 1] range."""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def visualize_augmentation(aug_name: str, transform: v2.Compose, num_samples: int, dataset: WeldingDetectionDataset, output_base_dir: Path, indices: Optional[List[int]] = None):
    """Applies a specific augmentation and saves visualized samples."""
    print(f"Running visualization for augmentation: {aug_name}")
        
    dataset.transformations = dataset._ensure_resize(transform)
    
    aug_output_dir = output_base_dir / aug_name
    aug_output_dir.mkdir(parents=True, exist_ok=True)
    
    if indices is None:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    else:
        indices = indices[:num_samples]
    
    for i, idx in enumerate(indices):
        img_tensor, target = dataset[idx]
        
        img_np = denormalize(img_tensor)
        boxes = target['boxes'].cpu().numpy()
        labels = target['labels'].cpu().numpy()
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img_np)
        ax = plt.gca()
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor='lime', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"Cls: {label}", color='lime', backgroundcolor='black', fontsize=10)
            
        plt.title(f"{aug_name} - Sample {i}")
        plt.axis('off')
        
        save_path = aug_output_dir / f"sample_{idx:04d}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def run_augmentation_sanity_check(config: Dict[str, Any]):
    dataset_hash = config["dataset_hash"]
    paths = resolve_paths(dataset_hash)
    
    images_dir = paths["images"]
    cache_dir = paths["cache"]
    output_base_dir = paths["artifacts"]
    
    num_samples = config.get("num_samples_per_aug", 5)
    
    # 2. Define augmentations to test (p=1.0 to force them)
    augmentations = {
        "HorizontalFlip": v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=1.0),
            v2.ToDtype(torch.float32, scale=True),
        ]),
        "VerticalFlip": v2.Compose([
            v2.ToImage(),
            v2.RandomVerticalFlip(p=1.0),
            v2.ToDtype(torch.float32, scale=True),
        ]),
        "Identity": v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    }
    
    # 3. Setup Dataset
    # We use None for transforms initially, then override in visualize_augmentation
    dataset = WeldingDetectionDataset(
        images_dir=str(images_dir),
        cache_dir=str(cache_dir),
        target_size=DL_LIB_ETALON_FIXED_SIZE,
        transformations=None
    )
    
    indices = [1, 2, 3, 4, 5]

    # 4. Run Loop
    for name, transform in augmentations.items():
        visualize_augmentation(name, transform, num_samples=num_samples, dataset=dataset, output_base_dir=output_base_dir, indices=indices)
        
    print(f"\nAugmentation sanity check complete. Visualizations saved to: {output_base_dir}")

def main():
    config = {
        "dataset_hash": "latest",
        "num_samples_per_aug": 5,
        "seed": 42
    }
    
    if config["dataset_hash"] == "latest":
        script_dir = Path(__file__).resolve().parent
        root_dir = script_dir
        while not (root_dir / 'data').exists(): root_dir = root_dir.parent
        cache_dir = root_dir / "data" / "labeling" / "cache"
        if cache_dir.exists():
            subdirs = [d for d in cache_dir.iterdir() if d.is_dir() and len(d.name) == 32]
            if subdirs:
                latest_hash_dir = max(subdirs, key=os.path.getmtime)
                config["dataset_hash"] = latest_hash_dir.name
                print(f"Using latest dataset hash: {config['dataset_hash']}")
            
    np.random.seed(config.get("seed", 42))
    run_augmentation_sanity_check(config)

if __name__ == "__main__":
    main()
