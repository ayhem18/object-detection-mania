import os
import json
from mypt.code_utils.pytorch_utils import seed_everything
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Dict, Any
from torchvision.transforms import v2
from torch.utils.data import DataLoader

# Internal Imports
from dl_lib.common_tools.path_utils import get_data_dir
from dl_lib.etalon_object_detection.modules.ds_utils import WeldingDetectionDataset, collate_fn, DL_LIB_ETALON_FIXED_SIZE
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_anchors import (
    anchor_spec_from_manifest_path,
    resolve_split_anchor_config,
)
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_detector import (
    build_dl_lab_etalon_retinanet_from_segmentation_weights,
)

def resolve_paths(dataset_hash: str) -> Dict[str, Path]:
    """Resolves all required project paths based on the dataset hash."""
    base_data = get_data_dir()
    obj_det_root = base_data / "labeling"
    dataset_cache_dir = obj_det_root / "cache" / dataset_hash
    
    return {
        "root": base_data.parent,
        "images": obj_det_root / "data_as_images",
        "cache": dataset_cache_dir,
        "weights": base_data / "weights" / "latest_segmentation_model.pt",
        "output_dir": base_data / "debug_data_sanity" / dataset_hash
    }

def denormalize(tensor, mean, std):
    """Reverses the normalization for visualization."""
    mean = torch.as_tensor(mean).view(-1, 1, 1)
    std = torch.as_tensor(std).view(-1, 1, 1)
    res = tensor * std + mean
    return torch.clamp(res * 255, 0, 255).byte().permute(1, 2, 0).numpy()

def visualize_raw_labels(cache_dir: Path, images_dir: Path, output_dir: Path, num_samples: int = 20):
    """
    Reads the master label file and visualizes images with labels on top without any transformation.
    """
    print(f"\n--- Checking Raw Labels (No Transformations) ---")
    raw_viz_dir = output_dir / "raw_labels"
    raw_viz_dir.mkdir(parents=True, exist_ok=True)
    
    master_json_path = cache_dir / "master_labels.json"
    if not master_json_path.exists():
        print(f"  [ERROR] Master labels not found at {master_json_path}")
        return
        
    with open(master_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    samples = data.get("samples", [])
    if not samples:
        print("  [ERROR] No samples found in master labels.")
        return
        
    # Select random samples
    indices = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)
    
    for i, idx in enumerate(indices):
        sample = samples[idx]
        img_path = images_dir / sample['img_path']
        lbl_path = cache_dir / sample['lbl_path']
        
        if not img_path.exists():
            print(f"  [SKIP] Image not found: {img_path}")
            continue
        if not lbl_path.exists():
            print(f"  [SKIP] Label not found: {lbl_path}")
            continue
            
        # Load Image
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        img_y_dim, img_x_dim = img.size
        
        # Load Labels
        with open(lbl_path, 'r', encoding='utf-8') as f:
            lbl_data = json.load(f)
            
        boxes = lbl_data.get("boxes", [])
        classes = lbl_data.get("classes", [])
        
        plt.figure(figsize=(15, 6))
        plt.imshow(img_np)
        ax = plt.gca()
        
        for box, label in zip(boxes, classes):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"Cls: {label}", color='lime', backgroundcolor='black', fontsize=8)
            
        plt.title(f"Raw Label | Index: {idx} | Size: {img_x_dim}x{img_y_dim} | {sample['img_path']}")
        plt.axis('off')
        
        save_path = raw_viz_dir / f"raw_sample_{idx:04d}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  [SAVED] {save_path}")

def visualize_dataset_output(dataset: WeldingDetectionDataset, output_dir: Path, num_samples: int = 20):
    """
    Tests and visualizes the raw output of the WeldingDetectionDataset.
    Ensures all images meet the target size.
    """
    print(f"\n--- Checking Dataset Output (Target Size: {DL_LIB_ETALON_FIXED_SIZE}) ---")
    ds_viz_dir = output_dir / "dataset_output"
    ds_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        image, target = dataset[idx]
        
        # 1. Size Verification
        y_dim, x_dim = image.shape[-2:]
        if (y_dim, x_dim) != DL_LIB_ETALON_FIXED_SIZE:
            raise ValueError(f"Sample {idx} has size ({y_dim}, {x_dim}), expected {DL_LIB_ETALON_FIXED_SIZE}")

        # 2. Visualization
        img_np = image.permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(15, 6))
        plt.imshow(img_np)
        ax = plt.gca()
        
        boxes = target['boxes'].cpu().numpy()
        labels = target['labels'].cpu().numpy()
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"Cls: {label}", color='lime', backgroundcolor='black', fontsize=8)
            
        plt.title(f"Dataset Output | Index: {idx} | Size: {x_dim}x{y_dim}")
        plt.axis('off')
        
        save_path = ds_viz_dir / f"ds_sample_{idx:04d}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  [SAVED] {save_path}")

def visualize_retinanet_transform_output(model: torch.nn.Module, dataset: WeldingDetectionDataset, output_dir: Path, is_training: bool, num_samples: int = 20):
    """
    Tests and visualizes the output of the internal RetinaNet.transform module.
    This includes normalization (mean/std) and potential batch padding.
    """

    if is_training:
        model.train()
    else:
        model.eval()

    print(f"\n--- Checking RetinaNet Transform Output ---")
    transform_viz_dir = output_dir / f"retinanet_transform_{'train' if is_training else 'eval'}_mode"
    transform_viz_dir.mkdir(parents=True, exist_ok=True)
    
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=False, collate_fn=collate_fn)
    
    images, targets = next(iter(loader))

    for img in images:
        if img.shape != (3,) + DL_LIB_ETALON_FIXED_SIZE:
            raise ValueError(f"Sample {i} has shape ({img.shape}), expected (3, {DL_LIB_ETALON_FIXED_SIZE[0]}, {DL_LIB_ETALON_FIXED_SIZE[1]})")

    # we know for sure that the images are of the correct shape
    images_list = [img.to(device) for img in images]
    targets_list = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    # Run internal transform (GeneralizedRCNNTransform)
    transformed_images, transformed_targets = model.transform(images_list, targets_list)
    
    image_mean = model.transform.image_mean
    image_std = model.transform.image_std
    
    # transformed_images.tensors is the [B, C, H, W] padded tensor
    batch_tensors = transformed_images.tensors.cpu()
    
    for i in range(len(images)):
        batch_shape = batch_tensors[i].shape
        if batch_shape != (3,) + DL_LIB_ETALON_FIXED_SIZE:
            raise ValueError(f"Sample {i} has shape ({batch_shape}), expected (3, {DL_LIB_ETALON_FIXED_SIZE[0]}, {DL_LIB_ETALON_FIXED_SIZE[1]})")
        
        # Denormalize for viewing
        img_np = denormalize(batch_tensors[i], image_mean, image_std)
        
        t_target = transformed_targets[i]
        boxes = t_target['boxes'].cpu().numpy()
        labels = t_target['labels'].cpu().numpy()
        t_size = transformed_images.image_sizes[i]


        plt.figure(figsize=(15, 6))
        plt.imshow(img_np)
        ax = plt.gca()
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='cyan', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"Cls: {label}", color='cyan', backgroundcolor='black', fontsize=8)
            
        plt.title(f"RetinaNet Transform Output | Sample {i} | T-Size: {t_size}")
        plt.axis('off')
        
        save_path = transform_viz_dir / f"transform_sample_{i:02d}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  [SAVED] {save_path}")


def run_data_sanity_check(config: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_hash = config["dataset_hash"]
    paths = resolve_paths(dataset_hash)
    
    images_dir = paths["images"]
    cache_dir = paths["cache"]
    weights_path = paths["weights"]
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup Dataset
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    dataset = WeldingDetectionDataset(
        images_dir=str(images_dir),
        cache_dir=str(cache_dir),
        target_size=DL_LIB_ETALON_FIXED_SIZE,
        transformations=None
    )
    
    # 2. Setup Model (to access its transform module)
    anchor_config_path, _ = resolve_split_anchor_config(dataset_hash)
    anchor_spec = anchor_spec_from_manifest_path(anchor_config_path)
    model = build_dl_lab_etalon_retinanet_from_segmentation_weights(
        anchor_spec=anchor_spec,
        segmentation_weights_path=str(weights_path),
        img_size=DL_LIB_ETALON_FIXED_SIZE,
        device=device,
    )
    # set the model to training model 
    # model.train()
    # model.eval()

    # 3. Run Sanity Checks
    num_samples = config.get("num_samples", 20)
    
    visualize_raw_labels(cache_dir, images_dir, output_dir, num_samples)
    visualize_dataset_output(dataset, output_dir, num_samples)
    visualize_retinanet_transform_output(model, dataset, output_dir, is_training=True, num_samples=num_samples)
    visualize_retinanet_transform_output(model, dataset, output_dir, is_training=False, num_samples=num_samples)

def main():
    seed_everything(42)
    config = {
        "dataset_hash": "latest",
        "num_samples": 20
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
            
    run_data_sanity_check(config)


def test_something():
    from dl_lib.common_tools.path_utils import get_data_dir
    
    # Use real model initialization which now includes the FixedSizeTransformWrapper
    base_data = get_data_dir()
    cache_dir = base_data / "labeling" / "cache"
    
    # Find latest hash dir for manifest
    subdirs = [d for d in cache_dir.iterdir() if d.is_dir() and len(d.name) == 32]
    latest_hash_dir = max(subdirs, key=os.path.getmtime)
    anchor_config_path, _ = resolve_split_anchor_config(latest_hash_dir.name)
    weights_path = base_data / "weights" / "latest_segmentation_model.pt"

    print(f"Testing model transform with FIX for target size: {DL_LIB_ETALON_FIXED_SIZE}")

    anchor_spec = anchor_spec_from_manifest_path(anchor_config_path)
    model = build_dl_lab_etalon_retinanet_from_segmentation_weights(
        anchor_spec=anchor_spec,
        segmentation_weights_path=str(weights_path),
        img_size=DL_LIB_ETALON_FIXED_SIZE,
        device=torch.device("cpu"),
    )

    shape = (3, ) + DL_LIB_ETALON_FIXED_SIZE
    images = [torch.randn(shape) for _ in range(10)]
    targets = []

    for i in range(10):
        targets.append({
            'boxes': torch.randn(2, 4).clamp(min=0, max=1000),
            'labels': torch.randint(0, 10, (2,)),
        })

    model.train()
    transformed_images_train, _ = model.transform(images, targets)
    print("output during training (with fix)")
    print(transformed_images_train.tensors.shape)

    model.eval()
    transformed_images_eval, _ = model.transform(images, targets)
    print("output during evaluation (with fix)")
    print(transformed_images_eval.tensors.shape)
    
    # Verify dimensions
    train_shape = transformed_images_train.tensors.shape[-2:]
    eval_shape = transformed_images_eval.tensors.shape[-2:]
    
    if train_shape == DL_LIB_ETALON_FIXED_SIZE and eval_shape == DL_LIB_ETALON_FIXED_SIZE:
        print("\n✅ SUCCESS: FixedSizeTransformWrapper corrected the dimension swap bug!")
    else:
        print(f"\n❌ FAILURE: Expected {DL_LIB_ETALON_FIXED_SIZE}, but got "
              f"train={train_shape}, eval={eval_shape}")

if __name__ == "__main__":
    # test_something()
    main()
