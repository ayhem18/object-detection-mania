import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from torchvision.transforms import v2
from object_detection_mania.yolo_v2.modules.yolov2_ds import YoloV2Dataset

def draw_boxes(ax, targets, img_size):
    """
    targets: (N, 5) -> [cls_id, cx, cy, w, h]
    """
    for target in targets:
        cls_id, cx, cy, w, h = target.tolist()
        xmin = (cx - w/2) * img_size[1]
        ymin = (cy - h/2) * img_size[0]
        rect_w = w * img_size[1]
        rect_h = h * img_size[0]
        
        rect = patches.Rectangle(
            (xmin, ymin), rect_w, rect_h,
            linewidth=2, edgecolor='cyan', facecolor='none'
        )
        ax.add_patch(rect)

def run_single_aug_check(aug_name, transform, config_path, artifacts_dir, img_size):
    print(f"Running sanity check for: {aug_name}...")
    output_dir = artifacts_dir / "visualizations" / "aug_sa" / aug_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ds = YoloV2Dataset(
        config_path=str(config_path),
        output_dir=str(artifacts_dir / "rendered" / "val_sa"),
        img_shape=img_size,
        transforms=transform
    )
    
    num_samples = 5
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 5 * num_samples))
    if num_samples == 1: axes = [axes]
    
    samples_found = 0
    idx = 0
    while samples_found < num_samples and idx < len(ds):
        img, targets = ds[idx]
        idx += 1
        
        if targets.shape[0] == 0:
            continue
            
        vis_img = img.permute(1, 2, 0).numpy()
        
        axes[samples_found].imshow(vis_img)
        draw_boxes(axes[samples_found], targets, img_size)
        axes[samples_found].set_title(f"{aug_name} Sample {idx-1}")
        axes[samples_found].axis('off')
        samples_found += 1
        
    plt.tight_layout()
    save_path = output_dir / f"{aug_name}_samples.png"
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved to {save_path}")

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent.parent.parent
    artifacts_dir = project_root / "src" / "object_detection_mania" / "artifacts" / "yolo_v2_artifacts" / "synthetic"
    
    config_path = artifacts_dir / "configs" / "val_2000_42_config.json"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return
        
    img_size = (416, 416)
    
    # Define individual augmentations to test
    # The dataset class will automatically append ToImage, ToDtype, and Resize
    augmentations = {
        "HorizontalFlip": v2.RandomHorizontalFlip(p=1.0),
        "VerticalFlip": v2.RandomVerticalFlip(p=1.0),
        "Rotation": v2.RandomRotation(degrees=30),
        "Affine": v2.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        "ResizedCrop": v2.RandomResizedCrop(size=img_size, scale=(0.5, 1.0))
    }
    
    for aug_name, transform in augmentations.items():
        run_single_aug_check(aug_name, transform, config_path, artifacts_dir, img_size)

if __name__ == "__main__":
    main()
