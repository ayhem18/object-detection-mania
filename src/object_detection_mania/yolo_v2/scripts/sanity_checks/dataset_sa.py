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
        # Convert normalized to pixel coordinates
        xmin = (cx - w/2) * img_size[1]
        ymin = (cy - h/2) * img_size[0]
        rect_w = w * img_size[1]
        rect_h = h * img_size[0]
        
        rect = patches.Rectangle(
            (xmin, ymin), rect_w, rect_h,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(xmin, ymin, f"Cls: {int(cls_id)}", color='white', backgroundcolor='red', fontsize=8)

def main():
    # 1. Setup paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent.parent.parent
    artifacts_dir = project_root / "src" / "object_detection_mania" / "artifacts" / "yolo_v2_artifacts" / "synthetic"
    
    config_path = artifacts_dir / "configs" / "val_2000_42_config.json"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return
        
    output_dir = artifacts_dir / "visualizations" / "dataset_sa"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Initialize Dataset
    img_size = (416, 416)
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(img_size)
    ])
    
    ds = YoloV2Dataset(
        config_path=str(config_path),
        output_dir=str(artifacts_dir / "rendered" / "val_sa"),
        img_shape=img_size,
        transforms=transforms
    )
    
    # 3. Visualize first 10 samples with objects
    num_samples = 10
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    samples_found = 0
    idx = 0
    while samples_found < num_samples and idx < len(ds):
        img, targets = ds[idx]
        idx += 1
        
        if targets.shape[0] == 0:
            continue
            
        # Convert (C, H, W) tensor to (H, W, C) for plotting
        vis_img = img.permute(1, 2, 0).numpy()
        
        axes[samples_found].imshow(vis_img)
        draw_boxes(axes[samples_found], targets, img_size)
        axes[samples_found].set_title(f"Sample {idx-1}")
        axes[samples_found].axis('off')
        samples_found += 1
        
    plt.tight_layout()
    save_path = output_dir / "dataset_samples.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Sanity check visualization saved to {save_path}")

if __name__ == "__main__":
    main()
