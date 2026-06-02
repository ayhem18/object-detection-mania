import os
import sys
import cv2
import json
import yaml
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torchvision.ops as ops

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
sys.path.insert(0, os.path.join(road_sign_root, 'scripts', 'training', 'patch_based'))
sys.path.insert(0, os.path.join(road_sign_root, 'scripts', 'training', 'patch_based', 'patch_inference'))

from home_made_od.yolo_family.yolov2.yolov2_model import YoloV2
from basic_inference import PatchInferenceDataset, patch_collate_fn
from mypt.code_utils.pytorch_utils import seed_everything

# Import from refactored train_utils
from road_sign.scripts.training.patch_based.train_scripts.train_utils import split_by_original_image, build_model

def get_val_stems(data_patch_dir, train_ratio=0.9, seed=42):
    _, val_subdirs = split_by_original_image(data_patch_dir, train_ratio=train_ratio, seed=seed)
    return val_subdirs

def main():
    seed_everything(42)
    
    # 1. Configuration - Set your experiment hash here
    EXPERIMENT_HASH = "89906949f97db5b404463e90b7cd7767"
    ARTIFACT_DIR = os.path.join(road_sign_root, 'artifacts', 'patch_based', EXPERIMENT_HASH)
    
    with open(os.path.join(ARTIFACT_DIR, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    DATA_DIR = os.path.join(road_sign_root, config.get("data_dir"))
    SPLIT_HASH_DIR = config.get("split_hash_dir", DATA_DIR)
    
    if not os.path.isabs(SPLIT_HASH_DIR) and not SPLIT_HASH_DIR.startswith(road_sign_root):
        SPLIT_HASH_DIR = os.path.join(DATA_DIR, os.path.basename(SPLIT_HASH_DIR))
    
    # Thresholds for statistics
    THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    
    OUTPUT_ROOT = os.path.join(ARTIFACT_DIR, 'diagnosis', 'precision_ea')
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    anchors_path = os.path.join(SPLIT_HASH_DIR, "anchors.json")
    if not os.path.exists(anchors_path):
        anchors_path = os.path.join(DATA_DIR, "anchors.json")
        
    with open(anchors_path, "r") as f:
        anchors = json.load(f)["anchors"]
        
    patch_config = config.get("patch_config", {})
    target_size = tuple(patch_config.get("target_size", config.get("img_size", [512, 512])))
    patch_scales = patch_config.get("scales", [512, 1024, 2048])
    
    model, transform_stats = build_model(config["num_classes"], len(anchors))
    model.to(device)
    
    checkpoint_path = os.path.join(ARTIFACT_DIR, "checkpoints", "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    # 3. Get Validation Images and Labels
    seed = config.get("seed", 42)
    train_ratio = config.get("train_ratio", 0.9)
    val_stems = get_val_stems(DATA_DIR, train_ratio=train_ratio, seed=seed)
    
    val_img_paths = [os.path.join(road_sign_root, 'data', 'train', 'images', f"{s}.jpg") for s in val_stems]
    val_img_paths = [p for p in val_img_paths if os.path.exists(p)]
    
    # 4. Prepare Background Patch Dataset
    preprocess = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(target_size),
        v2.Normalize(mean=transform_stats.mean, std=transform_stats.std)
    ])
    
    full_dataset = PatchInferenceDataset(
        image_paths=val_img_paths,
        patch_scales=patch_scales,
        target_size=target_size,
        transform=preprocess,
        stride_ratio=0.75
    )
    
    # Filter for background patches
    print("Filtering for background patches...")
    background_meta = []
    
    # Group GTs by image for faster lookup
    gts_by_img = {}
    for img_path in val_img_paths:
        stem = Path(img_path).stem
        label_path = os.path.join(road_sign_root, 'data', 'labels', 'annotations', f"{stem}.txt")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        h, w = img_bgr.shape[:2]
        
        img_gts = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = [float(x) for x in line.split()]
                    if len(parts) >= 5:
                        _, ncx, ncy, nw, nh = parts[:5]
                        x1, y1 = (ncx - nw/2) * w, (ncy - nh/2) * h
                        x2, y2 = (ncx + nw/2) * w, (ncy + nh/2) * h
                        img_gts.append([x1, y1, x2, y2])
        gts_by_img[img_path] = torch.tensor(img_gts) if img_gts else torch.zeros((0, 4))

    for meta in tqdm(full_dataset.patch_metadata, desc="Scanning for GT overlaps"):
        img_path = meta["img_path"]
        patch_box = torch.tensor([[meta["x1"], meta["y1"], meta["x2"], meta["y2"]]], dtype=torch.float32)
        img_gts = gts_by_img.get(img_path)
        
        if img_gts is None or len(img_gts) == 0:
            # Entire image is background
            background_meta.append(meta)
            continue
            
        # Check IoU
        ious = ops.box_iou(patch_box, img_gts)
        if ious.max() == 0:
            # Also check if any GT is completely INSIDE the patch even if IoU is tiny (large patch)
            # or if patch is completely inside a huge GT (unlikely here)
            # Intersection check
            # inter = [max(p1, g1), min(p2, g2)]
            # Actually box_iou handles intersection. If intersection is 0, IoU is 0.
            background_meta.append(meta)
            
    print(f"Found {len(background_meta)} background patches out of {len(full_dataset.patch_metadata)} total.")
    
    # Update dataset to only use background patches
    full_dataset.patch_metadata = background_meta
    
    dataloader = DataLoader(full_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=patch_collate_fn)
    
    # 5. Run Inference on Background Patches and Collect Stats
    patch_stats = []
    
    with torch.no_grad():
        for batch_tensors, batch_metadata in tqdm(dataloader, desc="Analyzing Background Patches"):
            batch_tensors = batch_tensors.to(device)
            # Get raw predictions (before NMS) to see what the model is "hallucinating"
            # model.forward returns (B, K*(5+C), fh, fw)
            raw_output = model(batch_tensors)
            
            # Decode to get scores: (B, num_preds)
            _, scores, _, obj_probs, max_cls_probs = model.decode_predictions(
                raw_output, anchors, target_size, return_all_probs=True
            )
            
            scores = scores.cpu().numpy()
            obj_probs = obj_probs.cpu().numpy()
            max_cls_probs = max_cls_probs.cpu().numpy()
            
            for i in range(len(batch_tensors)):
                s = scores[i]
                op = obj_probs[i]
                cp = max_cls_probs[i]
                
                stat = {
                    "max_score": float(np.max(s)),
                    "max_obj_prob": float(np.max(op)),
                    "max_cls_prob": float(np.max(cp)),
                    "avg_score": float(np.mean(s)),
                }
                
                for t in THRESHOLDS:
                    stat[f"count_above_{t}"] = int(np.sum(s > t))
                    
                patch_stats.append(stat)

    # 6. Aggregate and Save Statistics
    df = pd.DataFrame(patch_stats)
    summary_path = os.path.join(OUTPUT_ROOT, "background_patch_stats.csv")
    df.to_csv(summary_path, index=False)
    
    print("\nBackground Patch Analysis Summary:")
    print(df.describe())
    
    # Save a JSON summary
    summary = df.describe().to_dict()
    with open(os.path.join(OUTPUT_ROOT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nAnalysis complete. Results saved to {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
