import os
import sys
import json
import yaml
import torch
import pandas as pd
from pathlib import Path
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
sys.path.insert(0, os.path.join(road_sign_root, 'scripts', 'training', 'patch_based'))

from home_made_od.yolo_family.yolov2.yolov2_model import YoloV2
from home_made_od.yolo_family.diagnosis.yolov2_diagnosis import analyze_gt_matches, DiagnosisDataset, plot_diagnostic_results
from home_made_od.od_metrics import evaluate_model
from road_sign.utils.data_utils import YoloFormatDataset, yolov2_collate_fn
from mypt.backbones.resnetFE import ResnetFE
from mypt.code_utils.pytorch_utils import seed_everything

# Import from the refactored train_utils
from road_sign.scripts.training.patch_based.train_scripts.train_utils import split_by_original_image, gather_pairs_from_subdirs, build_model

def main():
    seed_everything(42)
    
    # 1. Configuration - Set your V2 experiment hash here
    EXPERIMENT_HASH = "89906949f97db5b404463e90b7cd7767" # Replace with your actual V2 hash
    ARTIFACT_DIR = os.path.join(road_sign_root, 'artifacts', 'patch_based', EXPERIMENT_HASH)
    
    if not os.path.exists(ARTIFACT_DIR):
        print(f"Error: Artifact directory not found at {ARTIFACT_DIR}")
        return

    with open(os.path.join(ARTIFACT_DIR, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    DATA_DIR = os.path.join(road_sign_root, config.get("data_dir"))
    SPLIT_HASH_DIR = config.get("split_hash_dir", DATA_DIR) # Fallback to DATA_DIR if not present in older configs
    
    # If split_hash_dir is just the basename, join it with DATA_DIR
    if not os.path.isabs(SPLIT_HASH_DIR) and not SPLIT_HASH_DIR.startswith(road_sign_root):
        SPLIT_HASH_DIR = os.path.join(DATA_DIR, os.path.basename(SPLIT_HASH_DIR))

    # Thresholds
    CONF_THRESH = 0.05
    NMS_THRESH = 0.4
    MATCH_IOU_THRESH = 0.5
    
    OUTPUT_ROOT = os.path.join(ARTIFACT_DIR, 'diagnosis', 'gt_analysis')
    VIS_DIR = os.path.join(OUTPUT_ROOT, 'visualizations')
    os.makedirs(VIS_DIR, exist_ok=True)

    # 2. Get exact Validation Split used during training
    # Use the seed and ratio from the config to guarantee identical split
    seed = config.get("seed", 42)
    train_ratio = config.get("train_ratio", 0.9)
    _, val_subdirs = split_by_original_image(DATA_DIR, train_ratio=train_ratio, seed=seed)
    val_pairs = gather_pairs_from_subdirs(DATA_DIR, val_subdirs)
    
    print(f"Evaluating model purely on {len(val_pairs)} generated validation patches.")

    # 3. Load Model and Configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load anchors from the split hash directory (or fallback to DATA_DIR for V1)
    anchors_path = os.path.join(SPLIT_HASH_DIR, "anchors.json")
    if not os.path.exists(anchors_path):
        anchors_path = os.path.join(DATA_DIR, "anchors.json")
        
    with open(anchors_path, "r") as f:
        anchors = json.load(f)["anchors"]
    
    IMG_SIZE = tuple(config.get("img_size", [512, 512]))
    NUM_CLASSES = config["num_classes"]
    NUM_ANCHORS = len(anchors)
    
    model, transform = build_model(NUM_CLASSES, NUM_ANCHORS)
    model.to(device)
    
    checkpoint_path = os.path.join(ARTIFACT_DIR, "checkpoints", "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    # 4. Prepare Dataset exactly as in training
    transform_stats = {'mean': transform.mean, 'std': transform.std}
    preprocess = v2.Compose([
        v2.Resize(IMG_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=transform_stats['mean'], std=transform_stats['std'])
    ])
    
    print(f"Computing pure model metrics on validation split (Conf: {CONF_THRESH}, NMS: {NMS_THRESH})...")
    val_ds = YoloFormatDataset(val_pairs, IMG_SIZE, preprocess)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=yolov2_collate_fn, num_workers=2)

    # 5. Compute Standard Metrics
    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        anchors=anchors,
        num_classes=NUM_CLASSES,
        conf_threshold=CONF_THRESH,
        nms_iou_threshold=NMS_THRESH,
        match_iou_threshold=MATCH_IOU_THRESH,
        device=device
    )
    
    class_mapping_path = os.path.join(road_sign_root, 'data', 'class_mapping.json')
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
        
    print("\nPure Patch Domain Metrics:")
    for k, v in metrics.items():
        if k.startswith("AP_class"):
            c_idx = k.split("_")[-1]
            c_name = class_mapping.get(str(c_idx), f"Class_{c_idx}")
            print(f"  {c_name} AP: {v:.4f}")
        elif k.startswith("Precision_class") or k.startswith("Recall_class"):
            continue
        else:
            print(f"  {k}: {v:.4f}")
    print("-" * 20)

    metrics_filename = f"pure_metrics_conf{CONF_THRESH}_nms{NMS_THRESH}.json"
    metrics_path = os.path.join(OUTPUT_ROOT, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


    # 6. Run GT Analysis Loop for deep diagnosis
    print("\nRunning GT Match Analysis (IoU, Areas, Objectness)...")
    val_img_paths = [p[0] for p in val_pairs]
    diag_ds = DiagnosisDataset(val_img_paths, IMG_SIZE, preprocess)
    diag_loader = DataLoader(diag_ds, batch_size=16, shuffle=False, num_workers=4)
    
    from home_made_od.yolo_family.diagnosis.yolov2_diagnosis import render_deferred_visualizations
    all_diagnostics = []
    all_vis_tasks = []
    
    for batch_tensors, batch_idxs in tqdm(diag_loader, desc="GT Matching"):
        batch_tensors = batch_tensors.to(device)
        
        batch_gt_boxes = []
        batch_orig_sizes = []
        batch_image_paths = []
        
        for idx_tensor in batch_idxs:
            idx = idx_tensor.item()
            img_path, label_path = val_pairs[idx]
            orig_w, orig_h = diag_ds.original_sizes[idx]
            
            gt = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = [float(x) for x in line.split()]
                        cls, ncx, ncy, nw, nh = parts[:5]
                        
                        x1 = (ncx - nw/2) * orig_h
                        y1 = (ncy - nh/2) * orig_w
                        x2 = (ncx + nw/2) * orig_h
                        y2 = (ncy + nh/2) * orig_w
                        
                        gt.append([cls, x1, y1, x2, y2])
            
            batch_gt_boxes.append(torch.tensor(gt, device=device) if gt else torch.zeros((0, 5), device=device))
            batch_orig_sizes.append((orig_w, orig_h))
            batch_image_paths.append(img_path)
            
        batch_diags, batch_vis_tasks = analyze_gt_matches(
            model=model,
            images=batch_tensors,
            gt_boxes_list=batch_gt_boxes,
            anchors=anchors,
            orig_sizes=batch_orig_sizes,
            visualize=True,
            output_dir=VIS_DIR,
            image_paths=batch_image_paths,
            class_mapping=class_mapping,
            defer_visualization=True
        )
        all_diagnostics.extend(batch_diags)
        all_vis_tasks.extend(batch_vis_tasks)

    # 7. Save and Plot Diagnostics
    df = pd.DataFrame(all_diagnostics)
    csv_path = os.path.join(OUTPUT_ROOT, "pure_gt_matching_results.csv")
    df.to_csv(csv_path, index=False)
    
    render_deferred_visualizations(all_vis_tasks)
    
    plot_diagnostic_results(csv_path, os.path.join(OUTPUT_ROOT, 'diagnostic_plots'))
    print(f"Deep Analysis complete. Plots saved to {OUTPUT_ROOT}/diagnostic_plots")

if __name__ == "__main__":
    main()
