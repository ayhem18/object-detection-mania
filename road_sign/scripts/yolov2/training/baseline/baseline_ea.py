import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
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
from home_made_od.yolo_v2.modules.yolov2_diagnosis import analyze_gt_matches, DiagnosisDataset, plot_diagnostic_results
from road_sign.utils.data_utils import YoloFormatDataset, yolov2_collate_fn
from home_made_od.od_metrics import evaluate_model

# We reuse the building logic and anchor calculation from the existing baseline inference
sys.path.insert(0, os.path.join(road_sign_root, 'scripts', 'training', 'baseline'))
from baseline_inference import build_model, get_data_pairs, get_anchors

def main():
    # 1. Configuration matching the baseline training setup
    DATA_DIR = os.path.join(road_sign_root, 'data_512')
    # Using the best model checkpoint from the baseline artifact
    CHECKPOINT_PATH = os.path.join(road_sign_root, 'artifacts', 'baseline', '893b8fced220f0a96492f50a02a8da7e', 'checkpoints', 'best_model.pt')
    
    # Save outputs to baseline artifact folder
    OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(CHECKPOINT_PATH)), 'diagnosis', 'gt_analysis')
    VIS_DIR = os.path.join(OUTPUT_ROOT, 'visualizations')
    os.makedirs(VIS_DIR, exist_ok=True)
    
    NUM_CLASSES = 8
    NUM_ANCHORS = 5
    IMG_SIZE = (512, 512)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    
    # 2. Reproduce Validation Split
    all_pairs = get_data_pairs(DATA_DIR)
    train_size = int(0.8 * len(all_pairs))
    
    # Reproduce random_split logic with seed
    import random
    indices = list(range(len(all_pairs)))
    random.seed(SEED)
    random.shuffle(indices)
    val_indices = indices[train_size:]
    val_pairs = [all_pairs[i] for i in val_indices]
    
    print(f"Analyzing {len(val_pairs)} validation images.")

    # 3. Get Class Mapping
    class_mapping_path = os.path.join(road_sign_root, 'data', 'class_mapping.json')
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)

    # 4. Recalculate Anchors (using full training set as per baseline.py logic for training)
    train_pairs = [all_pairs[i] for i in indices[:train_size]]
    anchors = get_anchors(train_pairs, num_anchors=NUM_ANCHORS)
    
    # 5. Build Model and load weights
    model, transform = build_model(NUM_CLASSES, NUM_ANCHORS)
    model.to(DEVICE)
    
    if not os.path.exists(CHECKPOINT_PATH):
         print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}.")
         return

    print(f"Loading weights from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
        
    transform_stats = {'mean': transform.mean, 'std': transform.std}
    preprocess = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(IMG_SIZE),
        v2.Normalize(mean=transform_stats['mean'], std=transform_stats['std'])
    ])

    # 6. Compute mAP on Validation Set
    CONF_THRESH = 0.05
    NMS_THRESH = 0.4
    MATCH_IOU_THRESH = 0.5

    print(f"Computing mAP on validation set (Conf: {CONF_THRESH}, NMS: {NMS_THRESH})...")
    val_ds = YoloFormatDataset(val_pairs, IMG_SIZE, preprocess)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=yolov2_collate_fn, num_workers=2)
    
    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        anchors=anchors,
        num_classes=NUM_CLASSES,
        conf_threshold=CONF_THRESH,
        nms_iou_threshold=NMS_THRESH,
        match_iou_threshold=MATCH_IOU_THRESH,
        device=DEVICE
    )
    
    print("\nValidation Metrics:")
    for k, v in metrics.items():
        if k.startswith("AP_class"):
            c_idx = k.split("_")[-1]
            c_name = class_mapping.get(str(c_idx), f"Class_{c_idx}")
            print(f"  {c_name} AP: {v:.4f}")
        elif k.startswith("Precision_class") or k.startswith("Recall_class"):
            continue # Clean up console output, keep it in JSON
        else:
            print(f"  {k}: {v:.4f}")
    print("-" * 20)

    # Save metrics to JSON with threshold-dependent name
    metrics_filename = f"metrics_conf{CONF_THRESH}_nms{NMS_THRESH}_match{MATCH_IOU_THRESH}.json"
    metrics_path = os.path.join(OUTPUT_ROOT, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # 7. Prepare Diagnosis Dataset (for easy orig_size access)
    val_img_paths = [p[0] for p in val_pairs]
    diag_ds = DiagnosisDataset(val_img_paths, IMG_SIZE, preprocess)
    
    # Run GT Analysis loop
    from home_made_od.yolo_v2.modules.yolov2_diagnosis import render_deferred_visualizations
    all_diagnostics = []
    all_vis_tasks = []
    
    dataloader = DataLoader(diag_ds, batch_size=16, shuffle=False, num_workers=4)
    
    for batch_tensors, batch_idxs in tqdm(dataloader, desc="GT Matching Analysis"):
        batch_tensors = batch_tensors.to(DEVICE)

        batch_gt_boxes = []
        batch_orig_sizes = []
        batch_image_paths = []
        
        for idx_tensor in batch_idxs:
            idx = idx_tensor.item()
            img_path, label_path = val_pairs[idx]
            orig_w, orig_h = diag_ds.original_sizes[idx]
            
            # Load and scale GT boxes to absolute original coords
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
            
            batch_gt_boxes.append(torch.tensor(gt, device=DEVICE) if gt else torch.zeros((0, 5), device=DEVICE))
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

    # 8. Save quantitative results
    df = pd.DataFrame(all_diagnostics)
    csv_path = os.path.join(OUTPUT_ROOT, "gt_matching_results.csv")
    df.to_csv(csv_path, index=False)
    
    render_deferred_visualizations(all_vis_tasks)

    plot_diagnostic_results(csv_path, os.path.join(OUTPUT_ROOT, 'diagnostic_plots'))
    print(f"Analysis complete. Results saved to {csv_path}")
    print(f"Visualizations saved to {VIS_DIR}")

if __name__ == "__main__":
    main()
