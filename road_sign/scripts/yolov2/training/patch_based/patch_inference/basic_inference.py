import os
import sys
import cv2
import json
import yaml
import torch
import numpy as np
import pandas as pd
import shutil
import concurrent.futures

from tqdm import tqdm
from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

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

from home_made_od.yolo_family.yolov2.yolov2_model import YoloV2
from mypt.code_utils.pytorch_utils import seed_everything
import torchvision.ops as ops

# Import from refactored train_utils
from road_sign.scripts.training.patch_based.train_scripts.train_utils import build_model


# --- STAGE 1: Fast Parallel Patch Generation ---

def _extract_and_save_patches(args):
    """Worker function for parallel patch extraction."""
    img_path, patch_list, output_dir, target_size = args
    img = cv2.imread(img_path)
    if img is None:
        return False
        
    for meta in patch_list:
        x1, y1, x2, y2 = meta["x1"], meta["y1"], meta["x2"], meta["y2"]
        scale = meta["scale"]
        patch = img[y1:y2, x1:x2]
        
        ph, pw = patch.shape[:2]
        if ph < scale or pw < scale:
            patch = cv2.copyMakeBorder(patch, 0, scale - ph, 0, scale - pw, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
        # Resize to model input size offline
        if (scale, scale) != target_size:
            patch = cv2.resize(patch, target_size)
            
        out_name = f"{meta['patch_id']}.jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), patch)
    return True

def prepare_patches_offline(image_paths, patch_scales, target_size, tmp_patch_dir, stride_ratio=0.75):
    """Orchestrates parallel extraction of patches to disk."""
    if os.path.exists(tmp_patch_dir):
        shutil.rmtree(tmp_patch_dir)
    os.makedirs(tmp_patch_dir, exist_ok=True)
    
    tasks = []
    all_metadata = []
    
    print("Pre-calculating patch coordinates...")
    for img_path in image_paths:
        img_path_str = str(img_path)
        img_info = cv2.imread(img_path_str) # Just for dims
        if img_info is None: continue
        orig_h, orig_w = img_info.shape[:2]
        
        img_patches = []
        for scale in patch_scales:
            stride = int(scale * stride_ratio)
            y_starts = sorted(list(set(list(range(0, orig_h, stride)) + [max(0, orig_h - scale)])))
            x_starts = sorted(list(set(list(range(0, orig_w, stride)) + [max(0, orig_w - scale)])))

            for y in y_starts:
                for x in x_starts:
                    patch_id = f"{Path(img_path).stem}_s{scale}_y{y}_x{x}"
                    meta = {
                        "patch_id": patch_id,
                        "img_path": img_path_str,
                        "x1": x, "y1": y, "x2": min(x + scale, orig_w), "y2": min(y + scale, orig_h),
                        "scale": scale, "orig_w": orig_w, "orig_h": orig_h
                    }
                    img_patches.append(meta)
                    all_metadata.append(meta)
        tasks.append((img_path_str, img_patches, tmp_patch_dir, target_size))

    print(f"Extracting {len(all_metadata)} patches using multiprocessing...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(_extract_and_save_patches, tasks), total=len(tasks), desc="Saving patches"))
    
    return all_metadata


# --- STAGE 2: High-Performance Inference ---

class PreSlicedPatchDataset(Dataset):
    """Loads pre-saved patches from disk."""
    def __init__(self, metadata, patch_dir, transform):
        self.metadata = metadata
        self.patch_dir = patch_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        patch_path = os.path.join(self.patch_dir, f"{meta['patch_id']}.jpg")
        img_bgr = cv2.imread(patch_path)
        if img_bgr is None:
            return torch.zeros((3, 512, 512)), idx # Fallback
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), idx

def global_nms(boxes, scores, labels, iou_threshold=0.4):
    if len(boxes) == 0: return np.array([]), np.array([]), np.array([])
    offsets = labels * 100000 
    keep = ops.nms(boxes + offsets.unsqueeze(1), scores, iou_threshold)
    return boxes[keep].cpu().numpy(), scores[keep].cpu().numpy(), labels[keep].cpu().numpy()

def run_optimized_patch_inference(artifact_dir, test_images_dir, output_csv, batch_size=64, conf_thresh=0.05, iou_thresh=0.4):
    # 1. Setup paths and load configs
    with open(os.path.join(artifact_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    data_dir_full = os.path.join(road_sign_root, config.get("data_dir"))
    SPLIT_HASH_DIR = config.get("split_hash_dir", data_dir_full)
    
    if not os.path.isabs(SPLIT_HASH_DIR) and not SPLIT_HASH_DIR.startswith(road_sign_root):
        SPLIT_HASH_DIR = os.path.join(data_dir_full, os.path.basename(SPLIT_HASH_DIR))
    
    anchors_path = os.path.join(SPLIT_HASH_DIR, "anchors.json")
    if not os.path.exists(anchors_path):
        anchors_path = os.path.join(data_dir_full, "anchors.json")
        
    with open(anchors_path, "r") as f:
        anchors = json.load(f)["anchors"]
        
    patch_config = config.get("patch_config", {})
    target_size = tuple(patch_config.get("target_size", config.get("img_size", [512, 512])))
    patch_scales = patch_config.get("scales", [512, 1024, 2048])
    
    tmp_patch_dir = os.path.join(road_sign_root, "temp_inference_patches")
    
    # 2. Stage 1: Parallel Patch Extraction
    test_images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        test_images.extend(list(Path(test_images_dir).glob(ext)))
        
    if not test_images:
        print(f"No images found in {test_images_dir}")
        return
        
    all_metadata = prepare_patches_offline(test_images, patch_scales, target_size, tmp_patch_dir)

    # 3. Stage 2: Fast Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_obj, transform_stats = build_model(config["num_classes"], len(anchors))
    model_obj.to(device)
    
    checkpoint = torch.load(os.path.join(artifact_dir, "checkpoints", "best_model.pt"), map_location=device)
    model_obj.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model_obj.eval()

    preprocess = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True), 
        v2.Normalize(mean=transform_stats.mean, std=transform_stats.std)
    ])
    
    dataset = PreSlicedPatchDataset(all_metadata, tmp_patch_dir, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

    aggregated_detections = {Path(p).stem: {"boxes": [], "scores": [], "labels": []} for p in test_images}

    with torch.no_grad():
        for batch_tensors, batch_idxs in tqdm(dataloader, desc="Inferencing Patches"):
            batch_tensors = batch_tensors.to(device)
            batch_preds = model_obj.inference(batch_tensors, anchors=anchors, conf_threshold=conf_thresh, nms_iou_threshold=iou_thresh)
            
            for i, preds in enumerate(batch_preds):
                if len(preds) == 0: continue
                meta = all_metadata[batch_idxs[i]]
                img_stem = Path(meta["img_path"]).stem
                
                # GPU Vectorized Scaling
                sx, sy = meta["scale"] / target_size[0], meta["scale"] / target_size[1]
                ox, oy = meta["x1"], meta["y1"]
                
                preds[:, 0] = torch.clamp(preds[:, 0] * sx + ox, 0, meta["orig_w"])
                preds[:, 1] = torch.clamp(preds[:, 1] * sy + oy, 0, meta["orig_h"])
                preds[:, 2] = torch.clamp(preds[:, 2] * sx + ox, 0, meta["orig_w"])
                preds[:, 3] = torch.clamp(preds[:, 3] * sy + oy, 0, meta["orig_h"])
                
                valid = (preds[:, 2] > preds[:, 0]) & (preds[:, 3] > preds[:, 1])
                if valid.any():
                    v_preds = preds[valid]
                    aggregated_detections[img_stem]["boxes"].append(v_preds[:, :4])
                    aggregated_detections[img_stem]["scores"].append(v_preds[:, 4])
                    aggregated_detections[img_stem]["labels"].append(v_preds[:, 5])

    # 4. Final Aggregation and Submission
    print("Applying global NMS and formatting submission...")
    class_mapping_path = os.path.join(road_sign_root, 'data', 'class_mapping.json')
    with open(class_mapping_path, 'r') as f:
        class_names = [v.replace(" ", "_") for v in json.load(f).values()]

    results = []
    for stem, dets in aggregated_detections.items():
        if not dets["boxes"]:
            results.append({"image_id": stem, "PredictionString": "No_parking 0.0001 0 0 1 1"})
            continue
        fb, fs, fl = global_nms(torch.cat(dets["boxes"]), torch.cat(dets["scores"]), torch.cat(dets["labels"]), iou_thresh)
        p_strs = [f"{class_names[int(l)]} {s:.4f} {int(b[0])} {int(b[1])} {int(b[2])} {int(b[3])}" for b, s, l in zip(fb, fs, fl)]
        results.append({"image_id": stem, "PredictionString": " ".join(p_strs)})

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    shutil.rmtree(tmp_patch_dir)
    print(f"Submission saved to {output_csv}")

if __name__ == "__main__":
    # Replace with your actual V2 experiment hash to run inference
    EXPERIMENT_HASH = "89906949f97db5b404463e90b7cd7767"
    ARTIFACT_DIR = os.path.join(road_sign_root, 'artifacts', 'patch_based', EXPERIMENT_HASH)
    TEST_DIR = os.path.join(road_sign_root, 'data', 'test', 'images')
    
    CONF_THRESH = 0.05
    NMS_THRESH = 0.4
    
    filename = f'submission_patch_conf{CONF_THRESH}_nms{NMS_THRESH}.csv'
    OUTPUT_CSV = os.path.join(ARTIFACT_DIR, 'submission', filename)
    
    run_optimized_patch_inference(ARTIFACT_DIR, TEST_DIR, OUTPUT_CSV, batch_size=4, conf_thresh=CONF_THRESH, iou_thresh=NMS_THRESH)
