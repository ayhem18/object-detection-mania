import os
import sys
import cv2
import torch
import yaml
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any
from torchvision.transforms import v2
from ensemble_boxes import weighted_boxes_fusion

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

from home_made_od.yolo_v2.modules.yolov2_model import YoloV2
from mypt.backbones.resnetFE import ResnetFE
from mypt.code_utils.pytorch_utils import seed_everything

# =============================================================================
# 1. Model Loading
# =============================================================================

def load_model_from_artifacts(artifact_dir: str, device: torch.device):
    """Loads configuration, anchors, and the best model from the artifact directory."""
    config_path = os.path.join(artifact_dir, "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Anchors were saved in the data directory by prepare_patch_ds.py
    data_dir = os.path.join(road_sign_root, config["data_dir"])
    anchors_path = os.path.join(data_dir, "anchors.json")
    with open(anchors_path, 'r') as f:
        anchors = json.load(f)["anchors"]

    # Reconstruct Model
    resnet_fe = ResnetFE(
        build_by_layer=True,
        num_extracted_layers=-1,
        num_extracted_bottlenecks=-1,
        freeze=2,
        freeze_by_layer=True,
        add_global_average=False,
        architecture=50
    )
    
    model = YoloV2(
        backbone=resnet_fe,
        backbone_out_channels=2048,
        num_anchors=len(anchors),
        num_classes=config["num_classes"],
        num_conv_blocks=2
    )

    # Load Weights
    checkpoint_path = os.path.join(artifact_dir, "checkpoints", "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    
    return model, config, anchors, resnet_fe.transform

# =============================================================================
# 2. Sliding Window & Patching Logic
# =============================================================================

def get_sliding_window_coords(img_shape: Tuple[int, int], scale: int, stride_ratio: float = 0.75) -> List[Tuple[int, int, int, int]]:
    """Calculates coordinates [x1, y1, x2, y2] for a sliding window at a specific scale."""
    h_img, w_img = img_shape
    stride = int(scale * stride_ratio)
    
    x_starts = list(range(0, w_img - scale + 1, stride))
    if x_starts[-1] + scale < w_img:
        x_starts.append(w_img - scale)
        
    y_starts = list(range(0, h_img - scale + 1, stride))
    if y_starts[-1] + scale < h_img:
        y_starts.append(h_img - scale)
        
    coords = []
    for y in y_starts:
        for x in x_starts:
            coords.append((x, y, x + scale, y + scale))
    return coords

def extract_and_preprocess_patch(img: np.ndarray, coords: Tuple[int, int, int, int], target_size: Tuple[int, int], transform: Any) -> torch.Tensor:
    """Crops, resizes, and transforms a single patch."""
    x1, y1, x2, y2 = coords
    patch = img[y1:y2, x1:x2]
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    
    # transform logic (v2 expects Image or Tensor)
    # Using the Imagenet transform passed from backbone
    img_tensor = v2.functional.to_image(patch_rgb)
    img_tensor = v2.functional.to_dtype(img_tensor, torch.float32, scale=True)
    img_tensor = v2.functional.resize(img_tensor, target_size)
    img_tensor = v2.functional.normalize(img_tensor, mean=transform.mean, std=transform.std)
    
    return img_tensor

# =============================================================================
# 3. Coordinate Remapping
# =============================================================================

def remap_to_global(patch_preds: torch.Tensor, patch_coords: Tuple[int, int, int, int], patch_scale: int, img_size: Tuple[int, int]) -> List[Dict]:
    """Converts patch-local detections to global image coordinates."""
    x_offset, y_offset, _, _ = patch_coords
    
    # scale_factor maps from the model input (512) back to the patch scale (e.g. 1024)
    scale_factor = patch_scale / 512.0
    
    global_dets = []
    for det in patch_preds:
        x1, y1, x2, y2, score, cls_id = det.cpu().numpy()
        
        # 1. Scale back to patch resolution
        p_x1, p_y1 = x1 * scale_factor, y1 * scale_factor
        p_x2, p_y2 = x2 * scale_factor, y2 * scale_factor
        
        # 2. Add global offset
        g_x1, g_y1 = p_x1 + x_offset, p_y1 + y_offset
        g_x2, g_y2 = p_x2 + x_offset, p_y2 + y_offset
        
        # 3. Normalize for WBF (ensemble_boxes expects 0.0-1.0)
        global_dets.append({
            "box": [g_x1 / img_size[0], g_y1 / img_size[1], g_x2 / img_size[0], g_y2 / img_size[1]],
            "score": score,
            "label": int(cls_id)
        })
        
    return global_dets

# =============================================================================
# 4. Inference & Fusion
# =============================================================================

def run_image_inference(model, img: np.ndarray, config: Dict, anchors: List, transform: Any, device: torch.device):
    """Processes a full image across multiple scales and fuses results."""
    h_orig, w_orig = img.shape[:2]
    scales = config.get("patch_config", {}).get("scales", [512, 1024, 2048])
    target_size = tuple(config["img_size"])
    batch_size = config["batch_size"]
    
    all_global_dets = []
    
    for scale in scales:
        coords_list = get_sliding_window_coords((h_orig, w_orig), scale)
        
        # Process patches in batches
        for i in range(0, len(coords_list), batch_size):
            batch_coords = coords_list[i : i + batch_size]
            batch_tensors = [extract_and_preprocess_patch(img, c, target_size, transform) for c in batch_coords]
            batch_input = torch.stack(batch_tensors).to(device)
            
            with torch.no_grad():
                # model.inference returns List[Tensor]
                batch_preds = model.inference(batch_input, anchors=anchors, conf_threshold=0.01, nms_iou_threshold=0.5)
            
            for j, preds in enumerate(batch_preds):
                if len(preds) > 0:
                    remapped = remap_to_global(preds, batch_coords[j], scale, (w_orig, h_orig))
                    all_global_dets.extend(remapped)
                    
    if not all_global_dets:
        return []

    # Prepare for WBF
    boxes = [d["box"] for d in all_global_dets]
    scores = [d["score"] for d in all_global_dets]
    labels = [d["label"] for d in all_global_dets]
    
    # ensemble_boxes expects list of lists
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        [boxes], [scores], [labels], weights=None, iou_thr=0.45, skip_box_thr=0.1
    )
    
    final_dets = []
    for b, s, l in zip(fused_boxes, fused_scores, fused_labels):
        # Scale back to absolute pixels
        x1, y1, x2, y2 = b[0] * w_orig, b[1] * h_orig, b[2] * w_orig, b[3] * h_orig
        final_dets.append((x1, y1, x2, y2, s, int(l)))
        
    return final_dets

# =============================================================================
# 5. Main Execution
# =============================================================================

def main(artifact_dir: str):
    seed_everything(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading environment from {artifact_dir}...")
    model, config, anchors, transform = load_model_from_artifacts(artifact_dir, DEVICE)
    
    # Setup paths
    test_dir = os.path.join(road_sign_root, 'data', 'test', 'images')
    submission_dir = os.path.join(artifact_dir, "submission")
    os.makedirs(submission_dir, exist_ok=True)
    output_csv = os.path.join(submission_dir, "submission_patch_wbf.csv")
    
    class_mapping_path = os.path.join(road_sign_root, 'data', 'class_mapping.json')
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    class_names = [class_mapping[str(i)].replace(" ", "_") for i in range(len(class_mapping))]
    
    test_images = list(Path(test_dir).glob("*.jpg")) + list(Path(test_dir).glob("*.png"))
    print(f"Found {len(test_images)} test images. Starting Patch-Based Inference...")
    
    results = []
    
    for img_path in tqdm(test_images):
        img = cv2.imread(str(img_path))
        if img is None:
            results.append({"image_id": img_path.stem, "PredictionString": "No_parking 0.0001 0 0 1 1"})
            continue
            
        detections = run_image_inference(model, img, config, anchors, transform, DEVICE)
        
        prediction_strings = []
        for d in detections:
            x1, y1, x2, y2, score, cls_id = d
            label = class_names[cls_id]
            prediction_strings.append(f"{label} {score:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}")
            
        final_str = " ".join(prediction_strings) if prediction_strings else "No_parking 0.0001 0 0 1 1"
        results.append({"image_id": img_path.stem, "PredictionString": final_str})
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n[Success] Submission saved to {output_csv}")

if __name__ == "__main__":
    # Example usage: provide the hash directory of your latest experiment
    # You can find this in road_sign/artifacts/patch_based/
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, required=True, help="Path to the experiment folder in artifacts/patch_based/")
    args = parser.parse_args()
    
    main(args.artifact_dir)
