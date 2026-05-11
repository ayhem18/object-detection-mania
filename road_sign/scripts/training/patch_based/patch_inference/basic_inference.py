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
sys.path.insert(0, os.path.join(road_sign_root, 'scripts', 'training', 'patch_based'))

from home_made_od.yolo_v2.modules.yolov2_model import YoloV2
from mypt.backbones.resnetFE import ResnetFE
from mypt.code_utils.pytorch_utils import seed_everything
import torchvision.ops as ops


class PatchInferenceDataset(Dataset):
    """
    A PyTorch Dataset that takes full-sized images and extracts overlapping patches.
    It returns the transformed patch tensor along with metadata needed to reconstruct
    the detection coordinates back to the original image space.
    """
    def __init__(self, image_paths, patch_scales, target_size, transform, stride_ratio=0.75):
        self.image_paths = image_paths
        self.patch_scales = patch_scales
        self.target_size = target_size
        self.transform = transform
        self.stride_ratio = stride_ratio
        
        self.patch_metadata = [] 
        
        # --- Caching Mechanism ---
        self._current_img_path = None
        self._current_img_bgr = None
        
        print(f"Initializing PatchInferenceDataset with scales {patch_scales} and stride ratio {stride_ratio}...")
        
        # We must sort the image paths so patches from the same image are contiguous.
        # This is critical for the caching mechanism to be effective across batch loading.
        self.image_paths = sorted([str(p) for p in self.image_paths])
        
        for img_path_str in tqdm(self.image_paths, desc="Scanning images for patches"):
            img = cv2.imread(img_path_str)
            if img is None:
                continue
            orig_h, orig_w = img.shape[:2]
            
            for scale in self.patch_scales:
                stride = int(scale * self.stride_ratio)
                
                y_starts = list(range(0, orig_h, stride))
                x_starts = list(range(0, orig_w, stride))
                
                if not y_starts or y_starts[-1] + scale < orig_h:
                    y_starts.append(max(0, orig_h - scale))
                if not x_starts or x_starts[-1] + scale < orig_w:
                    x_starts.append(max(0, orig_w - scale))
                
                y_starts = sorted(list(set(y_starts)))
                x_starts = sorted(list(set(x_starts)))

                for y in y_starts:
                    for x in x_starts:
                        self.patch_metadata.append({
                            "img_path": img_path_str,
                            "x1": x,
                            "y1": y,
                            "x2": min(x + scale, orig_w),
                            "y2": min(y + scale, orig_h),
                            "scale": scale,
                            "orig_w": orig_w,
                            "orig_h": orig_h
                        })

        # Sort metadata by img_path to maximize cache hits when DataLoader reads sequentially
        self.patch_metadata.sort(key=lambda x: x["img_path"])
        print(f"Created dataset with {len(self.patch_metadata)} total patches from {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.patch_metadata)

    def _get_image(self, img_path):
        """Loads image or returns cached version to prevent redundant disk I/O."""
        if self._current_img_path != img_path:
            self._current_img_bgr = cv2.imread(img_path)
            self._current_img_path = img_path
        return self._current_img_bgr

    def __getitem__(self, idx):
        meta = self.patch_metadata[idx]
        img_path = meta["img_path"]
        x1, y1, x2, y2 = meta["x1"], meta["y1"], meta["x2"], meta["y2"]
        
        img_bgr = self._get_image(img_path)
        
        if img_bgr is None:
             return torch.zeros((3, self.target_size[1], self.target_size[0])), meta
             
        patch_bgr = img_bgr[y1:y2, x1:x2]
        
        ph, pw = patch_bgr.shape[:2]
        pad_bottom = meta["scale"] - ph
        pad_right = meta["scale"] - pw
        
        if pad_bottom > 0 or pad_right > 0:
            patch_bgr = cv2.copyMakeBorder(
                patch_bgr, 0, pad_bottom, 0, pad_right, 
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(patch_rgb)
        
        return input_tensor, meta


def build_inference_model(num_classes, num_anchors):
    resnet_fe = ResnetFE(
        build_by_layer=True,
        num_extracted_layers=-1, 
        num_extracted_bottlenecks=-1,
        freeze=2,
        freeze_by_layer=True,
        add_global_average=False,
        architecture=50
    )
    
    backbone_out_channels = 2048 
    
    model = YoloV2(
        backbone=resnet_fe,
        backbone_out_channels=backbone_out_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        num_conv_blocks=2
    )
    
    return model, resnet_fe.transform


def patch_collate_fn(batch):
    tensors = [item[0] for item in batch]
    metadata = [item[1] for item in batch]
    
    return torch.stack(tensors, dim=0), metadata


def global_nms(boxes, scores, labels, iou_threshold=0.4):
    """
    Applies standard NMS across the aggregated detections of an entire image.
    Offset trick is used to make it class-aware.
    """
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Class-aware NMS trick
    max_wh = 100000 
    offsets = labels * max_wh
    
    boxes_for_nms = boxes + offsets.unsqueeze(1)
    keep = ops.nms(boxes_for_nms, scores, iou_threshold)
    
    return boxes[keep].numpy(), scores[keep].numpy(), labels[keep].numpy()


def run_patch_inference(artifact_dir, test_images_dir, output_csv, batch_size=32, conf_thresh=0.25, iou_thresh=0.4):
    """
    Main entry point for patch inference.
    """
    print(f"Starting patch inference using artifacts from: {artifact_dir}")
    
    config_path = os.path.join(artifact_dir, "config.yaml")
    checkpoint_path = os.path.join(artifact_dir, "checkpoints", "best_model.pt")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found in {artifact_dir}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    data_dir_name = config.get("data_dir")
    data_dir_full = os.path.join(road_sign_root, data_dir_name)
    
    anchors_path = os.path.join(data_dir_full, "anchors.json")
    with open(anchors_path, "r") as f:
        anchors = json.load(f)["anchors"]
        
    patch_config_path = os.path.join(data_dir_full, "config.yaml")
    with open(patch_config_path, "r") as f:
        patch_config = yaml.safe_load(f)
        
    class_mapping_path = os.path.join(road_sign_root, 'data', 'class_mapping.json')
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
        
    class_names = []
    for i in range(len(class_mapping)):
        raw_name = class_mapping.get(str(i), str(i))
        class_names.append(raw_name.replace(" ", "_"))
        
    num_classes = config["num_classes"]
    num_anchors = len(anchors)
    target_size = tuple(patch_config["target_size"])
    patch_scales = patch_config["scales"]

    print(f"Loaded config. Target Size: {target_size}, Patch Scales: {patch_scales}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, transform_stats = build_inference_model(num_classes, num_anchors)
    model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    preprocess = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(target_size),
        v2.Normalize(mean=transform_stats.mean, std=transform_stats.std)
    ])
    
    test_images = list(Path(test_images_dir).glob("*.jpg")) + list(Path(test_images_dir).glob("*.png"))
    
    # Num_workers=0 is required when using the caching mechanism in the dataset
    # because multiprocessing dataloaders spawn processes that copy the dataset,
    # defeating the stateful cache. Since disk I/O is reduced by ~95%, single-thread is fast.
    dataset = PatchInferenceDataset(
        image_paths=test_images,
        patch_scales=patch_scales,
        target_size=target_size,
        transform=preprocess,
        stride_ratio=0.75
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=patch_collate_fn
    )

    # Dictionary to hold chunked tensors per original image
    aggregated_detections = {Path(p).stem: {"boxes": [], "scores": [], "labels": []} for p in test_images}

    with torch.no_grad():
        for batch_tensors, batch_metadata in tqdm(dataloader, desc="Inferencing Patches"):
            batch_tensors = batch_tensors.to(device)
            
            # batch_preds is a list of tensors: (N, 6) -> [x1, y1, x2, y2, score, class_id]
            batch_preds = model.inference(batch_tensors, anchors=anchors, conf_threshold=conf_thresh, nms_iou_threshold=iou_thresh)
            
            for i, preds in enumerate(batch_preds):
                if len(preds) == 0:
                    continue
                    
                meta = batch_metadata[i]
                img_stem = Path(meta["img_path"]).stem
                
                # Vectorized Scaling and Offsets on GPU
                scale_x = meta["scale"] / target_size[0]
                scale_y = meta["scale"] / target_size[1]
                offset_x = meta["x1"]
                offset_y = meta["y1"]
                orig_w = meta["orig_w"]
                orig_h = meta["orig_h"]

                # Apply scaling
                preds[:, 0] = preds[:, 0] * scale_x + offset_x
                preds[:, 1] = preds[:, 1] * scale_y + offset_y
                preds[:, 2] = preds[:, 2] * scale_x + offset_x
                preds[:, 3] = preds[:, 3] * scale_y + offset_y
                
                # Clamp to image boundaries
                preds[:, 0] = torch.clamp(preds[:, 0], min=0, max=orig_w)
                preds[:, 1] = torch.clamp(preds[:, 1], min=0, max=orig_h)
                preds[:, 2] = torch.clamp(preds[:, 2], min=0, max=orig_w)
                preds[:, 3] = torch.clamp(preds[:, 3], min=0, max=orig_h)
                
                # Filter out degenerate boxes (x2 <= x1 or y2 <= y1)
                valid_mask = (preds[:, 2] > preds[:, 0]) & (preds[:, 3] > preds[:, 1])
                valid_preds = preds[valid_mask]
                
                if len(valid_preds) > 0:
                    # Append raw chunked tensors to dictionary (remains on GPU until NMS)
                    aggregated_detections[img_stem]["boxes"].append(valid_preds[:, :4])
                    aggregated_detections[img_stem]["scores"].append(valid_preds[:, 4])
                    aggregated_detections[img_stem]["labels"].append(valid_preds[:, 5])

    print("Applying global NMS and formatting submission...")
    results = []
    
    for img_stem, dets in aggregated_detections.items():
        if len(dets["boxes"]) == 0:
            results.append({"image_id": img_stem, "PredictionString": "No_parking 0.0001 0 0 1 1"})
            continue
            
        # Concatenate tensors for this image
        all_boxes = torch.cat(dets["boxes"], dim=0)
        all_scores = torch.cat(dets["scores"], dim=0)
        all_labels = torch.cat(dets["labels"], dim=0)
        
        # We modified global_nms to accept tensors
        final_boxes, final_scores, final_labels = global_nms(all_boxes, all_scores, all_labels, iou_threshold=iou_thresh)
        
        prediction_strings = []
        for i in range(len(final_boxes)):
            bx1, by1, bx2, by2 = final_boxes[i]
            score = final_scores[i]
            cls_id = int(final_labels[i])
            label_name = class_names[cls_id]
            
            pred_str = f"{label_name} {score:.4f} {int(bx1)} {int(by1)} {int(bx2)} {int(by2)}"
            prediction_strings.append(pred_str)
            
        if not prediction_strings:
             final_str = "No_parking 0.0001 0 0 1 1"
        else:
             final_str = " ".join(prediction_strings)
             
        results.append({
            "image_id": img_stem,
            "PredictionString": final_str
        })
        
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Patch-based submission saved to {output_csv}")


if __name__ == "__main__":
    seed_everything(42)
    
    # Path Configuration
    # Set this to the hash of the experiment you want to evaluate
    EXPERIMENT_HASH = "89906949f97db5b404463e90b7cd7767"
    
    ARTIFACT_DIR = os.path.join(road_sign_root, 'artifacts', 'patch_based', EXPERIMENT_HASH)
    TEST_DIR = os.path.join(road_sign_root, 'data', 'test', 'images')
    
    SUBMISSION_DIR = os.path.join(ARTIFACT_DIR, 'submission')
    OUTPUT_CSV = os.path.join(SUBMISSION_DIR, 'submission_patch_inference.csv')
    
    run_patch_inference(
        artifact_dir=ARTIFACT_DIR,
        test_images_dir=TEST_DIR,
        output_csv=OUTPUT_CSV,
        batch_size=8,
        conf_thresh=0.05,   # slightly lower confidence for recall, NMS will filter
        iou_thresh=0.4     # NMS threshold for aggregation
    )
