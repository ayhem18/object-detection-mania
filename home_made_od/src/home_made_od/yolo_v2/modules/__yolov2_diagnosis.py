import os
import cv2
import torch
import concurrent.futures

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision.transforms import v2
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader

from home_made_od.yolo_v2.modules.yolov2_model import YoloV2


class DiagnosisDataset(Dataset):
    """
    Dataset for fast inference during error analysis.
    Yields transformed tensors and the dataset index.
    Pre-computes and stores original image sizes.
    """
    def __init__(self, image_paths: List[str], target_size: Tuple[int, int], transform: v2.Compose):
        self.image_paths = image_paths
        self.target_size = target_size
        self.transform = transform
        
        self.original_sizes = {}
        print("Pre-computing original image sizes...")
        for i, p in enumerate(tqdm(self.image_paths, desc="Reading headers")):
            try:
                # PIL reads only the header, which is much faster than loading the entire image
                with Image.open(p) as img:
                    self.original_sizes[i] = img.size # (width, height)
            except Exception:
                self.original_sizes[i] = (0, 0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        img_bgr = cv2.imread(img_path)
        
        if img_bgr is None:
            return torch.zeros((3, self.target_size[1], self.target_size[0])), idx
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb)
        
        return input_tensor, idx


def get_raw_predictions(model: YoloV2, 
                        images: torch.Tensor, 
                        anchors: List[List[float] | Tuple[float, float]],
                        return_all_probs: bool = False) -> Tuple[torch.Tensor, ...]:
    """
    Extracts raw bounding boxes, scores, and class IDs from a YOLOv2 model without NMS or confidence filtering.
    """
    model.eval()
    with torch.no_grad():
        raw_output = model(images)
    
    img_h, img_w = images.shape[2], images.shape[3]
    return model.decode_predictions(raw_output, anchors, (img_h, img_w), return_all_probs)


def _draw_and_save_single_image(args):
    """
    Worker function to process and save a single diagnostic image in a separate process.
    """
    img_path, orig_w, orig_h, boxes, scores, cls_ids, target_size, conf_threshold, class_names, output_dir = args
    
    if orig_w == 0 or orig_h == 0:
        return False
        
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return False
        
    # Filter out very low confidence to avoid drawing thousands of boxes
    mask = scores > conf_threshold
    f_boxes = boxes[mask]
    f_scores = scores[mask]
    f_cls = cls_ids[mask]
    
    scale_y = orig_w / target_size[0]
    scale_x = orig_h / target_size[1]
    
    for j in range(len(f_boxes)):
        x1, y1, x2, y2 = f_boxes[j]
        score = f_scores[j]
        c_id = int(f_cls[j])
        
        ox1, oy1 = int(x1 * scale_x), int(y1 * scale_y)
        ox2, oy2 = int(x2 * scale_x), int(y2 * scale_y)
        
        label = f"{class_names[c_id]} {score:.2f}"
        
        # Thin box to see overlaps
        cv2.rectangle(img_bgr, (ox1, oy1), (ox2, oy2), (0, 0, 255), 1)
        cv2.putText(img_bgr, label, (ox1, max(oy1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    # # Resize for viewing
    # h, w = img_bgr.shape[:2]
    # scale = 1000 / max(w, h)
    # small_img = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    
    out_path = os.path.join(output_dir, f"raw_{Path(img_path).name}")
    cv2.imwrite(out_path, img_bgr)
    return True


def visualize_raw_predictions(image_paths: List[str],
                              model: YoloV2,
                              anchors: List[List[float] | Tuple[float, float]],
                              output_dir: str,
                              transform_stats: dict,
                              class_mapping: Dict[str, str],
                              target_size: Tuple[int, int] = (512, 512),
                              batch_size: int = 8,
                              conf_threshold: float = 0.05,
                              max_images: Optional[int] = 20):
    """
    Runs batched inference to get RAW predictions and plots them on the original images.
    Separates inference and rendering into two distinct phases for performance.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    
    preprocess = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(target_size),
        v2.Normalize(mean=transform_stats['mean'], std=transform_stats['std'])
    ])
    
    if max_images is not None:
        image_paths = image_paths[:max_images]
        
    dataset = DiagnosisDataset(image_paths, target_size, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    class_names = [class_mapping.get(str(i), f"Class_{i}") for i in range(len(class_mapping))]
    
    print(f"Running batched inference for {len(image_paths)} images...")
    
    all_predictions = {}
    
    # Phase 1: Batched Inference
    for batch_tensors, batch_idxs in tqdm(dataloader, desc="Inference"):
        batch_tensors = batch_tensors.to(device)
        boxes, scores, cls_ids = get_raw_predictions(model, batch_tensors, anchors)
        
        # Offload from GPU immediately
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        cls_ids = cls_ids.cpu().numpy()
        batch_idxs = batch_idxs.numpy()
        
        for i in range(len(batch_idxs)):
            idx = batch_idxs[i]
            all_predictions[idx] = (boxes[i], scores[i], cls_ids[i])
            
    # Phase 2: Multiprocessed Visualization
    print(f"Inference complete. Visualizing and saving to {output_dir}...")
    
    tasks = []
    for idx, (b, s, c) in all_predictions.items():
        img_path = dataset.image_paths[idx]
        orig_w, orig_h = dataset.original_sizes[idx]
        tasks.append((
            img_path, orig_w, orig_h, b, s, c, target_size, conf_threshold, class_names, output_dir
        ))
        
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(_draw_and_save_single_image, tasks), total=len(tasks), desc="Visualizing"))
        
    print("Visualization complete.")

import torchvision.ops as ops

def analyze_gt_matches(model: YoloV2, 
                       images: torch.Tensor, 
                       gt_boxes_list: List[torch.Tensor], 
                       anchors: List[List[float] | Tuple[float, float]], 
                       orig_sizes: List[Tuple[int, int]],
                       visualize: bool = False,
                       output_dir: Optional[str] = None,
                       image_paths: Optional[List[str]] = None,
                       class_mapping: Optional[Dict[str, str]] = None) -> List[Dict]:
    """
    Analyzes the highest IoU predictions for given ground truth bounding boxes.
    
    Args:
        model: Trained YoloV2 model.
        images: Batch of images (B, C, H, W).
        gt_boxes_list: List of length B. Each element is a tensor of shape (N, 5) 
                       representing [cls, x1, y1, x2, y2] in ABSOLUTE original image coordinates.
        anchors: List of anchors.
        orig_sizes: List of original sizes corresponding to the specific user convention 
                    where orig_sizes[0] (w) is mapped to height/y-axis and 
                    orig_sizes[1] (h) is mapped to width/x-axis.
        visualize: If True, saves images showing the GT and best prediction.
        output_dir: Directory to save visualizations.
        image_paths: Paths to original images (required if visualize=True).
        class_mapping: Mapping for class names (required if visualize=True).
    
    Returns:
        List of diagnostic dictionaries for each ground truth box.
    """
    if visualize and (output_dir is None or image_paths is None or class_mapping is None):
        raise ValueError("visualize=True requires output_dir, image_paths, and class_mapping.")

    out = get_raw_predictions(model, images, anchors, return_all_probs=True)
    boxes, scores, cls_ids, obj_probs, max_cls_probs = out
    
    diagnostics = []
    B = images.size(0)
    target_size = (images.size(2), images.size(3)) # H, W
    
    class_names = []
    if class_mapping:
        class_names = [class_mapping.get(str(i), f"Class_{i}") for i in range(len(class_mapping))]
    
    for b in range(B):
        img_boxes = boxes[b] # (num_preds, 4) in target_size coords
        img_scores = scores[b]
        img_obj_probs = obj_probs[b]
        img_cls_probs = max_cls_probs[b]
        img_cls_ids = cls_ids[b]
        
        orig_w, orig_h = orig_sizes[b] 
        
        # Scaling factors
        scale_y = orig_w / target_size[0] if target_size[0] > 0 else 0
        scale_x = orig_h / target_size[1] if target_size[1] > 0 else 0
        
        scaled_boxes = img_boxes.clone()
        scaled_boxes[:, 0] *= scale_x # x1
        scaled_boxes[:, 1] *= scale_y # y1
        scaled_boxes[:, 2] *= scale_x # x2
        scaled_boxes[:, 3] *= scale_y # y2
        
        gt_boxes = gt_boxes_list[b]
        if len(gt_boxes) == 0:
            continue
            
        gt_coords = gt_boxes[:, 1:5]
        gt_classes = gt_boxes[:, 0]
        
        # Calculate IoU between all GT boxes and all predicted boxes
        ious = ops.box_iou(gt_coords, scaled_boxes) # (N, num_preds)
        
        if visualize:
            img_path = image_paths[b]
            img_vis = cv2.imread(img_path)
            
            # --- Visualize IoU Matrix ---
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(10, 8))
            sns.heatmap(ious.cpu().numpy(), annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title(f"IoU Matrix - {Path(img_path).name}")
            plt.xlabel("Predictions")
            plt.ylabel("Ground Truths")
            plt.tight_layout()
            iou_matrix_path = os.path.join(output_dir, f"iou_matrix_{Path(img_path).stem}.png")
            plt.savefig(iou_matrix_path)
            plt.close()
        
        for i in range(len(gt_boxes)):
            gt_box = gt_coords[i]
            gt_cls = int(gt_classes[i].item())
            
            # Bbox size and area in original image
            w_gt = (gt_box[2] - gt_box[0]).item()
            h_gt = (gt_box[3] - gt_box[1]).item()
            gt_area = w_gt * h_gt
            
            # Resized dimensions
            gt_resized_w = w_gt / scale_x if scale_x > 0 else 0
            gt_resized_h = h_gt / scale_y if scale_y > 0 else 0
            
            # Feature map area and min dim
            stride = 32
            gt_area_feature_map = (gt_resized_w / stride) * (gt_resized_h / stride)
            min_dim_feature_map = min(gt_resized_w / stride, gt_resized_h / stride)
            
            # Total image area
            img_area = orig_w * orig_h 
            area_ratio = gt_area / img_area if img_area > 0 else 0
            
            # Highest IoU match
            max_iou_val, max_iou_idx = torch.max(ious[i], dim=0)
            max_iou_val = max_iou_val.item()
            idx = max_iou_idx.item()
            
            diag = {
                "image_path": image_paths[b] if image_paths else None,
                "gt_class": gt_cls,
                "gt_width": w_gt,
                "gt_height": h_gt,
                "gt_area": gt_area,
                "gt_width_resized": gt_resized_w,
                "gt_height_resized": gt_resized_h,
                "gt_area_feature_map": gt_area_feature_map,
                "min_dim_feature_map": min_dim_feature_map,
                "area_ratio": area_ratio,
                "best_iou": max_iou_val,
                "pred_score": img_scores[idx].item(),
                "pred_obj_prob": img_obj_probs[idx].item(),
                "pred_cls_prob": img_cls_probs[idx].item(),
                "pred_class": int(img_cls_ids[idx].item()),
                "correct_class_predicted": int(img_cls_ids[idx].item()) == gt_cls
            }
            diagnostics.append(diag)

            if visualize:
                # Draw GT
                gx1, gy1, gx2, gy2 = map(int, gt_box)
                cv2.rectangle(img_vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 3) # Green for GT
                
                # Draw Best Pred
                px1, py1, px2, py2 = map(int, scaled_boxes[idx])
                cv2.rectangle(img_vis, (px1, py1), (px2, py2), (0, 0, 255), 2) # Red for Best Match
                
                label = f"GT:{gt_cls} | Pred:{diag['pred_class']} S:{diag['pred_score']:.2f}"
                cv2.putText(img_vis, label, (gx1, max(gy1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if visualize:
            out_name = f"match_{Path(image_paths[b]).name}"
            cv2.imwrite(os.path.join(output_dir, out_name), img_vis)
            
    return diagnostics


def plot_diagnostic_results(csv_path: str, output_dir: str):
    """
    Reads the diagnostic CSV and generates analytical plots.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print("No data to plot.")
        return
        
    # 1. Feature Map Area vs Objectness Probability
    plt.figure(figsize=(10, 6))
    plt.scatter(df['gt_area_feature_map'], df['pred_obj_prob'], alpha=0.5, c='blue')
    plt.axvline(x=1.0, color='red', linestyle='--', label='1 Grid Cell Area')
    plt.title('Feature Map Area vs Objectness Probability')
    plt.xlabel('GT Area in Feature Map (cells)')
    plt.ylabel('P(Object)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fm_area_vs_obj_prob.png'))
    plt.close()

    # 2. Feature Map Area vs Class Probability (Conditioned on correct/incorrect)
    plt.figure(figsize=(10, 6))
    correct = df[df['correct_class_predicted'] == True]
    incorrect = df[df['correct_class_predicted'] == False]
    
    plt.scatter(correct['gt_area_feature_map'], correct['pred_cls_prob'], alpha=0.5, c='green', label='Correct Class')
    plt.scatter(incorrect['gt_area_feature_map'], incorrect['pred_cls_prob'], alpha=0.5, c='red', label='Incorrect Class')
    plt.axvline(x=1.0, color='black', linestyle='--')
    plt.title('Feature Map Area vs Class Probability')
    plt.xlabel('GT Area in Feature Map (cells)')
    plt.ylabel('P(Class | Object)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fm_area_vs_cls_prob.png'))
    plt.close()
    
    # 3. Feature Map Area vs Best IoU
    plt.figure(figsize=(10, 6))
    plt.scatter(df['gt_area_feature_map'], df['best_iou'], alpha=0.5, c='purple')
    plt.axvline(x=1.0, color='red', linestyle='--')
    plt.title('Feature Map Area vs Best IoU')
    plt.xlabel('GT Area in Feature Map (cells)')
    plt.ylabel('Max IoU with any Prediction')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fm_area_vs_iou.png'))
    plt.close()
    
    print(f"Saved diagnostic plots to {output_dir}")
