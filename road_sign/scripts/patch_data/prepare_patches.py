import os
import cv2
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two boxes in [x1, y1, x2, y2] format.
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / float(area1 + area2 - intersection_area)
    return iou

def get_patch_from_object(img_w: int, img_h: int, obj_box: np.ndarray, patch_size: int, position: str) -> Tuple[int, int, int, int]:
    """
    Generates patch coordinates [x1, y1, x2, y2] based on object position in patch.
    """
    cx, cy, _, _ = obj_box
    P = patch_size
    
    offsets = {
        'center': (0.5, 0.5),
        'top_left': (0.2, 0.2),
        'top_right': (0.8, 0.2),
        'bottom_left': (0.2, 0.8),
        'bottom_right': (0.8, 0.8)
    }
    
    rel_x, rel_y = offsets.get(position, (0.5, 0.5))
    x1 = int(cx - rel_x * P)
    y1 = int(cy - rel_y * P)
    x1 = max(0, min(x1, img_w - P))
    y1 = max(0, min(y1, img_h - P))
    
    return x1, y1, x1 + P, y1 + P

def select_patch_size(obj_w: float, obj_h: float, config: dict = None) -> int:
    """
    Selects the optimal patch size for an object based on its dimensions and area.
    Uses scales and min_scale_ratio_threshold from config.
    """
    config = config or {}
    scales = sorted(config.get("scales", [512, 1024, 2048]))
    ratio_thresh = config.get("min_scale_ratio_threshold", 0.5)
    
    max_dim = max(obj_w, obj_h)
    selected_scale = scales[-1]
    
    for scale in scales:
        if max_dim <= scale:
            selected_scale = scale
            break
            
    obj_area = obj_w * obj_h
    if (obj_area / (selected_scale * selected_scale)) >= ratio_thresh:
        idx = scales.index(selected_scale)
        if idx + 1 < len(scales):
            return scales[idx + 1]
            
    return selected_scale

def generate_positive_patches(img_shape: Tuple[int, int], gt_boxes: np.ndarray, config: dict = None) -> List[Tuple[int, int, int, int]]:
    """
    Returns a list of patch coordinates [x1, y1, x2, y2] for all objects.
    gt_boxes: (N, 5) -> [cls, cx, cy, w, h] in absolute pixels.
    """
    config = config or {}
    h, w = img_shape[:2]
    patches = []
    
    for box in gt_boxes:
        bw, bh = box[3], box[4]
        patch_size = select_patch_size(bw, bh, config)
        
        obj_area = bw * bh
        patch_area = patch_size * patch_size
        ratio = obj_area / patch_area
        
        if ratio < 0.5:
            positions = ['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right']
        elif ratio < 0.75:
            positions = ['center', random.choice(['top_left', 'top_right', 'bottom_left', 'bottom_right'])]
        else:
            positions = ['center']
            
        for pos in positions:
            patch = get_patch_from_object(w, h, box[1:], patch_size, pos)
            patches.append(patch)
            
    return patches

def generate_negative_patches(img: np.ndarray, gt_boxes: np.ndarray, config: dict = None) -> List[Tuple[int, int, int, int]]:
    """
    Uses Selective Search to find background patches with zero IoU with GT.
    """
    config = config or {}
    h, w = img.shape[:2]
    patch_sizes = config.get("scales", [512, 1024, 2048])
    K2 = config.get("K2", 2)
    
    scale = 1000.0 / max(w, h)
    small_img = cv2.resize(img, (int(w * scale), int(h * scale)))
    
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(small_img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process() 
    
    candidate_patches = []
    for (rx, ry, rw, rh) in rects:
        patch_size = random.choice(patch_sizes)
        cx = (rx + rw/2) / scale
        cy = (ry + rh/2) / scale
        
        x1 = int(cx - patch_size/2)
        y1 = int(cy - patch_size/2)
        x2, y2 = x1 + patch_size, y1 + patch_size
        
        if x1 < 0: x1, x2 = 0, patch_size
        if y1 < 0: y1, y2 = 0, patch_size
        if x2 > w: x2, x1 = w, w - patch_size
        if y2 > h: y2, y1 = h, h - patch_size
        
        if x2 <= w and y2 <= h:
            candidate_patches.append(np.array([x1, y1, x2, y2]))

    gt_xyxy = []
    for box in gt_boxes:
        bcx, bcy, bw, bh = box[1:]
        gt_xyxy.append([bcx - bw/2, bcy - bh/2, bcx + bw/2, bcy + bh/2])
    gt_xyxy = np.array(gt_xyxy)

    valid_negatives = []
    for patch in candidate_patches:
        is_background = True
        for gt in gt_xyxy:
            if calculate_iou(patch, gt) > 0:
                is_background = False
                break
        
        if is_background:
            valid_negatives.append(tuple(patch))
            if len(valid_negatives) >= K2 * 5: 
                break
                
    return valid_negatives[:K2]

def update_labels_for_patch(patch_coords: Tuple[int, int, int, int], gt_boxes: np.ndarray, config: dict = None) -> np.ndarray:
    """
    Calculates new normalized YOLO labels for objects inside the patch.
    Applies visibility thresholds to filter severely cropped objects.
    """
    config = config or {}
    px1, py1, px2, py2 = patch_coords
    pw, ph = px2 - px1, py2 - py1
    min_visibility = config.get("min_visibility_threshold", 0.4)
    
    new_labels = []
    for box in gt_boxes:
        cls, cx, cy, bw, bh = box
        
        # Check if original center is inside the patch
        if not (px1 <= cx <= px2 and py1 <= cy <= py2):
            continue
            
        orig_area = bw * bh
        
        x1_obj, y1_obj = cx - bw/2, cy - bh/2
        x2_obj, y2_obj = cx + bw/2, cy + bh/2
        
        x1_clip = max(x1_obj, px1)
        y1_clip = max(y1_obj, py1)
        x2_clip = min(x2_obj, px2)
        y2_clip = min(y2_obj, py2)
        
        new_bw = x2_clip - x1_clip
        new_bh = y2_clip - y1_clip
        new_area = new_bw * new_bh
        
        # Filter objects that lost too much of their original area
        if orig_area > 0 and (new_area / orig_area) < min_visibility:
            continue
            
        final_cx = x1_clip + new_bw/2 - px1
        final_cy = y1_clip + new_bh/2 - py1
        
        new_labels.append([
            cls,
            final_cx / pw,
            final_cy / ph,
            new_bw / pw,
            new_bh / ph
        ])
        
    return np.array(new_labels) if new_labels else np.zeros((0, 5))

def orchestrate_image_patches(image_path: str, label_path: str, output_dir: str, K2: int):
    config = {
        "scales": [512, 1024, 2048],
        "min_scale_ratio_threshold": 0.5,
        "min_visibility_threshold": 0.4,
        "K2": K2
    }
    
    img = cv2.imread(image_path)
    if img is None: return
    h, w = img.shape[:2]
    
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = [float(x) for x in line.split()]
                if len(parts) >= 5:
                    cls, ncx, ncy, nw, nh = parts[:5]
                    gt_boxes.append([
                        cls, ncx * w, ncy * h, nw * w, nh * h
                    ])
    gt_boxes = np.array(gt_boxes)

    pos_coords = generate_positive_patches((h, w), gt_boxes, config)
    neg_coords = generate_negative_patches(img, gt_boxes, config)
    
    all_patches = [('pos', c) for c in pos_coords] + [('neg', c) for c in neg_coords]
    
    base_name = Path(image_path).stem
    img_out_dir = Path(output_dir) / base_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    
    full_img_vis = img.copy()
    for type_prefix, (x1, y1, x2, y2) in all_patches:
        color = (0, 255, 0) if type_prefix == 'pos' else (0, 0, 255)
        cv2.rectangle(full_img_vis, (x1, y1), (x2, y2), color, 8)
    
    for box in gt_boxes:
        cls, cx, cy, bw, bh = box
        bx1, by1 = int(cx - bw/2), int(cy - bh/2)
        bx2, by2 = int(cx + bw/2), int(cy + bh/2)
        cv2.rectangle(full_img_vis, (bx1, by1), (bx2, by2), (255, 0, 0), 4)

    scale = 1000 / max(w, h)
    small_vis = cv2.resize(full_img_vis, (int(w * scale), int(h * scale)))
    cv2.imwrite(str(img_out_dir / "full_image_with_patches.jpg"), small_vis)

    for i, (type_prefix, coords) in enumerate(all_patches):
        x1, y1, x2, y2 = coords
        patch_img = img[y1:y2, x1:x2].copy()
        pw, ph = patch_img.shape[1], patch_img.shape[0]
        
        patch_labels = update_labels_for_patch(coords, gt_boxes, config)
        
        for lbl in patch_labels:
            cls_id, cx_norm, cy_norm, w_norm, h_norm = lbl
            
            pcx, pcy = int(cx_norm * pw), int(cy_norm * ph)
            pbw, pbh = int(w_norm * pw), int(h_norm * ph)
            
            px1, py1 = int(pcx - pbw/2), int(pcy - pbh/2)
            px2, py2 = int(pcx + pbw/2), int(pcy + pbh/2)
            
            cv2.rectangle(patch_img, (px1, py1), (px2, py2), (255, 255, 0), 4)
            cv2.putText(patch_img, f"cls:{int(cls_id)}", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        patch_filename = f"{type_prefix}_patch_{i}.jpg"
        cv2.imwrite(str(img_out_dir / patch_filename), patch_img)

def run_visualization_sample(data_dir: str, output_dir: str, K2: int = 2, max_images: int = 5):
    img_dir = Path(data_dir) / 'train' / 'images'
    lbl_dir = Path(data_dir) / 'labels' / 'annotations'
    
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"Data directories not found in {data_dir}")
        return

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    
    count = 0
    for img_path in images:
        if count >= max_images:
            break
            
        label_path = lbl_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
            
        print(f"Processing {img_path.name}...")
        orchestrate_image_patches(str(img_path), str(label_path), output_dir, K2)
        count += 1

if __name__ == "__main__":
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while 'road_sign' not in os.listdir(current_dir):
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir: break
        current_dir = parent_dir
    road_sign_root = os.path.join(current_dir, 'road_sign')
    
    data_dir = os.path.join(road_sign_root, 'data')
    out_dir = os.path.join(road_sign_root, 'artifacts', 'patch_visualization')
    
    run_visualization_sample(data_dir, out_dir, K2=2, max_images=3)
