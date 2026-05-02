import numpy as np
from typing import Dict, Tuple, Any, Optional, List

# =========================================================================================
# Efficient Spatial DP and Updated Area Logic
# =========================================================================================

def _split_box(container: Tuple[float, float, float, float], 
               box: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
    """
    Splits the container into up to 4 non-overlapping rectangles after removing the area 
    occupied by 'box'.
    Container/Box format: (x_min, y_min, x_max, y_max)
    """
    X1, Y1, X2, Y2 = container
    x1, y1, x2, y2 = box
    
    free_boxes = []
    
    # Top box
    if y1 > Y1:
        free_boxes.append((X1, Y1, X2, y1))
    # Bottom box
    if y2 < Y2:
        free_boxes.append((X1, y2, X2, Y2))
    # Left box (within vertical range of placed box)
    if x1 > X1:
        free_boxes.append((X1, y1, x1, y2))
    # Right box (within vertical range of placed box)
    if x2 < X2:
        free_boxes.append((x2, y1, X2, y2))
        
    return free_boxes

def _generate_multi_objects(rng: np.random.Generator, N: int, num_classes: int, num_colors: int) -> Dict[int, Tuple]:
    """
    Generates N objects using a Rectangular Partitioning (Guillotine-style) approach.
    Starts with the full image and splits 'free space' as objects are placed.
    """
    # Format: (x_min, y_min, x_max, y_max)
    available_boxes = [(0.0, 0.0, 1.0, 1.0)]
    objects = {}
    
    for i in range(N):
        if not available_boxes:
            break
            
        # 1. Pick a free box (weighted by area to prefer larger spaces)
        areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in available_boxes]
        total_area = sum(areas)
        probs = [a / total_area for a in areas]
        
        idx = rng.choice(len(available_boxes), p=probs)
        container = available_boxes.pop(idx)
        cX1, cY1, cX2, cY2 = container
        cw, ch = cX2 - cX1, cY2 - cY1
        
        # 2. Determine target size for the new object
        # We want to occupy a significant but not total portion of the chosen free box
        # Scale factor between 0.3 and 0.7 of the container's dimensions
        sw = rng.uniform(0.3, 0.7) * cw
        sh = rng.uniform(0.3, 0.7) * ch
        
        cls_id = int(rng.integers(0, num_classes))
        color_id = int(rng.integers(0, num_colors))
        if cls_id == 1: # Circle constraint
            size = min(sw, sh)
            sw = sh = size
            
        # 3. Random placement within the container
        bx_min = rng.uniform(cX1, cX2 - sw)
        by_min = rng.uniform(cY1, cY2 - sh)
        bx_max, by_max = bx_min + sw, by_min + sh
        
        placed_box = (bx_min, by_min, bx_max, by_max)
        
        # 4. Partition remaining space and update available boxes
        new_free = _split_box(container, placed_box)
        # Filter out tiny boxes that are useless for detection (e.g. < 5% of image side)
        for nf in new_free:
            if (nf[2] - nf[0]) > 0.05 and (nf[3] - nf[1]) > 0.05:
                available_boxes.append(nf)
        
        # 5. Store in YOLO format (cls, color_id, cx, cy, h, w)
        cx = (bx_min + bx_max) / 2.0
        cy = (by_min + by_max) / 2.0
        objects[i] = (cls_id, color_id, float(cx), float(cy), float(sh), float(sw))
        
    return objects

def _generate_single_random_object(rng: np.random.Generator, num_classes: int, num_colors: int) -> Tuple[int, int, float, float, float, float]:
    cls_id = int(rng.integers(0, num_classes))
    color_id = int(rng.integers(0, num_colors))
    # Slightly more variety in size for the single object case
    w = rng.uniform(0.15, 0.45)
    h = rng.uniform(0.15, 0.45)
    
    if cls_id == 1:
        w = h = min(w, h)
        
    x = rng.uniform(w/2, 1.0 - w/2)
    y = rng.uniform(h/2, 1.0 - h/2)
    return cls_id, color_id, x, y, h, w

def generate_scene_parameters(seed: int, num_classes: int = 4, num_colors: int = 5, force_n: Optional[int] = None) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    
    noise_mean = round(float(rng.uniform(0.0, 10.0)), 2)
    noise_std = round(float(rng.uniform(5.0, 20.0)), 2)
    
    N = force_n if force_n is not None else int(rng.integers(0, 6))
    
    if N == 0:
        objects = {}
    elif N == 1:
        obj = _generate_single_random_object(rng, num_classes, num_colors)
        objects = {0: obj}
    else:
        objects = _generate_multi_objects(rng, N, num_classes, num_colors)
        
    return {
        'noise_mean': noise_mean,
        'noise_std': noise_std,
        'num_objects': len(objects),
        'objects': objects
    }

if __name__ == "__main__":
    print("Testing DP-based Scene Generation:")
    sample = generate_scene_parameters(seed=42)
    print(f"Objects generated: {sample['num_objects']}")
    for idx, obj in sample['objects'].items():
        print(f"  [{idx}] Cls: {obj[0]}, Area: {obj[3]*obj[4]:.3f}, Pos: ({obj[1]:.3f}, {obj[2]:.3f})")
