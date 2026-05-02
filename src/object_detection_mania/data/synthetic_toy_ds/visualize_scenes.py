import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from typing import Tuple, List, Dict, Any

# # Add project root to sys.path
# SCRIPT_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

from object_detection_mania.data.synthetic_toy_ds.scene_generation import generate_scene_parameters
from object_detection_mania.data.synthetic_toy_ds.shapes import draw_rectangle, draw_circle, draw_triangle, draw_diamond
from mypt.code_utils.pytorch_utils import seed_everything

# =========================================================================================
# Scene Visualization
# Orchestrates parameter generation and drawing into grid plots
# =========================================================================================

def get_artifacts_dir():
    """Finds the yolo_v2_artifacts/synthetic directory by climbing up."""
    current_dir = Path(__file__).resolve().parent
    while "artifacts" not in os.listdir(current_dir):
        current_dir = current_dir.parent
        if current_dir == current_dir.parent:
            # Fallback
            fallback = Path(__file__).resolve().parent.parent / "artifacts" / "yolo_v2_artifacts" / "synthetic"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
    
    path = current_dir / "artifacts" / "yolo_v2_artifacts" / "synthetic"
    path.mkdir(parents=True, exist_ok=True)
    return path

def render_scene(params: Dict[str, Any], seed: int, img_shape: Tuple[int, int] = (416, 416)) -> np.ndarray:
    """Renders the scene using the provided parameters and the precise shape module."""
    h, w = img_shape
    # Base background (30, 30, 30)
    img = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)
    
    # Add noise
    rng = np.random.default_rng(int(abs(hash((params['noise_mean'], params['noise_std']))) % (2**32 - 1)))
    noise = rng.normal(params['noise_mean'], params['noise_std'], (h, w, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for idx, (cls_id, color_id, x, y, obj_h, obj_w) in params['objects'].items():
        color = colors[color_id % len(colors)]
        
        if cls_id == 0:
            draw_rectangle(img, x, y, obj_w, obj_h, color)
        elif cls_id == 1:
            draw_circle(img, x, y, obj_w, obj_h, color)
        elif cls_id == 2:
            # Randomize triangle orientation
            base_at_bottom = (hash(str(seed) + str(idx)) % 2 == 0)
            draw_triangle(img, x, y, obj_w, obj_h, color, base_at_bottom=base_at_bottom)
        elif cls_id == 3:
            draw_diamond(img, x, y, obj_w, obj_h, color)
            
    return img

def visualize_cases(num_samples_per_case: int = 10, img_shape: Tuple[int, int] = (512, 512)):
    save_dir = get_artifacts_dir()
    seed_everything(42)
    
    for N in range(6):
        print(f"Generating grid for N={N} objects...")
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i in range(num_samples_per_case):
            seed = 1000 * N + i
            params = generate_scene_parameters(seed=seed, force_n=N)
            img = render_scene(params, seed, img_shape)
            
            axes[i].imshow(img)
            
            # Overlay Bounding Boxes for verification
            for idx, (cls_id, color_id, x, y, obj_h, obj_w) in params['objects'].items():
                xmin = (x - obj_w/2) * img_shape[1]
                ymin = (y - obj_h/2) * img_shape[0]
                rect = patches.Rectangle((xmin, ymin), obj_w * img_shape[1], obj_h * img_shape[0], 
                                         linewidth=1, edgecolor='white', facecolor='none', alpha=0.8)
                axes[i].add_patch(rect)
                
            axes[i].set_title(f"Sample {i+1} (N={params['num_objects']})")
            axes[i].axis('off')
            
        plt.tight_layout()
        save_path = save_dir / f"grid_n_{N}.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"  Saved to {save_path}")

if __name__ == "__main__":
    visualize_cases()
