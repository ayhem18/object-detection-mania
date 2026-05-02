import cv2
import numpy as np
from typing import Tuple

# =========================================================================================
# Precise Shape Generators
# Focus: Mathematical alignment between the bounding box and the drawn pixels
# =========================================================================================

def _get_absolute_coords(img_shape: Tuple[int, int, ...], x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int]:
    """Helper to convert normalized [cx, cy, w, h] to absolute [x_min, y_min, x_max, y_max]."""
    img_h, img_w = img_shape[:2]
    
    x_min = int(round((x - w / 2) * img_w))
    y_min = int(round((y - h / 2) * img_h))
    x_max = int(round((x + w / 2) * img_w))
    y_max = int(round((y + h / 2) * img_h))
    
    return x_min, y_min, x_max, y_max

def draw_rectangle(img: np.ndarray, x: float, y: float, w: float, h: float, color: Tuple[int, int, int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = _get_absolute_coords(img.shape, x, y, w, h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness=-1)
    return img

def draw_circle(img: np.ndarray, x: float, y: float, w: float, h: float, color: Tuple[int, int, int]) -> np.ndarray:
    # Use a small epsilon for float comparison
    if abs(w - h) > 1e-5:
        raise ValueError(f"Circle bounding box must have equal width and height. Got w={w:.4f}, h={h:.4f}")
    
    img_h, img_w = img.shape[:2]
    cx = int(round(x * img_w))
    cy = int(round(y * img_h))
    
    # Radius based on pixel width
    radius = int(round((w * img_w) / 2))
    
    cv2.circle(img, (cx, cy), radius, color, thickness=-1)
    return img

def draw_triangle(img: np.ndarray, x: float, y: float, w: float, h: float, color: Tuple[int, int, int], base_at_bottom: bool = True) -> np.ndarray:
    x_min, y_min, x_max, y_max = _get_absolute_coords(img.shape, x, y, w, h)
    x_average = int(round((x_min + x_max) / 2))
    
    if base_at_bottom:
        # Base at y_max (bottom of the box), point at y_min (top of the box)
        pts = np.array([[x_min, y_max], [x_max, y_max], [x_average, y_min]], np.int32)
    else:
        # Base at y_min (top of the box), point at y_max (bottom of the box)
        pts = np.array([[x_min, y_min], [x_max, y_min], [x_average, y_max]], np.int32)
        
    cv2.fillPoly(img, [pts], color)
    return img

def draw_diamond(img: np.ndarray, x: float, y: float, w: float, h: float, color: Tuple[int, int, int]) -> np.ndarray:
    """
    A simpler replacement for the Star shape.
    It guarantees that the extreme vertices perfectly touch the bounding box edges.
    """
    x_min, y_min, x_max, y_max = _get_absolute_coords(img.shape, x, y, w, h)
    x_average = int(round((x_min + x_max) / 2))
    y_average = int(round((y_min + y_max) / 2))
    
    pts = np.array([
        [x_average, y_min], # Top center
        [x_max, y_average], # Right center
        [x_average, y_max], # Bottom center
        [x_min, y_average]  # Left center
    ], np.int32)
    
    cv2.fillPoly(img, [pts], color)
    return img
