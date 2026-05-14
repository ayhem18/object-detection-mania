import numpy as np
from typing import Any, Dict, List, Tuple

def iou_distance(boxes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Computes 1 - IoU between boxes and centroids.
    Because we only care about aspect ratios and scales, we assume all boxes 
    and centroids are centered at the same origin (e.g., (0,0) or (0.5, 0.5)).
    
    boxes: (N, 2) array of (w, h)
    centroids: (K, 2) array of (w, h)
    Returns: (N, K) distance matrix
    """
    # Expand dimensions to broadcast into (N, K, 2)
    boxes_exp = np.expand_dims(boxes, axis=1)      # (N, 1, 2)
    centroids_exp = np.expand_dims(centroids, axis=0) # (1, K, 2)
    
    # Since boxes share the same center, their intersection's width and height 
    # is just the minimum of their respective widths and heights.
    inter_w = np.minimum(boxes_exp[..., 0], centroids_exp[..., 0])
    inter_h = np.minimum(boxes_exp[..., 1], centroids_exp[..., 1])
    
    # Valid intersections only (though w, h should always be > 0)
    inter_area = np.maximum(inter_w, 0) * np.maximum(inter_h, 0)
    
    box_area = boxes_exp[..., 0] * boxes_exp[..., 1]
    centroid_area = centroids_exp[..., 0] * centroids_exp[..., 1]
    
    union_area = box_area + centroid_area - inter_area
    
    # Compute IoU and Distance
    iou = inter_area / (union_area + 1e-8)
    return 1.0 - iou

def kmeans_plus_plus_init(boxes: np.ndarray, num_anchors: int, seed: int = 42) -> np.ndarray:
    """
    Initializes centroids using the KMeans++ strategy to ensure spread.
    """
    rng = np.random.default_rng(seed)
    N = boxes.shape[0]
    
    # 1. Randomly select first centroid
    centroids = [boxes[rng.choice(N)]]
    
    # 2. Select remaining centroids
    for _ in range(1, num_anchors):
        # Distances from each box to the closest existing centroid
        dist_matrix = iou_distance(boxes, np.array(centroids)) # (N, current_K)
        min_dists = np.min(dist_matrix, axis=1)
        
        # Probabilities proportional to D(x)^2
        probs = min_dists ** 2
        sum_probs = np.sum(probs)
        
        if sum_probs > 0:
            probs = probs / sum_probs
        else:
            probs = np.ones(N) / N
            
        next_idx = rng.choice(N, p=probs)
        centroids.append(boxes[next_idx])
        
    return np.array(centroids)

def generate_anchors(wh_list: List[Tuple[float, float]], num_anchors: int, max_iters: int = 100, seed: int = 42) -> np.ndarray:
    """
    Finds YOLOv2 anchors using KMeans++ and IoU distance.
    
    Args:
        wh_list: List of (w, h) tuples representing dataset bounding boxes.
        num_anchors: Number of anchor boxes to generate (K).
        max_iters: Maximum iterations for KMeans convergence.
        seed: Random seed for initialization.
        
    Returns:
        (num_anchors, 2) numpy array of anchor (w, h) sorted by area.
    """
    if not wh_list:
        raise ValueError("wh_list cannot be empty")
        
    boxes = np.array(wh_list, dtype=np.float32)
    if num_anchors > boxes.shape[0]:
        raise ValueError("num_anchors cannot be greater than the number of boxes in the dataset")
        
    # Initialize using KMeans++
    centroids = kmeans_plus_plus_init(boxes, num_anchors, seed)
    
    for iteration in range(max_iters):
        # 1. Assign points to nearest centroid
        dist_matrix = iou_distance(boxes, centroids) # (N, K)
        assignments = np.argmin(dist_matrix, axis=1) # (N,)
        
        new_centroids = np.zeros_like(centroids)
        has_changed = False
        
        # 2. Update centroids
        for k in range(num_anchors):
            assigned_boxes = boxes[assignments == k]
            if len(assigned_boxes) > 0:
                # The YOLOv2 paper updates centroids using the median or mean of width and height.
                # Mean is standard.
                new_c = np.mean(assigned_boxes, axis=0)
                if not np.allclose(centroids[k], new_c):
                    has_changed = True
                new_centroids[k] = new_c
            else:
                # Edge case: If a centroid loses all points, re-initialize it randomly to recover
                rng = np.random.default_rng(seed + iteration + k)
                new_centroids[k] = boxes[rng.choice(boxes.shape[0])]
                has_changed = True
                
        centroids = new_centroids
        if not has_changed:
            print(f"KMeans-IoU converged at iteration {iteration+1}")
            break
            
    # Sort centroids by area for consistency in output
    areas = centroids[:, 0] * centroids[:, 1]
    sorted_idx = np.argsort(areas)
    
    return centroids[sorted_idx]

def compute_avg_iou(boxes: np.ndarray, anchors: np.ndarray) -> float:
    """
    Computes the average IoU between each box and its closest anchor.
    This metric helps evaluate how well the anchors represent the dataset.
    """
    dist_matrix = iou_distance(boxes, anchors)
    # IoU = 1 - Distance
    max_ious = 1.0 - np.min(dist_matrix, axis=1)
    return float(np.mean(max_ious))

def analyze_anchor_fit(wh_list: List[Tuple[float, float]], anchors: np.ndarray) -> Dict[str, Any]:
    """
    Computes diagnostics to assess how well anchors represent the dataset.
    
    Returns:
        Dictionary containing:
        - avg_iou: Mean of the best IoU for each box.
        - best_ious: (N,) array of best IoU per box.
        - log_residuals: (N, 2) array of log(w/wa) and log(h/ha) for assigned anchors.
        - assignments: (N,) array indicating which anchor index each box was assigned to.
    """
    boxes = np.array(wh_list, dtype=np.float32)
    dist_matrix = iou_distance(boxes, anchors)
    
    # 1. Best IoU per box
    best_iou_indices = np.argmin(dist_matrix, axis=1)
    best_ious = 1.0 - np.min(dist_matrix, axis=1)
    
    # 2. Log Residuals: log(target / anchor)
    assigned_anchors = anchors[best_iou_indices]
    # Use epsilon to avoid log(0)
    log_residuals = np.log(boxes / (assigned_anchors + 1e-8))
    
    return {
        "avg_iou": float(np.mean(best_ious)),
        "best_ious": best_ious,
        "log_residuals": log_residuals,
        "assignments": best_iou_indices
    }

def plot_anchor_diagnostics(diagnostics: Dict[str, Any], save_path: str = None):
    """Visualizes the quality of anchors."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Best IoU Distribution
    ax1.hist(diagnostics["best_ious"], bins=30, color='skyblue', edgecolor='black')
    ax1.axvline(diagnostics["avg_iou"], color='red', linestyle='--', label=f'Mean: {diagnostics["avg_iou"]:.3f}')
    ax1.set_title("Distribution of Best IoU per Box")
    ax1.set_xlabel("IoU")
    ax1.legend()
    
    # Plot 2: Log Width Residuals
    ax2.hist(diagnostics["log_residuals"][:, 0], bins=30, color='salmon', alpha=0.7, label='Log Width')
    ax2.hist(diagnostics["log_residuals"][:, 1], bins=30, color='seagreen', alpha=0.5, label='Log Height')
    ax2.set_title("Log-Space Residuals: log(dim / anchor_dim)")
    ax2.set_xlabel("Residual Value (Target = 0)")
    ax2.legend()
    
    # Plot 3: Box Assignment Balance
    counts = np.bincount(diagnostics["assignments"])
    ax3.bar(range(len(counts)), counts, color='plum', edgecolor='black')
    ax3.set_title("Anchor Assignment Frequency")
    ax3.set_xlabel("Anchor Index")
    ax3.set_ylabel("Number of Boxes")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Diagnostic plots saved to {save_path}")
    else:
        plt.show()
