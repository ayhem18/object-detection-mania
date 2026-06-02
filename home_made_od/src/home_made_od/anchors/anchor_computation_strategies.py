"""
Anchor optimization strategies for FPN-based detectors.

Methods (see ``compute_optimized_anchors``):
- ``area_based`` — assign GT to level by log-area; K2 aspect ratios via k-means on (y, x).
- ``dimension_based`` — assign to coarsest level with ``stride * coefficient <= min(h, w)``;
  cluster K1 base sizes (sqrt area) and K2 aspect ratios (h/w) independently per level.
- ``default`` — fixed multi-level anchor preset (no dataset clustering).

This module performs in-memory computation only. RetinaNet manifest assembly and
persistence live in ``retinanet_anchors``.
"""

from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from mypt.code_utils.pytorch_utils import seed_everything

# from dl_lib.etalon_object_detection.modules.ds_utils import WeldingDetectionDataset

ASSIGNMENT_METHODS = ("area_based", "dimension_based", "default")

# Canonical FPN level order (generic; not detector-specific).
FPN_LEVEL_ORDER: Tuple[str, ...] = ("P2", "P3", "P4", "P5", "P6", "P7")

logger = logging.getLogger(__name__)

# torchvision.models.detection.retinanet._default_anchorgen()
TORCHVISION_DEFAULT_LEVEL_BASE_SIZES = [32, 64, 128, 256, 512]
TORCHVISION_DEFAULT_ASPECT_RATIOS = (0.5, 1.0, 2.0)
TORCHVISION_DEFAULT_LEVELS = ("P3", "P4", "P5", "P6", "P7")

METHOD_REQUIRED_PARAMETERS: Dict[str, Tuple[str, ...]] = {
    "area_based": (
        "fpn_coefficient",
        "num_aspect_ratios",
        "num_base_sizes",
        "min_etalons_per_level",
        "scales",
        "seed",
        "max_iters",
    ),
    "dimension_based": (
        "fpn_coefficient",
        "num_aspect_ratios",
        "num_base_sizes",
        "min_etalons_per_level",
        "scales",
        "seed",
        "fallback_level",
        "max_iters",
    ),
    "default": (
        "fpn_coefficient",
        "seed",
    ),
}


@dataclass(frozen=True)
class AnchorOptimizationResult:
    """Raw anchor clustering output before detector-specific manifest finalization."""

    method: str
    method_parameters: Dict[str, Any]
    config_hash: str
    dataset_hash: str
    img_size: Tuple[int, int]
    fpn_specs: List[Dict[str, Any]]
    fpn_coefficient: int
    used_fpn_levels: List[str]
    level_aspect_ratios: Dict[str, List[float]]
    level_base_sizes: Dict[str, List[float]]
    enriched_samples: List[Dict[str, Any]]
    seed: int
    scales: List[float]


def validate_method_parameters(method: str, method_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``method_parameters`` after checking required keys for ``method``."""
    if method not in ASSIGNMENT_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Choose from {ASSIGNMENT_METHODS}."
        )
    if not isinstance(method_parameters, dict) or not method_parameters:
        raise ValueError("method_parameters must be a non-empty dict.")

    required = METHOD_REQUIRED_PARAMETERS[method]
    missing = [key for key in required if key not in method_parameters]
    if missing:
        raise ValueError(
            f"method_parameters for {method!r} missing required keys: {missing}. "
            f"Required: {list(required)}."
        )
    return method_parameters


def iou_distance(boxes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """1 - IoU between (x_dim, y_dim) boxes and centroids, shared-center assumption."""
    boxes_exp = np.expand_dims(boxes, axis=1)
    centroids_exp = np.expand_dims(centroids, axis=0)

    inter_x = np.minimum(boxes_exp[..., 0], centroids_exp[..., 0])
    inter_y = np.minimum(boxes_exp[..., 1], centroids_exp[..., 1])
    inter_area = np.maximum(inter_x, 0) * np.maximum(inter_y, 0)

    box_area = boxes_exp[..., 0] * boxes_exp[..., 1]
    centroid_area = centroids_exp[..., 0] * centroids_exp[..., 1]
    union_area = box_area + centroid_area - inter_area

    iou = inter_area / (union_area + 1e-8)
    return 1.0 - iou


def kmeans_plus_plus_init(boxes: np.ndarray, num_anchors: int, seed: int) -> np.ndarray:
    """KMeans++ initialization for 2D (y_dim, x_dim) boxes."""
    rng = np.random.default_rng(seed)
    n = boxes.shape[0]
    centroids = [boxes[rng.choice(n)]]

    for _ in range(1, num_anchors):
        dist_matrix = iou_distance(boxes, np.array(centroids))
        min_dists = np.min(dist_matrix, axis=1)
        probs = min_dists ** 2
        sum_probs = np.sum(probs)
        probs = probs / sum_probs if sum_probs > 0 else np.ones(n) / n
        centroids.append(boxes[rng.choice(n, p=probs)])

    return np.array(centroids)


def generate_anchors(
    y_then_x_dims: List[Tuple[float, float]],
    num_anchors: int,
    max_iters: int,
    seed: int,
) -> np.ndarray:
    """KMeans++ on (y_dim, x_dim) with IoU distance; returns sorted (y, x) centroids."""
    seed_everything(seed)

    if not y_then_x_dims:
        raise ValueError("y_then_x_dims cannot be empty")

    y_then_x_dims = sorted(y_then_x_dims)
    boxes = np.array(y_then_x_dims, dtype=np.float32)
    if num_anchors > boxes.shape[0]:
        raise ValueError("num_anchors cannot be greater than the number of boxes")

    centroids = kmeans_plus_plus_init(boxes, num_anchors, seed)

    for iteration in range(max_iters):
        dist_matrix = iou_distance(boxes, centroids)
        assignments = np.argmin(dist_matrix, axis=1)

        new_centroids = np.zeros_like(centroids)
        has_changed = False

        for k in range(num_anchors):
            assigned_boxes = boxes[assignments == k]
            if len(assigned_boxes) > 0:
                new_c = np.mean(assigned_boxes, axis=0)
                if not np.allclose(centroids[k], new_c):
                    has_changed = True
                new_centroids[k] = new_c
            else:
                rng = np.random.default_rng(seed + iteration + k)
                new_centroids[k] = boxes[rng.choice(boxes.shape[0])]
                has_changed = True

        centroids = new_centroids
        if not has_changed:
            break   

    areas = centroids[:, 0] * centroids[:, 1]
    return centroids[np.argsort(areas)]


def cluster_1d_values(
    values: List[float],
    k: int,
    seed: int,
    max_iters: int,
) -> np.ndarray:
    """Simple 1D k-means for independent clustering (base sizes or aspect ratios)."""
    if not values:
        raise ValueError("values cannot be empty")
    if k > len(values):
        raise ValueError("k cannot be greater than the number of values")

    rng = np.random.default_rng(seed)
    data = np.array(sorted(values), dtype=np.float64).reshape(-1, 1)

    centroids = np.array([[data[rng.choice(len(data))][0]]])
    for _ in range(1, k):
        dists = np.min(np.abs(data - centroids.T), axis=1)
        probs = dists ** 2
        total = probs.sum()
        probs = probs / total if total > 0 else np.ones(len(data)) / len(data)
        centroids = np.vstack([centroids, data[rng.choice(len(data), p=probs)]])

    for _ in range(max_iters):
        dists = np.abs(data - centroids.T)
        assignments = np.argmin(dists, axis=1)
        new_centroids = np.zeros_like(centroids)
        changed = False
        for i in range(k):
            members = data[assignments == i]
            if len(members) > 0:
                new_c = np.array([[float(np.mean(members))]])
                if not np.allclose(centroids[i], new_c):
                    changed = True
                new_centroids[i] = new_c
            else:
                new_centroids[i] = data[rng.choice(len(data))]
                changed = True
        centroids = new_centroids
        if not changed:
            break

    return np.sort(centroids.ravel())


def calculate_fpn_specs(coefficient: int) -> List[Dict[str, Any]]:
    """Strides and reference base sizes per FPN level (used by area/default paths)."""
    strides = [4, 8, 16, 32, 64, 128]
    levels = ["P2", "P3", "P4", "P5", "P6", "P7"]

    specs: List[Dict[str, Any]] = []
    for level, stride in zip(levels, strides):
        base_size = stride * coefficient
        specs.append(
            {
                "level": level,
                "stride": stride,
                "base_size": base_size,
                "nominal_area": base_size ** 2,
            }
        )
    return specs


def assign_to_level_by_area(
    box_dimensions: Tuple[float, float],
    specs: List[Dict[str, Any]],
    coefficient: int,) -> Dict[str, Any]:
    """Assign GT to FPN level by log-area proximity (original strategy)."""
    y_dim, x_dim = box_dimensions
    box_area = y_dim * x_dim
    log_area = np.log2(box_area) if box_area > 0 else 0.0

    best_level = specs[0]
    min_dist = float("inf")
    max_visible_level: Optional[str] = None

    for s in specs:
        dist = abs(log_area - np.log2(s["nominal_area"]))
        if dist < min_dist:
            min_dist = dist
            best_level = s

        if box_area >= (4 * (s["nominal_area"]) / (coefficient ** 2)):
            max_visible_level = s["level"]

    return {
        "assigned_level": best_level["level"],
        "assigned_base_size": best_level["base_size"],
        "max_visible_level": max_visible_level,
        "min_dimension": float(min(y_dim, x_dim)),
        "assignment_method": "area_based",
    }


def assign_to_level_by_dimension(
    box_dimensions: Tuple[float, float],
    specs: List[Dict[str, Any]],
    coefficient: int,
    fallback_level: str,
) -> Dict[str, Any]:
    """
    Assign GT to the **coarsest** FPN level L with ``stride(L) * coefficient <= min(h, w)``.

    Falls back to ``fallback_level`` (finest available in specs) when none qualify.
    """
    y_dim, x_dim = box_dimensions
    min_dim = float(min(y_dim, x_dim))
    box_area = y_dim * x_dim

    spec_by_level = {s["level"]: s for s in specs}
    qualifying = [s for s in specs if s["stride"] * coefficient <= min_dim]

    if qualifying:
        chosen = max(qualifying, key=lambda s: s["stride"])
    else:
        chosen = spec_by_level.get(fallback_level, specs[0])

    max_visible_level: Optional[str] = None
    for s in specs:
        if box_area >= (4 * (s["nominal_area"]) / (coefficient ** 2)):
            max_visible_level = s["level"]

    return {
        "assigned_level": chosen["level"],
        "assigned_base_size": chosen["base_size"],
        "max_visible_level": max_visible_level,
        "min_dimension": min_dim,
        "assignment_threshold": chosen["stride"] * coefficient,
        "assignment_method": "dimension_based",
    }


def compute_anchor_config_hash(
    method: str,
    method_parameters: Dict[str, Any],
) -> str:
    """Stable id for ``method`` + ``method_parameters`` (MD5 hex, 32 chars)."""
    payload = {
        "method": method,
        "method_parameters": validate_method_parameters(method, method_parameters),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def anchor_config_output_dir(
    dataset_hash: str,
    split_hash: str,
    config_hash: str,
) -> Path:
    """``artifacts/{dataset_hash}/{split_hash}/{config_hash}/anchors_data/``."""
    from dl_lib.etalon_object_detection.modules.path_layout import anchors_data_dir

    return anchors_data_dir(dataset_hash, split_hash, config_hash)


def _collect_enriched_samples(
    dataset: WeldingDetectionDataset,
    fpn_specs: List[Dict[str, Any]],
    assign_fn: Callable[[Tuple[float, float]], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Tuple[float, float]]]]:
    """Load labels, assign each GT box, return enriched samples and per-level (y, x) lists."""
    target_y_dim, target_x_dim = dataset.target_size
    level_etalons: Dict[str, List[Tuple[float, float]]] = {s["level"]: [] for s in fpn_specs}
    enriched_samples: List[Dict[str, Any]] = []

    for sample in tqdm(dataset.samples, desc="Analyzing Etalons"):
        lbl_path = dataset.cache_dir / sample["lbl_path"]
        if not lbl_path.exists():
            continue

        with open(lbl_path, "r", encoding="utf-8") as f:
            lbl_data = json.load(f)

        orig_x = lbl_data["image_x_dim"]
        orig_y = lbl_data["image_y_dim"]
        scale_x = target_x_dim / orig_x
        scale_y = target_y_dim / orig_y

        enriched_boxes = []
        for box in lbl_data["boxes"]:
            x1, y1, x2, y2 = box
            x1_scaled, x2_scaled = x1 * scale_x, x2 * scale_x
            y1_scaled, y2_scaled = y1 * scale_y, y2 * scale_y
            x_dim = x2_scaled - x1_scaled
            y_dim = y2_scaled - y1_scaled

            assignment = assign_fn((y_dim, x_dim))
            level_etalons[assignment["assigned_level"]].append((y_dim, x_dim))

            enriched_boxes.append(
                {
                    "coords": [x1_scaled, y1_scaled, x2_scaled, y2_scaled],
                    "x_dim": x_dim,
                    "y_dim": y_dim,
                    "area": x_dim * y_dim,
                    "assigned_level": assignment["assigned_level"],
                    "assigned_base_size": assignment["assigned_base_size"],
                    "max_visible_level": assignment["max_visible_level"],
                    "min_dimension": assignment.get("min_dimension"),
                    "assignment_threshold": assignment.get("assignment_threshold"),
                    "assignment_method": assignment.get("assignment_method"),
                }
            )

        s_copy = sample.copy()
        s_copy["enriched_etalons"] = enriched_boxes
        enriched_samples.append(s_copy)

    return enriched_samples, level_etalons


def _optimize_level_clusters_area_based(
    level_etalons: Dict[str, List[Tuple[float, float]]],
    fpn_specs: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> Tuple[List[str], Dict[str, List[float]], Dict[str, List[float]]]:
    """Original: k-means (y,x) per level → aspect ratios only; sizes from fpn base × scales."""
    min_etalons = params["min_etalons_per_level"]
    k2 = params["num_aspect_ratios"]
    scales = params["scales"]
    seed = params["seed"]
    max_iters = params["max_iters"]

    used_fpn_levels: List[str] = []
    level_aspect_ratios: Dict[str, List[float]] = {}
    level_base_sizes: Dict[str, List[float]] = {}

    print("\n--- Level-wise Anchor Optimization (area_based) ---")
    for level, yx_dims in level_etalons.items():
        count = len(yx_dims)
        if count < min_etalons:
            print(f"  {level}: Only {count} etalons. Pruning level.")
            continue

        used_fpn_levels.append(level)
        print(f"  {level}: {count} etalons. Clustering K2={k2} aspect ratios...")

        centroids = generate_anchors(yx_dims, num_anchors=k2, max_iters=max_iters, seed=seed)
        ratios = np.sort(centroids[:, 0] / centroids[:, 1])
        level_aspect_ratios[level] = ratios.tolist()

        spec = next(s for s in fpn_specs if s["level"] == level)
        level_base_sizes[level] = [spec["base_size"] * s for s in scales]

    return used_fpn_levels, level_aspect_ratios, level_base_sizes


def _optimize_level_clusters_dimension_based(
    level_etalons: Dict[str, List[Tuple[float, float]]],
    params: Dict[str, Any],
) -> Tuple[List[str], Dict[str, List[float]], Dict[str, List[float]]]:
    """K1 base sizes (sqrt area) and K2 aspect ratios (h/w) clustered independently per level."""
    min_etalons = params["min_etalons_per_level"]
    k1 = params["num_base_sizes"]
    k2 = params["num_aspect_ratios"]
    seed = params["seed"]
    max_iters = params["max_iters"]

    used_fpn_levels: List[str] = []
    level_aspect_ratios: Dict[str, List[float]] = {}
    level_base_sizes: Dict[str, List[float]] = {}

    print("\n--- Level-wise Anchor Optimization (dimension_based) ---")
    for level, yx_dims in level_etalons.items():
        count = len(yx_dims)
        if count < min_etalons:
            print(f"  {level}: Only {count} etalons. Pruning level.")
            continue

        used_fpn_levels.append(level)
        print(
            f"  {level}: {count} etalons. Clustering K1={k1} base sizes, K2={k2} aspect ratios..."
        )

        base_size_vals = [float(np.sqrt(y * x)) for y, x in yx_dims]
        ratio_vals = [float(y / x) if x > 0 else 1.0 for y, x in yx_dims]

        level_base_sizes[level] = cluster_1d_values(
            base_size_vals, k=k1, seed=seed, max_iters=max_iters
        ).tolist()
        level_aspect_ratios[level] = cluster_1d_values(
            ratio_vals, k=k2, seed=seed + 1, max_iters=max_iters
        ).tolist()

    return used_fpn_levels, level_aspect_ratios, level_base_sizes


def _build_torchvision_default_anchors() -> Tuple[List[str], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Fixed RetinaNet anchors from torchvision (``_default_anchorgen``).

    Per level: three sizes ``(s, s*2^(1/3), s*2^(2/3))`` and aspect ratios ``(0.5, 1, 2)``.
    """
    used_fpn_levels = list(TORCHVISION_DEFAULT_LEVELS)
    level_aspect_ratios: Dict[str, List[float]] = {}
    level_base_sizes: Dict[str, List[float]] = {}

    print("\n--- Anchor config (default / torchvision RetinaNet) ---")
    for level, base in zip(TORCHVISION_DEFAULT_LEVELS, TORCHVISION_DEFAULT_LEVEL_BASE_SIZES):
        sizes = [
            float(base),
            float(int(base * 2 ** (1.0 / 3))),
            float(int(base * 2 ** (2.0 / 3))),
        ]
        level_base_sizes[level] = sizes
        level_aspect_ratios[level] = list(TORCHVISION_DEFAULT_ASPECT_RATIOS)
        print(f"  {level}: sizes={sizes}, aspect_ratios={TORCHVISION_DEFAULT_ASPECT_RATIOS}")

    return used_fpn_levels, level_aspect_ratios, level_base_sizes


def compute_optimized_anchors(
    dataset: WeldingDetectionDataset,
    dataset_hash: str,
    method: str,
    method_parameters: Dict[str, Any],
) -> AnchorOptimizationResult:
    """
    Run anchor assignment and per-level clustering; return an in-memory result.

    Parameters
    ----------
    method :
        ``area_based`` | ``dimension_based`` | ``default``
    method_parameters :
        Required keys per ``METHOD_REQUIRED_PARAMETERS[method]``.
    """
    params = validate_method_parameters(method, method_parameters)
    seed_everything(params["seed"])

    config_hash = compute_anchor_config_hash(method, method_parameters)
    logger.info(
        "Computing anchors: %d samples, method=%s, config_hash=%s",
        len(dataset),
        method,
        config_hash,
    )

    target_y_dim, target_x_dim = dataset.target_size
    fpn_coefficient = params["fpn_coefficient"]
    fpn_specs = calculate_fpn_specs(fpn_coefficient)
    scales = list(params.get("scales", [1.0]))

    if method == "area_based":
        assign_fn = lambda dims: assign_to_level_by_area(dims, fpn_specs, fpn_coefficient)
    elif method == "dimension_based":
        fallback = params["fallback_level"]
        assign_fn = lambda dims: assign_to_level_by_dimension(
            dims, fpn_specs, fpn_coefficient, fallback_level=fallback
        )
    else:
        assign_fn = lambda dims: assign_to_level_by_area(dims, fpn_specs, fpn_coefficient)

    enriched_samples, level_etalons = _collect_enriched_samples(
        dataset, fpn_specs, assign_fn
    )

    if method == "area_based":
        used_fpn_levels, level_aspect_ratios, level_base_sizes = (
            _optimize_level_clusters_area_based(level_etalons, fpn_specs, params)
        )
    elif method == "dimension_based":
        used_fpn_levels, level_aspect_ratios, level_base_sizes = (
            _optimize_level_clusters_dimension_based(level_etalons, params)
        )
    else:
        used_fpn_levels, level_aspect_ratios, level_base_sizes = (
            _build_torchvision_default_anchors()
        )

    return AnchorOptimizationResult(
        method=method,
        method_parameters=params,
        config_hash=config_hash,
        dataset_hash=dataset_hash,
        img_size=(target_y_dim, target_x_dim),
        fpn_specs=fpn_specs,
        fpn_coefficient=fpn_coefficient,
        used_fpn_levels=used_fpn_levels,
        level_aspect_ratios=level_aspect_ratios,
        level_base_sizes=level_base_sizes,
        enriched_samples=enriched_samples,
        seed=params["seed"],
        scales=scales,
    )