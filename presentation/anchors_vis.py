"""
Visualize a detection grid (stride_x × stride_y) and sample anchor boxes on an image.

Anchors are ``(y_dim, x_dim)`` — height and width in the same pixel space as the image
(e.g. resized training dimensions), centered on grid cell centers.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

AnchorSize = Tuple[float, float]
PathLike = Union[str, Path]

_ANCHOR_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (255, 128, 0),
    (0, 200, 255),
    (255, 0, 200),
    (0, 255, 128),
    (200, 128, 255),
)


def _normalize_anchors(anchors: Sequence[Sequence[float]]) -> List[AnchorSize]:
    out: List[AnchorSize] = []
    for a in anchors:
        if len(a) != 2:
            raise ValueError(f"Each anchor must be [y_dim, x_dim], got {a!r}")
        y_dim, x_dim = float(a[0]), float(a[1])
        if y_dim <= 0 or x_dim <= 0:
            raise ValueError(f"Anchor dimensions must be positive, got ({y_dim}, {x_dim})")
        out.append((y_dim, x_dim))
    if not out:
        raise ValueError("anchors must be a non-empty list")
    return out


def _grid_shape(height: int, width: int, stride_y: float, stride_x: float) -> Tuple[int, int]:
    grid_h = int(np.ceil(height / stride_y))
    grid_w = int(np.ceil(width / stride_x))
    return grid_h, grid_w


def _cell_rect(
    row: int,
    col: int,
    *,
    image_height: int,
    image_width: int,
    stride_y: float,
    stride_x: float,
) -> Tuple[int, int, int, int]:
    x1 = int(col * stride_x)
    y1 = int(row * stride_y)
    x2 = int(min((col + 1) * stride_x, image_width))
    y2 = int(min((row + 1) * stride_y, image_height))
    return x1, y1, x2, y2


def _cell_center(row: int, col: int, stride_y: float, stride_x: float) -> Tuple[float, float]:
    cx = (col + 0.5) * stride_x
    cy = (row + 0.5) * stride_y
    return cx, cy


def _anchor_xyxy(
    center_x: float,
    center_y: float,
    y_dim: float,
    x_dim: float,
    *,
    image_height: int,
    image_width: int,
) -> Tuple[int, int, int, int]:
    half_h = y_dim / 2.0
    half_w = x_dim / 2.0
    x1 = int(max(0, round(center_x - half_w)))
    y1 = int(max(0, round(center_y - half_h)))
    x2 = int(min(image_width, round(center_x + half_w)))
    y2 = int(min(image_height, round(center_y + half_h)))
    return x1, y1, x2, y2


def _draw_grid(
    img_bgr: np.ndarray,
    *,
    stride_y: float,
    stride_x: float,
    line_color: Tuple[int, int, int] = (70, 70, 70),
    thickness: int = 1,
) -> None:
    height, width = img_bgr.shape[:2]
    grid_h, grid_w = _grid_shape(height, width, stride_y, stride_x)
    for row in range(grid_h):
        for col in range(grid_w):
            x1, y1, x2, y2 = _cell_rect(
                row,
                col,
                image_height=height,
                image_width=width,
                stride_y=stride_y,
                stride_x=stride_x,
            )
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), line_color, thickness)


def _sample_anchor_placements(
    grid_h: int,
    grid_w: int,
    num_anchors: int,
    num_samples: int,
    *,
    seed: Optional[int],
) -> List[Tuple[int, int, int]]:
    """Return ``(row, col, anchor_index)`` tuples."""
    placements = [
        (row, col, anchor_idx)
        for row in range(grid_h)
        for col in range(grid_w)
        for anchor_idx in range(num_anchors)
    ]
    if not placements:
        return []

    rng = random.Random(seed)
    k = min(num_samples, len(placements))
    return rng.sample(placements, k=k)


def visualize_anchors_on_grid(
    image_path: PathLike,
    stride_x: float,
    stride_y: float,
    anchors: Sequence[Sequence[float]],
    num_samples: int,
    *,
    seed: Optional[int] = 42,
    save_path: Optional[PathLike] = None,
    show: bool = False,
) -> np.ndarray:
    """
    Draw the stride grid and ``num_samples`` anchor boxes on the image.

    Parameters
    ----------
    image_path
        Path to a BGR/RGB image file readable by OpenCV.
    stride_x, stride_y
        Pixel stride of the detection grid along width and height.
    anchors
        List of ``[y_dim, x_dim]`` anchor sizes in image pixels.
    num_samples
        Number of anchor boxes to draw (sampled over grid cells × anchor templates).
    seed
        RNG seed for reproducible sampling; ``None`` for non-deterministic draws.
    save_path
        If set, write the visualization to this path.
    show
        If True, open an OpenCV window until a key is pressed.

    Returns
    -------
    np.ndarray
        Annotated image in BGR (``uint8``), same size as the input.
    """
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError(f"strides must be positive, got stride_x={stride_x}, stride_y={stride_y}")
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")

    path = Path(image_path)
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    anchor_sizes = _normalize_anchors(anchors)
    height, width = img_bgr.shape[:2]
    grid_h, grid_w = _grid_shape(height, width, stride_y, stride_x)

    _draw_grid(img_bgr, stride_y=stride_y, stride_x=stride_x)

    placements = _sample_anchor_placements(
        grid_h,
        grid_w,
        len(anchor_sizes),
        num_samples,
        seed=seed,
    )
    for row, col, anchor_idx in placements:
        y_dim, x_dim = anchor_sizes[anchor_idx]
        cx, cy = _cell_center(row, col, stride_y, stride_x)
        x1, y1, x2, y2 = _anchor_xyxy(
            cx,
            cy,
            y_dim,
            x_dim,
            image_height=height,
            image_width=width,
        )
        color = _ANCHOR_COLORS[anchor_idx % len(_ANCHOR_COLORS)]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img_bgr)

    if show:
        cv2.imshow("anchors_on_grid", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_bgr

if __name__ == "__main__":
    image_path = "data/synthetic_toy_ds/grid_n_6.png"
    stride_x = 16
    stride_y = 16
    anchors = [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256]]
    num_samples = 100
    visualize_anchors_on_grid(image_path, stride_x, stride_y, anchors, num_samples)