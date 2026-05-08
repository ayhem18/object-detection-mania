import os
from pathlib import Path

def get_project_root() -> Path:
    """
    Dynamically finds the project root by climbing up from the current script's directory
    until it finds a directory containing 'src'.
    """
    current_dir = Path(__file__).resolve().parent
    while 'src' not in os.listdir(current_dir):
        parent_dir = current_dir.parent
        if parent_dir == current_dir:
            raise RuntimeError("Could not find project root (directory containing 'src').")
        current_dir = parent_dir
    return current_dir

def get_yolo_v2_artifacts_dir() -> Path:
    """Returns the path to the YOLOv2 synthetic artifacts directory."""
    root = get_project_root()
    path = root / "src" / "object_detection_mania" / "artifacts" / "yolo_v2_artifacts" / "synthetic"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_yolo_v2_config_dir() -> Path:
    """Returns the path to the YOLOv2 synthetic configs directory."""
    path = get_yolo_v2_artifacts_dir() / "configs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_yolo_v2_visualization_dir(subfolder: str) -> Path:
    """Returns a visualization subfolder path within YOLOv2 artifacts."""
    path = get_yolo_v2_artifacts_dir() / "visualizations" / subfolder
    path.mkdir(parents=True, exist_ok=True)
    return path
