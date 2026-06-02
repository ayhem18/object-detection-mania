import cv2
import json
import torch
import numpy as np

from tqdm import tqdm
from typing import Dict
from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import DataLoader

# Project Imports
from dl_lib.common_tools.path_utils import get_data_dir
from dl_lib.etalon_object_detection.modules.ds_utils import (
    DL_LIB_ETALON_FIXED_SIZE,
    WeldingDetectionDataset,
    collate_fn,
    get_path_split_callables
)
from dl_lib.etalon_object_detection.modules.retinanet.retinanet_detector import (
    build_dl_lab_etalon_detector_from_trained_weights,
)

COLOR_MAP = {
    1: (0, 255, 0),   # wired (etalon) -> using green for visibility in drawing loop below if needed
    0: (255, 255, 255) # background
}

def resolve_experiment_paths(dataset_hash: str = "latest", split_hash: str = "latest", exp_hash: str = "latest") -> Dict[str, Path]:
    """Autodiscovers the latest or specific experiment directory and its dependencies."""
    base_data = get_data_dir()
    runs_root = base_data / "labeling" / "artifacts" / "runs"
    
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found at {runs_root}")

    # 1. Resolve Dataset Hash
    if dataset_hash == "latest":
        ds_dirs = [d for d in runs_root.iterdir() if d.is_dir() and len(d.name) == 32]
        if not ds_dirs: raise FileNotFoundError("No dataset runs found.")
        dataset_hash = max(ds_dirs, key=lambda d: d.stat().st_mtime).name
    
    ds_path = runs_root / dataset_hash

    # 2. Resolve Split Hash
    if split_hash == "latest":
        split_dirs = [d for d in ds_path.iterdir() if d.is_dir() and len(d.name) == 32]
        if not split_dirs: raise FileNotFoundError(f"No splits found for dataset {dataset_hash}")
        split_hash = max(split_dirs, key=lambda d: d.stat().st_mtime).name
        
    split_path = ds_path / split_hash

    # 3. Resolve Experiment Hash
    if exp_hash == "latest":
        exp_dirs = [d for d in split_path.iterdir() if d.is_dir() and len(d.name) == 32]
        if not exp_dirs: raise FileNotFoundError(f"No experiments found for split {split_hash}")
        exp_hash = max(exp_dirs, key=lambda d: d.stat().st_mtime).name
        
    exp_path = split_path / exp_hash
    
    # 4. Final Path Assembly
    cache_root = base_data / "labeling" / "cache" / dataset_hash
    
    return {
        "exp_dir": exp_path,
        "config": exp_path / "experiment_config.json",
        "checkpoint": exp_path / "checkpoints" / "best_model.pt",
        "anchor_config": split_path / "anchor_config.json",
        "master_json": cache_root / "master_labels.json",
        "images_dir": base_data / "labeling" / "data_as_images",
        "cache_dir": cache_root,
        "output_dir": exp_path / "val_inference"
    }

def save_comparison_visualization(
    image_tensor: torch.Tensor,
    target: Dict[str, torch.Tensor],
    pred: Dict[str, torch.Tensor],
    label_map: Dict[int, str],
    save_path: Path,
    score_thresh: float = 0.3
):
    """Saves a high-quality visualization comparing GT and Predictions."""
    # Convert tensor [C, Y, X] to [Y, X, C] and [0, 255]
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Draw GT (Green thin)
    for box, lbl in zip(target['boxes'].cpu().numpy(), target['labels'].cpu().numpy()):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"GT: {label_map.get(int(lbl), lbl)}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw Predictions (Red thick)
    p_boxes = pred['boxes'].cpu().numpy()
    p_scores = pred['scores'].cpu().numpy()
    p_labels = pred['labels'].cpu().numpy()
    
    keep = p_scores >= score_thresh
    for box, lbl, score in zip(p_boxes[keep], p_labels[keep], p_scores[keep]):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label_text = f"P: {label_map.get(int(lbl), lbl)} ({score:.2f})"
        cv2.putText(img_bgr, label_text, (x1, y2 + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img_bgr)

def run_validation_inference(dataset_hash: str = "latest", split_hash: str = "latest", exp_hash: str = "latest", score_thresh: float = 0.4):
    """Orchestrates loading a specific experiment and running inference on its validation split."""
    
    # 1. Resolve Paths
    paths = resolve_experiment_paths(dataset_hash, split_hash, exp_hash)
    print(f"Targeting Experiment: {paths['exp_dir'].name}")
    
    # 2. Load Config
    with open(paths['config'], 'r') as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Reconstruct Validation Dataset
    _, val_filter, _ = get_path_split_callables(
        master_json_path=str(paths['master_json']),
        val_ratio=config['split_params']['val_ratio'],
        seed=config['split_params']['seed']
    )
    
    val_dataset = WeldingDetectionDataset(
        images_dir=str(paths['images_dir']),
        cache_dir=str(paths['cache_dir']),
        target_size=DL_LIB_ETALON_FIXED_SIZE,
        initial_2_ds_cls_ids=config['initial_2_ds_cls_ids'],
        transformations=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        path_filter=val_filter
    )
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    print(f"Validation split reconstructed: {len(val_dataset)} samples.")

    # 4. Build Model from trained weights
    model_params = config['model_params']
    model = build_dl_lab_etalon_detector_from_trained_weights(
        checkpoint_path=str(paths['checkpoint']),
        manifest_path=str(paths['anchor_config']),
        img_size=model_params['img_size'],
        device=device
    )
    model.eval()
    print("Model loaded successfully from checkpoint.")

    # 5. Inference & Visualization Loop
    paths['output_dir'].mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to: {paths['output_dir']}")

    img_idx = 0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating"):
            images_device = [img.to(device) for img in images]
            outputs = model(images_device)
            
            for i in range(len(images)):
                img_path_rel = val_dataset.samples[img_idx]['img_path']
                dcm_name = Path(img_path_rel).parent.name
                frame_stem = Path(img_path_rel).stem
                
                save_filename = f"{dcm_name}_{frame_stem}_val_det.png"
                save_path = paths['output_dir'] / save_filename
                
                save_comparison_visualization(
                    image_tensor=images[i],
                    target=targets[i],
                    pred=outputs[i],
                    label_map=config.get('cls_id_2_cls_name', {1: "etalon"}),
                    save_path=save_path,
                    score_thresh=score_thresh
                )
                img_idx += 1

    print(f"\nInference complete. Processed {img_idx} validation images.")

if __name__ == "__main__":
    # You can specify specific hashes here if needed
    run_validation_inference(
        dataset_hash="latest", 
        split_hash="latest", 
        exp_hash="latest",
        score_thresh=0.5
    )
