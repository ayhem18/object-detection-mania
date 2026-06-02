import os
import cv2
import json
import torch
import shutil
import zipfile
import numpy as np
import logging

from tqdm import tqdm
from typing import List, Tuple
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Project Imports
from dl_lib.common_tools.path_utils import get_data_dir
from dl_lib.etalon_object_detection.scripts.labeling.seg_model import build_welding_mask_rcnn, filter_predictions

logging.basicConfig(level=logging.INFO, format='%(levelname)s: [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

BATCH_SIZE_INFERENCE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_ID_WIRED = 1 # Segmentation model ID
YOLO_CLASS_ID_WIRED = 0 # YOLO/CVAT ID
SAVE_VISUALIZATION = True # Flag to enable/disable visualization saving

CVAT_PREFIX="digital_lab_wired_etalons/data_as_images"


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    try:
        ext = os.path.splitext(path)[1]
        ok, buf = cv2.imencode(ext, img)
        if not ok: return False
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
        return True
    except Exception as e:
        print(f"Failed to write image {path}: {e}")
        return False

def imread_unicode(path: str) -> np.ndarray:
    try:
        with open(path, 'rb') as f:
            chunk = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(chunk, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

class DcmFramesDataset(Dataset):
    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img_bgr = imread_unicode(path)
        if img_bgr is None:
            return torch.zeros((3, 1244, 1244), dtype=torch.float32), path, np.zeros((1244, 1244, 3), dtype=np.uint8)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_scaled = img_rgb.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(img_scaled.transpose(2, 0, 1))
        
        return input_tensor, path, img_rgb

def collate_fn(batch):
    tensors = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    imgs = [item[2] for item in batch]
    
    # Check if all tensors have the same shape
    first_shape = tensors[0].shape
    all_same = all(t.shape == first_shape for t in tensors)
    
    if all_same:
        tensors = torch.stack(tensors)
        
    return tensors, paths, imgs

def create_cvat_zip(batch_data_dir: str, output_zip: str):
    """Packages the prediction folders into a ZIP file formatted for CVAT."""
    print(f"Creating ZIP archive: {output_zip}")
    with zipfile.ZipFile(output_zip, 'w') as zf:
        for root, _, files in os.walk(batch_data_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path_in_zip = os.path.relpath(abs_path, batch_data_dir)
                zf.write(abs_path, rel_path_in_zip)

def run_inference_and_generate_data(model, dataloader, viz_dir: str, labels_tmp_dir: str, score_thresh: float = 0.5) -> List[Tuple[str, str]]:
    """
    Runs inference, saves visualizations (optional), and generates YOLO text files.
    Returns a list of (rel_dcm_name, frame_filename) for all processed frames.
    """
    model.eval()
    
    if SAVE_VISUALIZATION:
        viz_path = Path(viz_dir)
        viz_path.mkdir(parents=True, exist_ok=True)
    
    labels_path = Path(labels_tmp_dir)
    labels_path.mkdir(parents=True, exist_ok=True)

    processed_frames = []
    
    with torch.no_grad():
        for batch_tensors, batch_paths, batch_imgs in tqdm(dataloader, desc="Inference"):
            if isinstance(batch_tensors, list):
                batch_tensors = [t.to(DEVICE) for t in batch_tensors]
            else:
                batch_tensors = batch_tensors.to(DEVICE)
                
            outputs = model(batch_tensors)
            
            for i, output in enumerate(outputs):
                path = Path(batch_paths[i])
                img_rgb = batch_imgs[i]
                if img_rgb is None or img_rgb.size == 0 or np.all(img_rgb == 0):
                    continue
                
                h_img, w_img = img_rgb.shape[:2]
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) if SAVE_VISUALIZATION else None
                preds = filter_predictions(output, conf_threshold=score_thresh, nms_threshold=0.3)
                
                boxes = preds['boxes'].cpu().numpy()
                labels = preds['labels'].cpu().numpy()
                scores = preds['scores'].cpu().numpy()
                
                yolo_lines = []
                for box, label, score in zip(boxes, labels, scores):
                    if label == CLASS_ID_WIRED:
                        x1, y1, x2, y2 = box
                        
                        # Visualization
                        if SAVE_VISUALIZATION:
                            cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(img_bgr, f"wired {score:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # YOLO format: <class> <cx> <cy> <w> <h> normalized
                        bw = x2 - x1
                        bh = y2 - y1
                        cx = x1 + bw / 2
                        cy = y1 + bh / 2
                        
                        yolo_lines.append(f"{YOLO_CLASS_ID_WIRED} {cx/w_img:.6f} {cy/h_img:.6f} {bw/w_img:.6f} {bh/h_img:.6f}")
                
                # Save Visualization
                rel_dcm_name = path.parent.name
                frame_name = path.name
                
                if SAVE_VISUALIZATION:
                    viz_save_path = viz_path / rel_dcm_name / frame_name
                    viz_save_path.parent.mkdir(parents=True, exist_ok=True)
                    imwrite_unicode(str(viz_save_path), img_bgr)
                
                # Save YOLO Labels
                if yolo_lines:
                    label_save_path = labels_path / rel_dcm_name / (path.stem + ".txt")
                    label_save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(label_save_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(yolo_lines))
                
                processed_frames.append((rel_dcm_name, frame_name))
                
    return processed_frames

def finalize_cvat_zip(processed_frames: List[Tuple[str, str]], labels_tmp_dir: str, output_zip: str):
    """Organizes the labels into CVAT/YOLO structure and zips them."""
    temp_root = Path(labels_tmp_dir).parent / "cvat_upload_temp"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    
    # 1. Create labels/train/ structure
    dest_labels_dir = temp_root / "labels" / "train" 
    dest_labels_dir = Path(os.path.join(dest_labels_dir, CVAT_PREFIX))
    dest_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Move labels from temp_labels_dir to dest_labels_dir
    src_labels = Path(labels_tmp_dir)
    if src_labels.exists():
        for item in src_labels.iterdir():
            shutil.move(str(item), str(dest_labels_dir))
    
    
    # 2. Create train.txt
    train_txt_path = temp_root / "train.txt"
    with open(train_txt_path, "w", encoding="utf-8") as f:
        for rel_dcm, frame_name in processed_frames:
            # Match the structure expected by CVAT upload
            line = f"data/images/train/{CVAT_PREFIX}/{rel_dcm}/{frame_name}".replace("\\", "/")
            f.write(line + "\n")
            
    # 3. Create data.yaml
    data_yaml_content = f"names:\n  {YOLO_CLASS_ID_WIRED}: wired\npath: .\ntrain: train.txt\n"
    with open(temp_root / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml_content)
        
    # 4. Zip it
    create_cvat_zip(str(temp_root), output_zip)
    
    # return 

    # Cleanup
    shutil.rmtree(temp_root)
    if src_labels.exists():
        shutil.rmtree(src_labels)

def main():
    data_dir = get_data_dir()
    images_dir = data_dir / "labeling" / "data_as_images"
    viz_dir = data_dir / "labeling" / "initial_predictions"
    cvat_zip_path = data_dir / "labeling" / "cvat_upload.zip"
    labels_tmp_dir = data_dir / "labeling" / "temp_yolo_labels"
    
    weights_path = data_dir / "weights" / "latest_segmentation_model.pt"
    manifest_path = Path(__file__).parent / "anchor_config.json"
    
    index_path = images_dir / "index.json"
    if not index_path.exists():
        print(f"ERROR: {index_path} not found. Please run initial_data_extraction.py first.")
        return
        
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
        
    image_paths = []
    for rel_path in index.values():
        dir_path = images_dir / rel_path
        if dir_path.exists():
            for f in sorted(os.listdir(dir_path)):
                if f.endswith(".png"):
                    image_paths.append(str(dir_path / f))
                    
    print(f"Total images found: {len(image_paths)}")
    if not image_paths:
        print("No images to process.")
        return

    print("--- Phase 1: Model Setup ---")
    model = build_welding_mask_rcnn(
        num_classes=4, 
        manifest_path=str(manifest_path),
        weights_path=None, 
        device=torch.device(DEVICE)
    )
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path}")
    print(f"Loading weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
        
    print("--- Phase 2: Inference, Visualization & YOLO Generation ---")
    dataset = DcmFramesDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_INFERENCE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    processed_info = run_inference_and_generate_data(model, dataloader, str(viz_dir), str(labels_tmp_dir), score_thresh=0.5)
    
    print("--- Phase 3: Finalizing CVAT Zip ---")
    finalize_cvat_zip(processed_info, str(labels_tmp_dir), str(cvat_zip_path))
    
    print(f"Pipeline completed. CVAT zip created at {cvat_zip_path}")

if __name__ == "__main__":
    main()
