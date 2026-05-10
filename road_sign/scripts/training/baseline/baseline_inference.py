import os
import sys
import cv2
import torch
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from typing import Optional
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
while True:
    if 'road_sign' in os.listdir(current_dir) and os.path.isdir(os.path.join(current_dir, 'road_sign')):
        break
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:
        raise RuntimeError("Could not find 'road_sign' directory in the parent path.")
    current_dir = parent_dir

workspace_root = current_dir
road_sign_root = os.path.join(workspace_root, 'road_sign')
sys.path.insert(0, workspace_root)

from home_made_od.yolo_v2.modules.yolov2_model import YoloV2
from home_made_od.yolo_v2.modules.anchors.anchor_utils import generate_anchors
from mypt.backbones.resnetFE import ResnetFE
from mypt.code_utils.pytorch_utils import seed_everything

class InferenceDataset(Dataset):
    def __init__(self, image_paths, img_size, transform):
        self.image_paths = image_paths
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # Return a dummy tensor if read fails, handle in collate or inference
            return torch.zeros((3, self.img_size[1], self.img_size[0])), img_path, (0, 0)
        
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        input_tensor = self.transform(img_rgb)
        return input_tensor, img_path, (orig_w, orig_h)

def get_data_pairs(data_dir):
    img_dir = os.path.join(data_dir, 'train', 'images')
    label_dir = os.path.join(data_dir, 'labels', 'annotations')
    
    pairs = []
    if not os.path.exists(img_dir):
        print(f"Warning: Image directory not found at {img_dir}")
        return pairs

    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            stem = Path(img_file).stem
            label_file = os.path.join(label_dir, f"{stem}.txt")
            if not os.path.exists(label_file):
                continue
            pairs.append((os.path.join(img_dir, img_file), label_file))
    return pairs

def get_anchors(data_pairs, num_anchors=5):
    print("Generating anchors from dataset...")
    wh_list = []
    for _, label_path in data_pairs:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    w, h = float(parts[3]), float(parts[4])
                    wh_list.append((w, h))
    anchors = generate_anchors(wh_list, num_anchors=num_anchors)
    print(f"Generated Anchors: {anchors.tolist()}")
    return anchors.tolist()

def build_model(num_classes, num_anchors):
    resnet_fe = ResnetFE(
        build_by_layer=True,
        num_extracted_layers=-1, 
        num_extracted_bottlenecks=-1,
        freeze=2,
        freeze_by_layer=True,
        add_global_average=False,
        architecture=50
    )
    
    backbone_out_channels = 2048 
    
    model = YoloV2(
        backbone=resnet_fe,
        backbone_out_channels=backbone_out_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        num_conv_blocks=2
    )
    
    return model, resnet_fe.transform

def visualize_predictions(results_dict, output_dir, max_images:Optional[int]=10):
    """
    Visualizes bounding boxes on the original images for a subset of the results.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Visualizing up to {max_images} predictions to {output_dir}")
    
    count = 0
    for img_path, preds in results_dict.items():
        if max_images is not None and count >= max_images:
            break
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        for det in preds:
            x1, y1, x2, y2, score, label_name = det
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
            label_str = f"{label_name} {score:.2f}"
            cv2.putText(img, label_str, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            
        # Resize for easier viewing
        h, w = img.shape[:2]
        scale = 1000 / max(w, h)
        small_img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        out_path = os.path.join(output_dir, f"pred_{Path(img_path).name}")
        cv2.imwrite(out_path, small_img)
        count += 1

def generate_submission(model, test_dir, output_csv, img_size, transform, class_mapping, device, anchors, batch_size=16, conf_thresh=0.25, boost_confidence=False):
    """
    Runs batched inference on all images in test_dir and saves a Kaggle-formatted CSV.
    """
    model.eval()
    
    # Class mapping dict (e.g. {"0": "Bus stop"}) -> array of names with underscores
    class_names = []
    for i in range(len(class_mapping)):
        raw_name = class_mapping.get(str(i), str(i))
        class_names.append(raw_name.replace(" ", "_"))

    # The transform pipeline expecting numpy arrays (H,W,C) -> Tensor (C,H,W)
    preprocess = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(img_size),
        v2.Normalize(mean=transform.mean, std=transform.std)
    ])

    test_images = list(Path(test_dir).glob("*.jpg")) + list(Path(test_dir).glob("*.png"))
    print(f"Found {len(test_images)} test images.")
    
    dataset = InferenceDataset(test_images, img_size, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    results = []
    visualization_data = {} # {img_path: [(x1, y1, x2, y2, score, label_name), ...]}
    
    with torch.no_grad():
        for batch_tensors, batch_paths, batch_orig_dims in tqdm(dataloader, desc="Running Inference"):
            batch_tensors = batch_tensors.to(device)
            orig_widths = batch_orig_dims[0].numpy()
            orig_heights = batch_orig_dims[1].numpy()
            
            # Run batched inference
            batch_preds = model.inference(batch_tensors, anchors=anchors, conf_threshold=conf_thresh, nms_iou_threshold=0.4)
            
            for i, preds in enumerate(batch_preds):
                img_path = batch_paths[i]
                orig_w = orig_widths[i]
                orig_h = orig_heights[i]
                
                # Handle dummy tensor from failed cv2.imread
                if orig_w == 0 or orig_h == 0:
                    results.append({"image_id": Path(img_path).stem, "PredictionString": "No_parking 0.0001 0 0 1 1"})
                    continue
                    
                prediction_strings = []
                vis_preds = []
                
                for det in preds:
                    x1, y1, x2, y2, score, cls_id = det.cpu().numpy()
                    
                    if boost_confidence:
                        score = min(score + 0.5, 0.99)

                    # YOLO outputs absolute coordinates based on the 512x512 input size.
                    # We need to scale these back to the original image dimensions.
                    scale_x = orig_w / img_size[0]
                    scale_y = orig_h / img_size[1]
                    
                    orig_x1 = int(x1 * scale_x)
                    orig_y1 = int(y1 * scale_y)
                    orig_x2 = int(x2 * scale_x)
                    orig_y2 = int(y2 * scale_y)
                    
                    label_name = class_names[int(cls_id)]
                    pred_str = f"{label_name} {score:.4f} {orig_x1} {orig_y1} {orig_x2} {orig_y2}"
                    prediction_strings.append(pred_str)
                    
                    vis_preds.append((orig_x1, orig_y1, orig_x2, orig_y2, score, label_name))
                
                if not prediction_strings:
                    final_str = "No_parking 0.0001 0 0 1 1"
                else:
                    final_str = " ".join(prediction_strings)

                results.append({
                    "image_id": Path(img_path).stem,
                    "PredictionString": final_str
                })
                
                # Store a few for visualization
                if len(visualization_data) < 20 and len(vis_preds) > 0:
                    visualization_data[img_path] = vis_preds

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")
    
    return visualization_data

def main(boost_confidence=False):
    seed_everything(42)
    
    # Configuration
    DATA_DIR = os.path.join(road_sign_root, 'data_512')
    # CHECKPOINT_PATH = os.path.join(road_sign_root, 'artifacts', 'baseline', 'checkpoints', 'best_model.pt')
    CHECKPOINT_PATH = os.path.join(road_sign_root, 'artifacts', 'baseline', '893b8fced220f0a96492f50a02a8da7e', 'checkpoints', 'best_model.pt')
    TEST_DIR = os.path.join(road_sign_root, 'data', 'test', 'images')
    
    # Submission path: artifacts/baseline/submission/submission.csv
    SUBMISSION_DIR = os.path.join(os.path.dirname(os.path.dirname(CHECKPOINT_PATH)), 'submission')
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    filename = 'submission_boosted.csv' if boost_confidence else 'submission.csv'
    OUTPUT_CSV = os.path.join(SUBMISSION_DIR, filename)
    VISUALIZATION_DIR = os.path.join(SUBMISSION_DIR, 'visualizations')
    
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 32
    NUM_CLASSES = 8
    NUM_ANCHORS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Get Data Pairs from training set to recalculate same anchors used in training
    all_pairs = get_data_pairs(DATA_DIR)
    
    # 2. Recalculate Anchors
    anchors = get_anchors(all_pairs, num_anchors=NUM_ANCHORS)

    import json
    class_mapping_path = os.path.join(road_sign_root, 'data', 'class_mapping.json')
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)

    print("Building model...")
    model, transform_stats = build_model(NUM_CLASSES, NUM_ANCHORS)
    model.to(DEVICE)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}. Using untrained weights.")
        return

    print(f"Loading weights from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    vis_data = generate_submission(
        model, TEST_DIR, OUTPUT_CSV, IMG_SIZE, transform_stats, 
        class_mapping, DEVICE, anchors, batch_size=BATCH_SIZE, conf_thresh=0.1,
        boost_confidence=boost_confidence
    )
    
    visualize_predictions(vis_data, VISUALIZATION_DIR, max_images=None)


if __name__ == "__main__":
    main(boost_confidence=True)
