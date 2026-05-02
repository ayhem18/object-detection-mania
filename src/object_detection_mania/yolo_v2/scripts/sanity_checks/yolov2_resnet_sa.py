import torch
import json
import random

from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset

from mypt.backbones.resnetFE import ResnetFE
from object_detection_mania.yolo_v2.modules.yolov2_ds import YoloV2Dataset, yolov2_collate_fn
from object_detection_mania.yolo_v2.modules.yolov2_model import YoloV2
from object_detection_mania.yolo_v2.modules.target_calculation import YoloV2TargetCalculator
from object_detection_mania.yolo_v2.modules.yolov2_loss import YoloV2Loss
from object_detection_mania.general.path_utils import get_yolo_v2_artifacts_dir, get_yolo_v2_config_dir

from mypt.code_utils.pytorch_utils import seed_everything

def select_diverse_subset(configs: List[Dict], num_classes: int) -> List[int]:
    """Selects a diverse 8-sample subset for overfitting."""
    # 2 samples with largest numbers of objects
    sorted_by_obj = sorted(enumerate(configs), key=lambda x: len(x[1]['objects']), reverse=True)
    top_2_indices = [sorted_by_obj[0][0], sorted_by_obj[1][0]]
    
    # 2 background samples (0 objects)
    bg_indices = [i for i, c in enumerate(configs) if len(c['objects']) == 0][:2]
    
    # For each class, choose an image containing that class at random
    class_indices = []
    for c_id in range(num_classes):
        potential = [i for i, c in enumerate(configs) if any(obj[0] == c_id for obj in c['objects'].values())]
        if potential:
            class_indices.append(random.choice(potential))
            
    return list(set(top_2_indices + bg_indices + class_indices))

def get_overfit_dataloaders(config_path: Path, artifacts_dir: Path, selected_indices: List[int], img_shape: Tuple[int, int]):
    """Initializes the dataset and loader for the subset."""
    resnet_mean = [0.485, 0.456, 0.406]
    resnet_std = [0.229, 0.224, 0.225]
    
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(img_shape),
        v2.Normalize(mean=resnet_mean, std=resnet_std)
    ])
    
    full_ds = YoloV2Dataset(
        config_path=str(config_path),
        output_dir=str(artifacts_dir / "rendered" / "val_sa"),
        img_shape=img_shape,
        transforms=transforms
    )
    
    subset_ds = Subset(full_ds, selected_indices)
    loader = DataLoader(subset_ds, batch_size=len(selected_indices), collate_fn=yolov2_collate_fn, shuffle=False)
    return loader

def build_sa_model(num_classes: int, num_anchors: int):
    """Constructs the ResNet18-YOLOv2 model for sanity check."""
    backbone_fe = ResnetFE(
        architecture=50,
        build_by_layer=True,
        num_extracted_layers=4,
        num_extracted_bottlenecks=-1,
        freeze=False,
        freeze_by_layer=True,
        add_global_average=False
    )
    
    model = YoloV2(
        backbone=backbone_fe,
        backbone_out_channels=2048, # resnet50 outputs 2048 channels (Resnet (34 / 18) outputs 512 channels)
        num_anchors=num_anchors,
        num_classes=num_classes,
        num_conv_blocks=2
    )
    return model

def run_overfit_loop(model, loader, target_calc, criterion, optimizer, device, max_epochs=5000):
    """Executes the overfit loop until thresholds are met."""
    print(f"Starting overfit sanity check on {device}...")
    for epoch in tqdm(range(max_epochs), desc="running overfit sanity check"):
        model.train()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            preds = model(inputs)
            with torch.no_grad():
                targets = target_calc(labels, inputs.size(0))
                
            loss_dict = criterion(preds, targets, reduce=True, return_all_losses=True)
            loss = loss_dict['total_loss']
            
            loss.backward()
            optimizer.step()
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Obj: {loss_dict['loss_obj']:.6f}, Reg: {loss_dict['loss_reg']:.6f}, Cls: {loss_dict['loss_cls']:.6f}\n\n")
            
        # Thresholds: CEL < 10^-4, MSE < 10^-3, Obj < 10^-4
        if (loss_dict['loss_cls'] < 1e-4 and 
            loss_dict['loss_reg'] < 1e-3 and 
            loss_dict['loss_obj'] < 1e-4):
            print(f"\n[SUCCESS] Thresholds met at epoch {epoch}!")
            return True, loss_dict
            
    return False, loss_dict

def main():
    seed_everything(seed=42)
    # 1. Paths
    artifacts_dir = get_yolo_v2_artifacts_dir()
    config_dir = get_yolo_v2_config_dir()
    config_path = config_dir / "val_2000_42_config.json"
    
    if not config_path.exists():
        print(f"Config not found: {config_path}. Please generate data first.")
        return

    with open(config_path, 'r') as f:
        full_configs = json.load(f)
        
    # 2. Setup
    num_classes = 4
    num_anchors = 5
    img_shape = (256, 256)
    anchors = [(0.1, 0.1), (0.3, 0.3), (0.5, 0.5), (0.7, 0.7), (0.9, 0.9)]
    
    selected_indices = select_diverse_subset(full_configs, num_classes)
    loader = get_overfit_dataloaders(config_path, artifacts_dir, selected_indices, img_shape=img_shape)
    
    model = build_sa_model(num_classes=num_classes, num_anchors=num_anchors)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    f_map_size = img_shape[0] // 32
    target_calc = YoloV2TargetCalculator(
        num_classes=num_classes,
        anchors=anchors,
        feature_map_shape=(f_map_size, f_map_size)
    )
    
    criterion = YoloV2Loss(
        num_classes=num_classes,
        num_anchors=num_anchors,
        reg_loss_type="mse",
        background_obj_coeff=0.5
    )
    
    # 3. Run
    success, final_losses = run_overfit_loop(model, loader, target_calc, criterion, optimizer, device)
    
    if not success:
        print(f"\n[FAILURE] Failed to reach thresholds.")
    
    print(f"Final Losses - Obj: {final_losses['loss_obj']:.7f}, Reg: {final_losses['loss_reg']:.7f}, Cls: {final_losses['loss_cls']:.7f}")

if __name__ == "__main__":
    main()
