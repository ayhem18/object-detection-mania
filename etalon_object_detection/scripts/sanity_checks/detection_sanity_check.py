# import math
# import torch
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt

# from pathlib import Path
# from torchvision.ops import box_iou
# from typing import Dict, List, Tuple, Any
# from torchvision.transforms import v2
# from torch.utils.data import DataLoader, Subset

# # Internal Imports
# from dl_lib.common_tools.object_detection_analysis import find_max_allowable_delta
# from dl_lib.etalon_object_detection.modules.ds_utils import WeldingDetectionDataset, collate_fn, DL_LIB_ETALON_FIXED_SIZE
# from dl_lib.etalon_object_detection.modules.retinanet.retinanet_train import train_single_epoch
# from dl_lib.etalon_object_detection.modules.retinanet.retinanet_anchors import (
#     anchor_spec_from_manifest_path,
#     resolve_split_anchor_config,
# )
# from dl_lib.etalon_object_detection.modules.retinanet.retinanet_detector import (
#     build_dl_lab_etalon_retinanet_from_segmentation_weights,
# )

# from dl_lib.common_tools.path_utils import get_data_dir, get_label_map

# # Constants
# LABEL_MAP = get_label_map()

# def resolve_paths(dataset_hash: str) -> Dict[str, Path]:
#     """Resolves all required project paths based on the dataset hash."""
#     base_data = get_data_dir()
#     obj_det_root = base_data / "labeling"
#     dataset_cache_dir = obj_det_root / "cache" / dataset_hash
    
#     return {
#         "root": base_data.parent,
#         "images": obj_det_root / "data_as_images",
#         "cache": dataset_cache_dir,
#         "weights": base_data / "weights" / "latest_segmentation_model.pt",
#         "artifacts": obj_det_root / "artifacts" / "sanity_checks" / "detection_sa" / dataset_hash
#     }

# def select_diverse_samples(dataset: WeldingDetectionDataset) -> List[int]:
#     """
#     Selects a diverse subset of indices:
#     - 4 densest samples (most objects)
#     - 2 empty samples (background)
#     - 2 samples with only ONE 'duplex'
#     - 2 samples with only ONE 'groove'
#     - 2 samples with only ONE 'wired'
#     """
#     counts = []
#     empty_indices = []
#     # Map class_id -> list of indices where only 1 etalon of that class exists
#     class_to_single_indices = {1: [], 2: [], 3: []}
    
#     print("Scanning dataset for diverse samples...")
#     for i in range(len(dataset)):
#         _, target = dataset[i]
#         labels = target['labels'].tolist()
#         num_labels = len(labels)
        
#         if num_labels == 0:
#             empty_indices.append(i)
#         else:
#             counts.append((i, num_labels))
#             if num_labels == 1:
#                 cls_id = labels[0]
#                 if cls_id in class_to_single_indices:
#                     class_to_single_indices[cls_id].append(i)
                
#     counts.sort(key=lambda x: x[1], reverse=True)
#     dense_selected = [c[0] for c in counts[:4]]
#     empty_selected = empty_indices[:2]
    
#     selected_indices = list(dense_selected)
#     for idx in empty_selected:
#         if idx not in selected_indices:
#             selected_indices.append(idx)
    
#     # Add 2 samples for each specific class that contain ONLY one etalon of that class
#     for cls_id in [1, 2, 3]:
#         count = 0
#         for idx in class_to_single_indices[cls_id]:
#             if idx not in selected_indices:
#                 selected_indices.append(idx)
#                 count += 1
#                 if count >= 2:
#                     break
                    
#     print(f"Selected {len(selected_indices)} diverse samples for sanity check.")
#     return selected_indices

# def simulate_imperfect_boxes(gt_boxes: torch.Tensor, delta_rel: float, img_x_dim: int, img_y_dim: int) -> torch.Tensor:
#     """Simulates boxes with coordinate error (delta) for passing threshold."""
#     delta_px = delta_rel * (img_x_dim + img_y_dim) / 2
#     imperfect_boxes = gt_boxes.clone()
#     imperfect_boxes[:, [0, 2]] += delta_px
#     imperfect_boxes[:, [1, 3]] += delta_px
#     return imperfect_boxes

# def get_golden_target_loss(
#     model: torch.nn.Module, 
#     samples: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
#     device: torch.device,
#     max_delta_rel: float,
#     target_p: float = 1 - 1e-6) -> Dict[str, float]:
#     """Computes theoretical passing losses by constructing synthetic model outputs."""
#     model.eval()
#     images, targets = zip(*samples)
#     images = [img.to(device) for img in images]
#     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
#     with torch.no_grad():
#         transformed_images, transformed_targets = model.transform(images, targets)
#         features = list(model.backbone(transformed_images.tensors).values())
#         anchors = model.anchor_generator(transformed_images, features)
        
#         matched_idxs = []
#         for anchors_per_image, targets_per_image in zip(anchors, transformed_targets):
#             if targets_per_image["boxes"].numel() == 0:
#                 matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=device))
#                 continue
#             match_matrix = box_iou(targets_per_image["boxes"], anchors_per_image)
#             matched_idxs.append(model.proposal_matcher(match_matrix))
            
#         total_anchors = anchors[0].shape[0]
#         num_classes = model.head.classification_head.num_classes
#         bg_bias = -math.log((1 - 0.01) / 0.01)
#         synthetic_cls = torch.full((len(images), total_anchors, num_classes), bg_bias, device=device)
#         synthetic_reg = torch.zeros((len(images), total_anchors, 4), device=device)
#         pos_logit = math.log(target_p / (1 - target_p))
        
#         for i, m_idx in enumerate(matched_idxs):
#             pos_mask = m_idx >= 0
#             if not pos_mask.any(): continue
#             synthetic_cls[i, pos_mask, transformed_targets[i]["labels"][m_idx[pos_mask]]] = pos_logit
#             img_y_dim, img_x_dim = transformed_images.image_sizes[i]
#             imperfect_boxes = simulate_imperfect_boxes(transformed_targets[i]["boxes"][m_idx[pos_mask]], max_delta_rel, img_x_dim, img_y_dim)
#             synthetic_reg[i, pos_mask] = model.box_coder.encode_single(imperfect_boxes, anchors[i][pos_mask])
            
#         head_outputs = {"cls_logits": synthetic_cls, "bbox_regression": synthetic_reg}
#         loss_dict = model.head.compute_loss(transformed_targets, head_outputs, anchors, matched_idxs)
        
#     return {
#         "total": sum(l for l in loss_dict.values()).item(),
#         "classification": loss_dict["classification"].item(),
#         "bbox_regression": loss_dict["bbox_regression"].item()
#     }

# def get_max_allowable_delta(dataset: WeldingDetectionDataset, target_iou: float, target_recall: float) -> float:
#     """Calculates relative coordinate error (delta) allowed for quality targets."""
#     print("--- Phase 1: Analyzing Dataset Quality Thresholds ---")
#     dim_distribution = {}
#     total_objs = 0
#     target_y_dim, target_x_dim = dataset.target_size
    
#     for i in range(len(dataset)):
#         _, target = dataset[i]
#         for box in target['boxes']:
#             x_dim, y_dim = (box[2] - box[0]).item() / target_x_dim, (box[3] - box[1]).item() / target_y_dim
#             dims = (round(x_dim, 4), round(y_dim, 4))
#             dim_distribution[dims] = dim_distribution.get(dims, 0) + 1
#             total_objs += 1
#     dim_distribution = {k: v / total_objs for k, v in dim_distribution.items()}
#     delta = find_max_allowable_delta(dim_distribution, target_iou=target_iou, target_recall=target_recall)
#     print(f"Calculated Max Allowable Relative Delta: {delta:.6f}")
#     return delta

# def get_cached_golden_target(model, samples, device, delta, artifact_dir, target_iou, target_recall, target_p) -> Dict[str, float]:
#     """Retrieves or computes the golden target loss."""
#     cache_file = artifact_dir / f"golden_target_iou{target_iou}_rec{target_recall}.pkl"
#     if cache_file.exists():
#         with open(cache_file, "rb") as f:
#             golden = pickle.load(f)
#         print(f"Loaded Golden Target from cache: {cache_file}")
#         return golden

#     print("Computing fresh Golden Target simulation...")
#     golden = get_golden_target_loss(model, samples, device, delta, target_p)
#     with open(cache_file, "wb") as f:
#         pickle.dump(golden, f)
#     return golden

# def overfit_model(model, loader, optimizer, device, golden_total, save_path, max_epochs: int):
#     """Attempts to overfit the model until it reaches the golden loss threshold."""
#     print(f"Attempting to overfit for {max_epochs} epochs...")
#     best_loss = float('inf')
#     for epoch in range(max_epochs):
#         metrics = train_single_epoch(model, loader, optimizer, device)
#         if metrics['epoch_loss'] < best_loss:
#             best_loss = metrics['epoch_loss']
#             torch.save(model.state_dict(), save_path)
        
#         print_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
#         print(print_str)

#         with open(save_path.with_suffix(".txt"), "a") as f:
#             f.write(print_str + "\n")
            
#         if metrics['epoch_loss'] <= golden_total:
#             print(f"\n✅ SUCCESS! Model reached capacity threshold in {epoch+1} epochs.")
#             return True
#     print(f"\n❌ FAILURE: Model could not reach threshold ({best_loss:.6f} > {golden_total:.6f})")
#     return False

# def visualize_overfit_results(model, loader, device, viz_dir, score_thresh=0.5):
#     """Generates comparison plots of GT vs Preds."""
#     model.eval()
#     viz_dir.mkdir(parents=True, exist_ok=True)
#     print(f"Generating visualizations in {viz_dir}...")
#     count = 0
#     with torch.no_grad():
#         for images, targets in loader:
#             images_list = [img.to(device) for img in images]
#             outputs = model(images_list)
#             for i in range(len(images)):
#                 img_np = np.clip(images[i].cpu().permute(1, 2, 0).numpy(), 0, 1)
#                 plt.figure(figsize=(15, 10))
#                 plt.imshow(img_np)
#                 ax = plt.gca()
                
#                 for box, lbl in zip(targets[i]['boxes'].cpu().numpy(), targets[i]['labels'].cpu().numpy()):
#                     x1, y1, x2, y2 = box
#                     ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2, linestyle='--'))
#                     ax.text(x1, y1-15, f"GT: {LABEL_MAP.get(lbl, lbl)}", color='lime', backgroundcolor='black', fontsize=8)
                
#                 p_boxes, p_scores, p_labels = outputs[i]['boxes'].cpu().numpy(), outputs[i]['scores'].cpu().numpy(), outputs[i]['labels'].cpu().numpy()
#                 keep = p_scores > score_thresh
#                 for box, lbl, score in zip(p_boxes[keep], p_labels[keep], p_scores[keep]):
#                     x1, y1, x2, y2 = box
#                     ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2))
#                     ax.text(x1, y2+15, f"Pred: {LABEL_MAP.get(lbl, lbl)} ({score:.2f})", color='red', backgroundcolor='black', fontsize=8)
                
#                 plt.axis('off')
#                 plt.savefig(viz_dir / f"sample_{count:02d}.png", bbox_inches='tight')
#                 plt.close()
#                 count += 1

# def run_capacity_check(config: Dict[str, Any]):
#     # from mypt.code_utils.pytorch_utils import seed_everything
#     # seed_everything(config.get("seed", 42))
#     torch.manual_seed(config.get("seed", 42))
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     dataset_hash = config["dataset_hash"]
#     paths = resolve_paths(dataset_hash)
#     paths['artifacts'].mkdir(parents=True, exist_ok=True)

#     # 1. Dataset & Analytics
#     dataset = WeldingDetectionDataset(
#         str(paths['images']), 
#         str(paths['cache']), 
#         target_size=DL_LIB_ETALON_FIXED_SIZE,
#         transforms=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
#     )
    
#     target_iou = config.get("target_iou", 0.95)
#     target_recall = config.get("target_recall", 0.95)
#     target_p = config.get("target_p", 0.99999)
#     max_epochs = config.get("epochs", 500)
    
#     max_delta = get_max_allowable_delta(dataset, target_iou, target_recall)
    
#     # 2. Model & Golden Target
#     indices = select_diverse_samples(dataset)
#     subset = Subset(dataset, indices)
#     # We load all samples in one batch for overfitting simplicity
#     loader = DataLoader(subset, batch_size=len(subset), shuffle=False, collate_fn=collate_fn)
    
#     anchor_config_path, _ = resolve_split_anchor_config(dataset_hash)
#     anchor_spec = anchor_spec_from_manifest_path(anchor_config_path)
#     model = build_dl_lab_etalon_retinanet_from_segmentation_weights(
#         anchor_spec=anchor_spec,
#         segmentation_weights_path=str(paths["weights"]),
#         img_size=DL_LIB_ETALON_FIXED_SIZE,
#         freeze_layers=0,
#         device=device,
#     )
    
#     golden = get_cached_golden_target(
#         model, 
#         [dataset[i] for i in indices], 
#         device, 
#         max_delta, 
#         paths['artifacts'],
#         target_iou,
#         target_recall,
#         target_p
#     )
    
#     print(f"\nGOLDEN TARGET: {golden['total']:.6f} (Cls: {golden['classification']:.6f}, Reg: {golden['bbox_regression']:.6f})\n")

#     # 3. Training with Caching
#     model_save_path = paths['artifacts'] / f"best_overfit_iou{target_iou}_rec{target_recall}_loss{golden['total']:.6f}.pt"
#     viz_dir = paths['artifacts'] / f"viz_iou{target_iou}_rec{target_recall}_loss{golden['total']:.6f}"

#     if model_save_path.exists():
#         print(f"Checkpoint found for target loss {golden['total']:.6f}. Skipping training.")
#     else:
#         overfit_model(
#             model, 
#             loader, 
#             torch.optim.AdamW(model.parameters(), lr=1e-3), 
#             device, 
#             golden['total'], 
#             model_save_path,
#             max_epochs
#         )

#     # 4. Visualization
#     model.load_state_dict(torch.load(model_save_path, weights_only=True))
#     visualize_overfit_results(model, loader, device, viz_dir)

# def main():
#     config = {
#         "dataset_hash": "latest",
#         "target_iou": 0.95,
#         "target_recall": 0.95,
#         "target_p": 0.99999,
#         "epochs": 500,
#         "seed": 42
#     }
    
#     if config["dataset_hash"] == "latest":
#         script_dir = Path(__file__).resolve().parent
#         root_dir = script_dir
#         while not (root_dir / 'data').exists(): root_dir = root_dir.parent
#         cache_dir = root_dir / "data" / "labeling" / "cache"
#         if cache_dir.exists():
#             subdirs = [d for d in cache_dir.iterdir() if d.is_dir() and len(d.name) == 32]
#             if subdirs:
#                 latest_hash_dir = max(subdirs, key=os.path.getmtime)
#                 config["dataset_hash"] = latest_hash_dir.name
#                 print(f"Using latest dataset hash: {config['dataset_hash']}")
    
#     run_capacity_check(config)

# if __name__ == "__main__":
#     main()
