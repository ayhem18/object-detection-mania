#!/usr/bin/env bash
# Parallel capacity sweeps — run each command in its own terminal (from repo root).
#
# Prerequisites:
#   uv run python src/dl_lib/etalon_object_detection/scripts/anchors/register_anchor_configs.py
#   uv run python src/dl_lib/etalon_object_detection/scripts/labeling/obj_det_ds_prep.py
#
# Optional anchor config hash as first argument to model_params_sweep.py (default: dimension_based).

# GPU 4 — full dataset
CUDA_VISIBLE_DEVICES=7 uv run python src/dl_lib/etalon_object_detection/scripts/object_detection/parameter_sweeps/model_params_sweep.py --dataset-version full_dataset

# GPU 5 — large diameters (batch_large_diameters)
CUDA_VISIBLE_DEVICES=5 uv run python src/dl_lib/etalon_object_detection/scripts/object_detection/parameter_sweeps/model_params_sweep.py --dataset-version batch_large_diameters

# GPU 6 — small diameters (batch_small_diameteres_1 + batch_small_diameteres_2)
CUDA_VISIBLE_DEVICES=6 uv run python src/dl_lib/etalon_object_detection/scripts/object_detection/parameter_sweeps/model_params_sweep.py --dataset-version small_diameteres
