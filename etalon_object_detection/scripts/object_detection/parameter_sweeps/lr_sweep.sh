CUDA_VISIBLE_DEVICES=7 uv run python src/dl_lib/etalon_object_detection/scripts/object_detection/parameter_sweeps/lr_sweep.py --dataset-version full_dataset

# GPU 5 — large diameters (batch_large_diameters)
CUDA_VISIBLE_DEVICES=5 uv run python src/dl_lib/etalon_object_detection/scripts/object_detection/parameter_sweeps/model_params_sweep.py --dataset-version batch_large_diameters
