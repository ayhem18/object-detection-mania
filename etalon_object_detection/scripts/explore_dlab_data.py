import os
import cv2
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Union, List

# Project Imports
from dl_lib.common_tools.path_utils import get_data_dir
from dl_lib.common_tools.dcm_utils import clean_name, get_dcm_files
from dl_lib.common_tools.data_extraction.data_extractor import process_frames_special
from dl_lib.preprocessing.utils.leveling_wrapper_filter import leveling_defect_filter_u8
from dl_lib.preprocessing.utils.scli import superpixel_regionwise_norm_u16_to_u8

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

# Restrict logs from the data_extractor module to WARNING and above
logging.getLogger("dl_lib.common_tools.data_extraction.data_extractor").setLevel(logging.WARNING)

# ==========================================
# CONFIGURATION
# ==========================================
# By default, use a path that might be passed or default to a known dir if needed.
# For this script, we'll configure it to take an input directory.
# You can change this or pass it as an argument.
DEFAULT_TEST_DCM_DIR = os.path.join(get_data_dir(), "data_dcm") 
PREPROCESSING_TYPE = "scli"  # Options: "leveling", "scli"
# PREPROCESSING_TYPE = "leveling"  # Options: "leveling", "scli"
# ==========================================


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """Save an image path containing unicode characters."""
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


def make_u16_heatmap(frame_u16: np.ndarray) -> np.ndarray:
    """Convert a uint16 frame into a contrast-stretched heatmap for quick inspection."""
    if frame_u16.dtype != np.uint16:
        raise ValueError("Expected a uint16 frame")

    valid_pixels = frame_u16[frame_u16 > 0]
    if valid_pixels.size == 0:
        normalized = np.zeros_like(frame_u16, dtype=np.uint8)
    else:
        lo = float(np.percentile(valid_pixels, 1.0))
        hi = float(np.percentile(valid_pixels, 99.0))
        if hi <= lo:
            hi = float(valid_pixels.max())
        if hi <= lo:
            normalized = np.zeros_like(frame_u16, dtype=np.uint8)
        else:
            clipped = np.clip(frame_u16.astype(np.float32), lo, hi)
            normalized = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)

    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def process_and_save_dcm_frames(input_dcm_dir: Union[str, Path] = DEFAULT_TEST_DCM_DIR):
    """
    Finds all DCM files in the input directory, extracts their frames,
    applies the leveling filter (converting to uint8), and saves them to the data directory.
    """
    input_dir = Path(input_dcm_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input DCM directory not found at {input_dir}")
        return

    # Define output directory
    data_dir = get_data_dir()
    out_dir = data_dir / "data_as_images" / PREPROCESSING_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect DCM files
    dcm_files = get_dcm_files(input_dir)
    
    if not dcm_files:
        print(f"No .dcm files found in {input_dir}")
        return
        
    print(f"Found {len(dcm_files)} DCM files. Starting extraction and processing with '{PREPROCESSING_TYPE}'...")

    total_dcms = 0
    total_frames = 0
    fail_count = 0

    max_dcm_files = 10

    for dcm_path_str in tqdm(dcm_files, desc="Processing DCMs"):
        dcm_path = Path(dcm_path_str)
        dcm_stem = dcm_path.stem
        cleaned_stem = clean_name(dcm_stem)
        dcm_out_dir = out_dir / cleaned_stem

        if dcm_out_dir.exists() and len(os.listdir(dcm_out_dir)) > 0:
            # print(f"Cached.. skipping {dcm_stem}")
            continue

        try:
            frames = process_frames_special(dcm_path)
        except ValueError as e:
            # print(f"Invalid DCM format or assumption failed for {dcm_stem}: {e}")
            fail_count += 1
            continue
        except Exception as e:
            # print(f"Error processing {dcm_stem}: {e}")
            fail_count += 1
            continue
            
        if frames is None:
            fail_count += 1
            continue

        dcm_out_dir.mkdir(parents=True, exist_ok=True)
        total_dcms += 1

        if total_dcms >= max_dcm_files:
            print(f"Reached max DCM file limit ({max_dcm_files}). Stopping further processing.")
            break

        for i, frame in enumerate(frames):
            frame_idx = i + 1
            total_frames += 1

            # Preprocess
            try:
                if frame.dtype != np.uint16:
                    frame = frame.astype(np.uint16)
                    
                if PREPROCESSING_TYPE == "leveling":
                    # Apply wrapper filter which returns uint8
                    proc_frame_u8 = leveling_defect_filter_u8(frame)
                elif PREPROCESSING_TYPE == "scli":
                    # Apply SLIC region-wise normalization
                    proc_frame_u8 = superpixel_regionwise_norm_u16_to_u8(frame)
                else:
                    raise ValueError(f"Unknown PREPROCESSING_TYPE: {PREPROCESSING_TYPE}")
                
            except Exception as e:
                print(f"Error preprocessing {dcm_stem} frame {frame_idx}: {e}")
                fail_count += 1
                continue

            dest_file = dcm_out_dir / f"frame_{frame_idx:03d}.png"
            if not imwrite_unicode(str(dest_file), proc_frame_u8):
                fail_count += 1

            heatmap_file = dcm_out_dir / f"frame_{frame_idx:03d}_u16_heatmap.png"
            try:
                frame_heatmap = make_u16_heatmap(frame)
            except Exception as e:
                print(f"Error building heatmap for {dcm_stem} frame {frame_idx}: {e}")
                fail_count += 1
                continue

            if not imwrite_unicode(str(heatmap_file), frame_heatmap):
                fail_count += 1

    print(f"\n--- Data Extraction Summary ---")
    print(f"Total DCMs processed:   {total_dcms}")
    print(f"Total frames processed: {total_frames}")
    print(f"Failures (read/write):  {fail_count}")
    print(f"Output directory:       {out_dir}")
    print(f"-------------------------------\n")

if __name__ == "__main__":
    process_and_save_dcm_frames()
