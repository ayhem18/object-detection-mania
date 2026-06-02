import os
import cv2
import json
import logging
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List

from dl_lib.common_tools.path_utils import get_data_dir
from dl_lib.common_tools.dcm_utils import clean_name, get_dcm_files
from dl_lib.etalon_object_detection.modules.path_layout import get_data_maps
from dl_lib.preprocessing.modules.utils.leveling_wrapper_filter import leveling_defect_filter_u8

from weldbook_dcm.reader import read_cyfracon

logging.basicConfig(level=logging.INFO, format='%(levelname)s: [%(name)s] %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("dl_lib.common_tools.data_extraction.data_extractor").setLevel(logging.WARNING)

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

def get_frames_digital_lab(file_path: str) -> List[np.ndarray]:
    """DCM image reading function.
    Processes images as if they were sent by the backend to the module.
    Flips image vertically, calculates the frame length.
    Args:
        file_path: a string containing the full path to the file
    Returns:
        img: the numpy array representing the image in uint16
        img_dim: the length of one from along the x-axis (number of columns)
    """
    images = read_cyfracon(file_path)
    flipped_frames = [f[0][::-1, :] for f in images]
    return flipped_frames

def extract_and_index_data(location: str = "local") -> Dict[str, str]:
    """
    Extracts frames from DCM files using leveling preprocessing and saves them
    in a ``target_batch/dcm_name`` folder structure with per-batch and combined
    index.json files.
    """
    data_maps = get_data_maps(location)
    dest_root = get_data_dir() / "labeling" / "data_as_images"
    dest_root.mkdir(parents=True, exist_ok=True)

    combined_index = {}

    for source_dir, target_batch in data_maps.items():
        batch_dir = dest_root / target_batch
        batch_dir.mkdir(parents=True, exist_ok=True)

        batch_index = {}
        dcm_files = sorted(get_dcm_files(source_dir))
        print(f"Found {len(dcm_files)} DCM files for extraction at {source_dir} -> {target_batch}.")

        for dcm_file in tqdm(dcm_files, desc=f"Extracting {target_batch}"):
            file_name = Path(dcm_file).stem
            cleaned_name = clean_name(file_name)
            dcm_key = str(dcm_file)

            rel_path = f"{target_batch}/{cleaned_name}"
            file_dest_dir = batch_dir / cleaned_name

            # Skip if already cached
            # if file_dest_dir.exists() and len(os.listdir(file_dest_dir)) > 0:
            #     batch_index[dcm_key] = cleaned_name
            #     combined_index[dcm_key] = rel_path
            #     continue

            # try:
            frames = get_frames_digital_lab(dcm_file)
            # except Exception as e:
            #     continue

            if frames is None or len(frames) == 0:
                logging.warning(f"frames is either empty or None: {frames}")
                continue

            file_dest_dir.mkdir(parents=True, exist_ok=True)
            batch_index[dcm_key] = cleaned_name
            combined_index[dcm_key] = rel_path

            for i, frame in enumerate(frames):
                frame_idx = i + 1
                if frame.dtype != np.uint16:
                    frame = frame.astype(np.uint16)

                try:
                    # Strictly using leveling preprocessing
                    proc_frame_u8 = leveling_defect_filter_u8(frame)
                except Exception as e:
                    print(f"Error preprocessing {file_name} frame {frame_idx}: {e}")
                    continue

                save_path = file_dest_dir / f"frame_{frame_idx:03d}.png"
                imwrite_unicode(str(save_path), proc_frame_u8)

        batch_index_path = batch_dir / "index.json"
        with open(batch_index_path, "w", encoding="utf-8") as f:
            json.dump(batch_index, f, indent=4, ensure_ascii=False)
        print(f"Batch '{target_batch}' complete. Index saved to {batch_index_path}")

    combined_index_path = dest_root / "index.json"
    with open(combined_index_path, "w", encoding="utf-8") as f:
        json.dump(combined_index, f, indent=4, ensure_ascii=False)

    print(f"Extraction complete. Combined index saved to {combined_index_path}")
    return combined_index

if __name__ == "__main__":
    extract_and_index_data(location="remote")
    # extract_and_index_data(location="local")
