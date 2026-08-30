"""Preprocessing for single-image OCT inference -- mirrors the training pipeline's
percentile_normalize exactly (rrd-oct-pipeline/src/preprocessing.py), and adapts
center_crop_pad from 3D volumes to a single 2D image (no Z axis here)."""

import numpy as np


def percentile_normalize(data: np.ndarray, p_low: int = 1, p_high: int = 99) -> np.ndarray:
    """Clip intensity to percentiles and scale to [0, 1]. Identical to the training
    pipeline's version -- do not change without also changing the training pipeline,
    or inference input distribution will no longer match what the model was trained on."""
    low, high = np.percentile(data, (p_low, p_high))
    data = np.clip(data, low, high)
    data = (data - low) / (high - low + 1e-8)
    return data.astype(np.float32)


def center_crop_pad_2d(data: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Crop or pad a single 2D image to target_shape, centered. 2D analog of the
    training pipeline's center_crop_pad (which operates on 3D volumes with a Z axis
    we don't have for a single uploaded image)."""
    curr_shape = data.shape
    new_data = np.zeros(target_shape, dtype=data.dtype)

    h_src_start = max(0, (curr_shape[0] - target_shape[0]) // 2)
    h_src_end = min(curr_shape[0], h_src_start + target_shape[0])
    h_dst_start = max(0, (target_shape[0] - curr_shape[0]) // 2)
    h_dst_end = min(target_shape[0], h_dst_start + (h_src_end - h_src_start))

    w_src_start = max(0, (curr_shape[1] - target_shape[1]) // 2)
    w_src_end = min(curr_shape[1], w_src_start + target_shape[1])
    w_dst_start = max(0, (target_shape[1] - curr_shape[1]) // 2)
    w_dst_end = min(target_shape[1], w_dst_start + (w_src_end - w_src_start))

    new_data[h_dst_start:h_dst_end, w_dst_start:w_dst_end] = \
        data[h_src_start:h_src_end, w_src_start:w_src_end]

    return new_data
