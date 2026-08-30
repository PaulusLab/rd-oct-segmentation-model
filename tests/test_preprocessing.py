import numpy as np
from PIL import Image
from pathlib import Path
from src.preprocessing import percentile_normalize, center_crop_pad_2d

FIXTURE = Path(__file__).parent / "fixtures" / "sample_bscan.png"


def test_percentile_normalize_output_range():
    img = np.array(Image.open(FIXTURE).convert("L"), dtype=np.float32)
    normed = percentile_normalize(img)
    assert normed.dtype == np.float32
    assert normed.min() >= 0.0
    assert normed.max() <= 1.0
    # a real B-scan has real contrast -- normalized output shouldn't collapse to a constant
    assert normed.std() > 0.05


def test_center_crop_pad_2d_pads_smaller_image():
    small = np.ones((100, 80), dtype=np.float32)
    result = center_crop_pad_2d(small, (512, 512))
    assert result.shape == (512, 512)
    # original content should be centered, not at the edge
    assert result[256, 256] == 1.0
    assert result[0, 0] == 0.0


def test_center_crop_pad_2d_crops_larger_image():
    large = np.ones((600, 700), dtype=np.float32)
    result = center_crop_pad_2d(large, (512, 512))
    assert result.shape == (512, 512)
    assert (result == 1.0).all()  # fully inside the source, no padding needed
