import numpy as np
from pathlib import Path
from inference import predict

FIXTURE = Path(__file__).parent / "fixtures" / "sample_bscan.png"
CHECKPOINT = Path(__file__).parent.parent / "checkpoints" / "biomarker_fold4_best.pth"


def test_predict_returns_expected_shapes():
    result = predict(str(FIXTURE), checkpoint_path=str(CHECKPOINT))
    assert result["mask"].shape == (512, 512)
    assert result["mask"].dtype == np.uint8
    assert result["mask"].max() <= 4
    assert result["confidence"].shape == (512, 512)
    assert 0.0 <= result["confidence"].min() and result["confidence"].max() <= 1.0
    assert set(result["class_pixel_pct"].keys()) == {"SRF", "ORC", "IRC", "ERM"}
    for pct in result["class_pixel_pct"].values():
        assert 0.0 <= pct <= 100.0
