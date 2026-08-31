"""Single-image inference for the RD OCT segmentation model.

Loads a B-scan image, preprocesses it exactly like training (percentile-normalize,
center-crop/pad to 512x512), duplicates it into the 3-channel input the 2.5D model
expects (a known simplification -- the model was trained on 3 genuine adjacent
B-scans, not one repeated slice; documented in this repo's README, not hidden)."""

import numpy as np
import torch
from PIL import Image

from src.model import BiomarkerSegFormer
from src.preprocessing import percentile_normalize, center_crop_pad_2d

LABEL_NAMES = ["BG", "SRF", "ORC", "IRC", "ERM"]
INPUT_SIZE = (512, 512)

_model_cache = {}


def _load_model(checkpoint_path: str, device: torch.device) -> BiomarkerSegFormer:
    if checkpoint_path in _model_cache:
        return _model_cache[checkpoint_path]

    model = BiomarkerSegFormer(
        encoder="nvidia/mit-b4", in_channels=3, out_channels=5, pretrained=False
    ).to(device)
    # weights_only=False: PyTorch >=2.6 defaults torch.load to weights_only=True, which
    # rejects this checkpoint's non-tensor metadata (e.g. numpy scalars in the optimizer
    # state / epoch fields saved alongside model_state_dict). This checkpoint is a trusted,
    # locally-downloaded lab artifact (not loaded from an untrusted/remote source), so
    # disabling weights-only unpickling here is safe.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    _model_cache[checkpoint_path] = model
    return model


def predict(image_path: str, checkpoint_path: str = "checkpoints/biomarker_fold4_best.pth") -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)

    img = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
    img = percentile_normalize(img)
    img = center_crop_pad_2d(img, INPUT_SIZE)

    stack = np.stack([img, img, img], axis=0)  # (3, H, W) -- single slice duplicated 3x
    input_tensor = torch.from_numpy(stack).unsqueeze(0).float().to(device)  # (1, 3, H, W)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = probs.max(dim=1)

    mask = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    conf_map = confidence.squeeze(0).cpu().numpy().astype(np.float32)

    total_pixels = mask.size
    class_pixel_pct = {
        LABEL_NAMES[c]: round(100.0 * (mask == c).sum() / total_pixels, 2)
        for c in range(1, 5)
    }

    return {"mask": mask, "confidence": conf_map, "class_pixel_pct": class_pixel_pct}
