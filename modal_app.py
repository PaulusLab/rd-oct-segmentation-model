"""Modal app serving RD OCT segmentation inference. Deploy with:
    modal deploy modal_app.py
Wraps inference.predict() with a color overlay renderer and per-class stats, exposed as a public
HTTP POST endpoint (FastAPI-based under the hood, CORS enabled by default so the tool page's
cross-origin fetch() call works without extra configuration)."""

import base64
import io
import os

import modal
from fastapi import File, HTTPException, UploadFile

app = modal.App("rd-oct-segmentation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.35", "numpy", "scipy", "pillow", "fastapi[standard]")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("inference.py", remote_path="/root/inference.py")
    .add_local_file(
        "checkpoints/biomarker_fold4_best.pth",
        remote_path="/root/checkpoints/biomarker_fold4_best.pth",
    )
)

# Fixed color scheme -- the tool page's legend/hover tooltips (Task 8) must use these
# exact colors so the overlay and legend agree visually.
CLASS_COLORS = {
    1: (239, 68, 68),    # SRF -- red
    2: (245, 158, 11),   # ORC -- amber
    3: (59, 130, 246),   # IRC -- blue
    4: (168, 85, 247),   # ERM -- purple
}

RELIABILITY = {
    "SRF": {"dice": 0.780, "tag": "High"},
    "IRC": {"dice": 0.515, "tag": "Moderate"},
    "ORC": {"dice": 0.369, "tag": "Experimental"},
    "ERM": {"dice": 0.348, "tag": "Experimental"},
}

# Global cache: populated on first call in a warm container, reused on subsequent
# calls to the same warm container -- avoids reloading the model every request.
_state = {}


def _get_predict():
    if "predict" not in _state:
        import sys
        sys.path.insert(0, "/root")
        os.chdir("/root")
        from inference import predict
        _state["predict"] = predict
    return _state["predict"]


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
async def segment(image: UploadFile = File(...)):
    import numpy as np
    from PIL import Image as PILImage

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="No image provided.")
    tmp_path = "/tmp/upload.png"
    with open(tmp_path, "wb") as f:
        f.write(contents)

    predict = _get_predict()
    result = predict(tmp_path, checkpoint_path="/root/checkpoints/biomarker_fold4_best.pth")

    overlay = np.zeros((*result["mask"].shape, 4), dtype=np.uint8)
    for class_idx, color in CLASS_COLORS.items():
        m = result["mask"] == class_idx
        overlay[m, 0] = color[0]
        overlay[m, 1] = color[1]
        overlay[m, 2] = color[2]
        overlay[m, 3] = 140

    overlay_img = PILImage.fromarray(overlay, mode="RGBA")
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    stats = {
        name: {
            "pixel_pct": result["class_pixel_pct"][name],
            "dice": RELIABILITY[name]["dice"],
            "reliability": RELIABILITY[name]["tag"],
        }
        for name in ["SRF", "ORC", "IRC", "ERM"]
    }

    return {"overlay_png_base64": overlay_b64, "stats": stats}
