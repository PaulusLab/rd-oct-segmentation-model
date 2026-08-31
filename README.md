# RD OCT Segmentation Model

SegFormer-B4 model segmenting four retinal detachment biomarkers from OCT B-scan images:
SRF (subretinal fluid), ORC (outer retinal cysts), IRC (intraretinal cysts), and ERM (epiretinal
membrane). Developed by the Paulus Lab, Johns Hopkins University.

**⚠️ Research prototype — not for clinical use.**

## Live demo

Try it at [pauluslab.github.io/tools/rd-oct-segmentation-tool.html](https://pauluslab.github.io/tools/rd-oct-segmentation-tool.html),
served by a [Modal](https://modal.com) serverless endpoint running the exact code in this repo.

## Performance

Measured on a 5-fold cross-validation split of 35 manually annotated cases (3D Slicer, single
annotator). Fold 4 (shipped here) had the best per-class Dice of the 5 trained folds:

| Class | Dice | Reliability |
|---|---:|---|
| SRF (Subretinal Fluid) | 0.780 | High |
| IRC (Intraretinal Cysts) | 0.515 | Moderate |
| ORC (Outer Retinal Cysts) | 0.369 | Experimental |
| ERM (Epiretinal Membrane) | 0.348 | Experimental |

ORC and ERM performance is not yet reliable — predictions for these two classes should be treated as
low-confidence starting points, not findings. This model is under active development via an
active-learning correction loop (predict → manually correct → retrain), so these numbers are expected
to improve; this README will be updated as new folds are trained.

## Usage

```python
from inference import predict

result = predict("path/to/bscan.png")
result["mask"]             # (512, 512) uint8, 0=BG 1=SRF 2=ORC 3=IRC 4=ERM
result["confidence"]       # (512, 512) float32, max-softmax per pixel
result["class_pixel_pct"]  # {"SRF": 2.1, "ORC": 0.0, "IRC": 0.4, "ERM": 0.0}
```

**Known limitation:** this model is 2.5D — it was trained on 3 adjacent B-scan slices as a 3-channel
input, not a single flat image. `predict()` duplicates the single input image into all 3 channels as
a simplification for single-image use, which does not give the model the neighboring-slice context it
was trained with. Expect somewhat lower accuracy on single images than the Dice scores above, which
were measured on true 3-slice volume input.

## Model architecture

`nvidia/mit-b4` (SegFormer-B4) encoder + lightweight MLP decoder, 5-class output upsampled to input
resolution. See `src/model.py`.
