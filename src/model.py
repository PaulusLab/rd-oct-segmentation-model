"""BiomarkerSegFormer -- exact mirror of rrd-oct-pipeline/src/models/biomarker_unet.py.
Do not modify the architecture; it must match the trained checkpoint's state_dict keys."""

import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class BiomarkerSegFormer(nn.Module):
    """SegFormer-B4 encoder with a lightweight MLP decoder. Output is upsampled 4x
    (from H/4, W/4) to match input resolution."""

    def __init__(self, encoder="nvidia/mit-b4", in_channels=3, out_channels=5, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                encoder,
                num_labels=out_channels,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(encoder, num_labels=out_channels)
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
