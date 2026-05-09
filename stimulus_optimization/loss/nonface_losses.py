"""Use face detector to classify face vs. nonface"""

import os, sys

# add project root path to pythonpath
sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))

import numpy as np
import torch
from torch.nn.functional import interpolate
from yolov5_face.models.experimental import attempt_load
from yolov5_face.utils.general import check_img_size
from utils import smoothmax


class NonFaceProbabilityLoss(torch.nn.Module):
    """some approximation of negative log face probability"""

    def __init__(
        self,
        detector_weight_path,
        classifier_weight_path,
        alpha=20,
        model_device="cpu",
        agg_func="mean",
    ):
        super(NonFaceProbabilityLoss, self).__init__()
        self.alpha = alpha
        self.model_device = model_device
        self.face_detector = attempt_load(
            detector_weight_path, map_location=model_device
        )
        classifier_weights = np.load(classifier_weight_path)
        self.coef = classifier_weights["coef"].item()
        self.intercept = classifier_weights["intercept"].item()
        if agg_func == "mean":
            self.agg_func = torch.mean
        elif agg_func == "max":
            self.agg_func = torch.max

    def forward(self, ims):
        # image range [0, 1]
        resize = check_img_size(ims.shape[-1], s=self.face_detector.stride.max())
        assert resize == ims.shape[-1]
        ims = interpolate(ims, size=32)  # reduce image size

        pred = self.face_detector(ims)[0]
        pred = pred[..., -1] * pred[..., 4]

        prob = smoothmax(pred, alpha=self.alpha, dim=1)
        prob = 1 / (1 + torch.exp(-(prob * self.coef + self.intercept)))

        return self.agg_func(-torch.log(prob))
