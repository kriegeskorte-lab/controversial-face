import numpy as np
from torchvision.transforms import Normalize, Compose


def torchvision(im_tensor):  # previously, vgg16_transform
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize(im_tensor)


def vggface2_transform(im_tensor):
    normalize = Normalize(mean=[0.6068, 0.4517, 0.3800], std=[0.2492, 0.2173, 0.2082])
    return normalize(im_tensor)


def bfm_transform(im_tensor):
    normalize = Normalize(
        mean=[0.55537174, 0.50970546, 0.48330758],
        std=[0.28882495, 0.26824081, 0.26588868],
    )
    return normalize(im_tensor)


def bfm_inverse_transform(im_tensor):
    normalize = Compose(
        [
            Normalize(
                mean=[0, 0, 0], std=1 / np.asarray([0.28882495, 0.26824081, 0.26588868])
            ),
            Normalize(
                mean=-np.asarray([0.55537174, 0.50970546, 0.48330758]), std=[1, 1, 1]
            ),
        ]
    )
    return normalize(im_tensor)


preprocessing_funs = {
    "torchvision": torchvision,
    "vggface2_transform": vggface2_transform,
    "bfm_transform": bfm_transform,
}
