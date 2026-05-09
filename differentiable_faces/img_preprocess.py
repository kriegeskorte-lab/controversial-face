"""Image preprocessing"""

from typing import Callable
import torch
from torchvision import transforms

from utils import zoom_crop


def crop_head_torch(
    face_tensors: torch.Tensor,
    extend: float = 0.1,
    newdim: int = 224,
    enable_jitter: bool = False,
    overflow_method="squeeze",
    crop_tightness="tight",
    bg_intensity=0.5,
    max_jitter=5,
    max_scale_jitter=0.05,
):
    """Detect boundaries and crop synthetic heads presented on a uniform gray background.

    Args:
        face_tensors (torch.Tensor):
            (N, 3, H, W)
        extend (float):
            Expand the face boundaries by some factor.
            Specifically, expand the total sum of width and height of the tightly cropped face.
        newdim (int):
            Resize the cropped images to have image size (newdim,newdim)
        enable_jitter (float):
            If True, randomly reduce or expand the boundaries slightly.
        overflow_method (str) 'squeeze'|'expand':
            when the calculated borders go beyond the image bounderies,
            whether to squeeze the crop to fit or to expand the background by gray pixels.
        crop_tightness (str) 'tight'|'loose'|'nocrop'
            'tight' crops according to (width + height)/2
            'loose' crops according to max(width,height)
            'nocrop' just downsizes the image. extend, and overflow_method are ignored.
        bg_intensity (float)
        max_jitter (int) maximum amount of jitter to apply to the crop, in pixels
        max_scale_jitter (float) scale jitter in fraction 0.5 is +-5% scale jitter

    """

    device = face_tensors.device

    N, C, H, W = face_tensors.shape

    # figure out head location for adaptive cropping methods
    if crop_tightness in ["loose", "tight"]:

        # locate head
        # just look at R
        if face_tensors.ndim == 3:
            face_tensors = face_tensors.unsqueeze(0)

        imgs = face_tensors[:, 0, :, :] != bg_intensity
        # print(imgs[1,0,:,:])
        max_x = imgs.shape[2]
        max_y = imgs.shape[1]

        h = torch.any(imgs, keepdim=False, axis=2)
        w = torch.any(imgs, keepdim=False, axis=1)

        def first_nonzero_along_dim(x, dim):
            """return the first nonzero element along dim from a tensor"""
            BIG = torch.tensor(torch.iinfo(torch.int64).max, device=x.device)
            return torch.where(
                x != 0, torch.ones_like(x, dtype=torch.int64).cumsum(dim=dim) - 1, BIG
            ).min(dim=dim)[0]

        def last_nonzero_along_dim(x, dim):
            """return the last nonzero element along dim from a tensor"""
            SMALL = torch.tensor(torch.iinfo(torch.int64).min, device=x.device)
            return torch.where(
                x != 0, torch.ones_like(x, dtype=torch.int64).cumsum(dim=dim) - 1, SMALL
            ).max(dim=dim)[0]

        h_pts = torch.stack(
            [first_nonzero_along_dim(h, dim=1), last_nonzero_along_dim(h, dim=1)], dim=1
        )
        w_pts = torch.stack(
            [first_nonzero_along_dim(w, dim=1), last_nonzero_along_dim(w, dim=1)], dim=1
        )

        # print('height points')
        # print(h_pts)
        # print('width points')
        # print(w_pts)

        # bounding box
        width = w_pts[:, 1] - w_pts[:, 0]
        height = h_pts[:, 1] - h_pts[:, 0]
        # print(width,height)

        if crop_tightness == "tight":
            length = torch.true_divide(width + height, 2)
        elif crop_tightness == "loose":
            length = torch.maximum(width, height)
        else:
            raise ValueError("unsupported crop_tightness value.")
        half_length = length / 2

        x_center = w_pts[:, 0] + torch.true_divide(width, 2)
        y_center = h_pts[:, 0] + torch.true_divide(height, 2)
        # print(x_center, y_center)
        # print(length)
    elif crop_tightness == "nocrop":
        x_center = torch.tensor([(W - 1) / 2] * N, device=device)
        y_center = torch.tensor(
            [(H - 1) / 2] * N,
            device=device,
        )
        half_length = torch.tensor([max((H - 1) / 2, (W - 1) / 2)] * N, device=device)
    else:
        raise ValueError

    if enable_jitter and max_scale_jitter > 0:
        scale_jitter = (
            torch.distributions.uniform.Uniform(
                1 - max_scale_jitter, 1 + max_scale_jitter
            )
            .sample((N,))
            .to(device)
        )
    else:
        scale_jitter = 1

    x1 = x_center - (1 + extend) * half_length * scale_jitter
    y1 = y_center - (1 + extend) * half_length * scale_jitter
    x2 = x_center + (1 + extend) * half_length * scale_jitter
    y2 = y_center + (1 + extend) * half_length * scale_jitter

    if enable_jitter:
        x_jitter = torch.randint(low=-max_jitter, high=max_jitter + 1, size=(N,)).to(
            device
        )
        y_jitter = torch.randint(low=-max_jitter, high=max_jitter + 1, size=(N,)).to(
            device
        )
        x1 += x_jitter
        x2 += x_jitter
        y1 += y_jitter
        y2 += y_jitter

    # print('crop dimensions:', x1[0].item(),x2[0].item(),y1[0].item(),y2[0].item())

    if overflow_method == "squeeze":
        x1 = torch.clamp(x1, min=0).round().int()
        y1 = torch.clamp(y1, min=0).round().int()
        x2 = torch.clamp(x2, max=max_x).round().int()
        y2 = torch.clamp(y2, max=max_y).round().int()

        cropped = []
        for i in range(N):
            crop = face_tensors[i, :, y1[i] : y2[i], x1[i] : x2[i]]
            crop = crop.unsqueeze(0)
            crop = torch.nn.functional.interpolate(
                crop, size=newdim, mode="bilinear", align_corners=True
            )
            cropped.append(crop)

        face_cropped = torch.stack(cropped, dim=0)
        face_cropped = face_cropped.squeeze()
    elif overflow_method == "expand":
        # the image is shifted by 0.5 so the padding would introduce middle gray pixels
        face_cropped = (
            zoom_crop(
                face_tensors - bg_intensity,
                x1,
                x2,
                y1,
                y2,
                output_H=newdim,
                output_W=newdim,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            + bg_intensity
        )
    else:
        raise ValueError("Unsupported overflow_method value.")

    return face_cropped
