"""
Basel Face Model 2019 module, a child class of the BFM module.

Example:

    1.initialize the BFM model parameters
        from .load_bfm import load_BFM
        model=load_BFM(MODEL_PATH, device=torch.device('cuda'))

    2.initialize a batch of Face_19.Face19 objects.
        face_batch=Face19(model, num_faces=64, is_shape_random=True, is_expression_random=True, is_texture_random=True, device=torch.device('cuda'))

    3.render and save the images
        face_tensors=face_batch.render_face(imsize=512,im_path=IMAGE_ROOT_DIR)
"""

from typing import Union, Optional, Tuple, List
import os
import sys
import math
from pathlib import Path

# add project root path to pythonpath
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import torch
import numpy as np
import pandas as pd
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from torchvision.utils import save_image
from utils import chunks
from differentiable_faces.load_bfm import load_BFM
from differentiable_faces.BFM_obj import BFM
from differentiable_faces.rendering import image_renderer, image_rendering


class Face19(BFM):
    """BFM 2019"""

    def __init__(
        self,
        model: dict,
        num_faces: int,
        is_shape_random: bool,
        is_expression_random: bool,
        is_texture_random: bool,
        is_angle_random: bool,
        angles_list: Optional[Union[list, torch.Tensor]] = None,
        is_lighting_random: bool = False,
        light_direction: Union[list, torch.Tensor] = [0, 0, 10],
        light_intensity: Union[list, torch.Tensor] = [0.6, 0.6, 0.6],
        is_background_random: bool = False,
        background_color: tuple = (0.5, 0.5, 0.5),
        camera="FoVPerspectiveCameras",
        fov: float = 15.762,
        scale: float = 1.0,
        translation: Union[list, tuple] = [0, -15, 0],
        seed: Optional[int] = None,
        device: str = None,
        grayscale: bool = False,
        rgba: bool = False,
    ):
        """
        Args:
            model (dict):
                A dictionary of model parameters returned by using load_BFM function in load_bfm.py.
            num_faces (int):
                Number of faces to be generated
            is_shape_random (bool):
                If True, sample random shape latents from Normal(0,1) to generate random shape map.
            is_expression_random (bool):
                If True, sample random expression latents from Normal(0,1) to generate random shape map.
            is_texture_random (bool):
                If True, sample random texture latents from Normal(0,1) to generate random texture map.
            is_angle_random (bool):
                If True, generate random euler angles (in radians) for head orientations of each face from truncated normal distributions.
                Distribution parameters were set by heuristic normal human head turn range.
            angles_list (:obj:`list`, or torch.Tensor, optional):
                (N, 3) user-defined angle vectors. A list of 3-dim angles in radians that has shape (num_face,3).
            is_lighting_random (bool):
                If True, sample random ambient light intensity and light direction vectors.
                Defaults to False.
            light_direction (:obj:`list`, or torch.Tensor):
                (1,3) or (N,3) user-defined light_direction vector.
                Defaults to [0,0,10], directional light on the z axis (the front of the face).
            light_intensity: (:obj:`list`, or torch.Tensor):
                (1,3) or (N,3) user-defined light_direction vector.
                Intensity values should be between (0,1), Defaults to [0.6,0.6,0.6].
            is_background_random (bool):
                If True, sample different background color (different degrees of gray) for each face.
            background_color (tuple):
                User-defined uniform background color. Defaults to (0.5,0.5,0.5).
            fov (float): field of view angle of the camera.
            scale (float):
                Scaling factor that adjusts shape coordinates and subsequently influences visual size of the face on the rendered images.
                Defaults to 0.9.
            translation(float)
            seed (:obj:`int`):
                Fixed seed integer for generating random face latents.
            device (str):
                Torch device. If None, set to torch.device('cuda') if GPU is available, otherwise use CPU.
            grayscale (bool):
                If True, turn colored images into grayscale images.

        """

        if device is None:
            if torch.cuda.is_available():
                if num_faces < torch.cuda.device_count():
                    device = [
                        torch.device(f"cuda:{i_gpu}") for i_gpu in range(num_faces)
                    ]
                else:
                    device = [
                        torch.device(f"cuda:{i_gpu}")
                        for i_gpu in range(torch.cuda.device_count())
                    ]
            else:
                device = [torch.device("cpu")]
                print(
                    "PyTorch3d rendering with CPU is slow, consider switching to GPUs"
                )

        print(f"Device: {device}")
        self.device = device
        self.batch = len(self.device)
        batch_size = num_faces // self.batch
        assert batch_size >= 1
        self.batch_size = [batch_size] * (self.batch - 1) + [
            num_faces - batch_size * (self.batch - 1)
        ]
        self.cumsum = [0] + np.cumsum(self.batch_size).tolist()
        print(self.batch_size, self.cumsum)

        super().__init__(
            model,
            num_faces,
            scale=scale,
            translation=translation,
            seed=seed,
            device=self.device,
        )

        self.num_faces = num_faces
        self.grayscale = grayscale
        self.rgba = rgba
        self.background_color = background_color
        self.is_background_random = is_background_random
        self.camera = camera
        self.fov = fov

        self.latent_types = [
            "shape_coefs",
            "expr_coefs",
            "tex_coefs",
            "light_direction",
            "light_intensity",
            "angles",
        ]
        self.shape_coefs, self.expr_coefs = super().get_shape_coef(
            is_shape_random, is_expression_random
        )
        self.tex_coefs = super().get_tex_coef(is_texture_random)

        if angles_list is None:
            self.angles = super().get_angle(is_angle_random)
        else:
            if torch.is_tensor(angles_list) is False:
                angles = torch.tensor(
                    angles_list, dtype=torch.float32, device=self.device[0]
                )
            else:
                angles = angles_list.to(self.device[0])
            self.angles = angles
            if angles.ndim != 2 or angles.shape != (self.num_faces, 3):
                msg = "Expected euler angles to have shape (num_faces, 3); got %r"
                raise ValueError(msg % repr(angles.shape))

        if is_lighting_random == True:
            self.light_direction, self.light_intensity = super().get_lighting_coef()

        else:
            if torch.is_tensor(light_direction) is False:
                light_direction = torch.tensor(
                    light_direction, dtype=torch.float32, device=self.device[0]
                ).reshape(-1, 3)
            else:
                light_direction = light_direction.to(self.device[0])
            if torch.is_tensor(light_intensity) is False:
                light_intensity = torch.tensor(
                    light_intensity, dtype=torch.float32
                ).reshape(-1, 3)
            else:
                light_intensity = light_intensity.to(self.device[0])
            if light_direction.ndim != 2 or light_direction.shape[1] != 3:
                msg = "Expected light direction to have shape (N, 3); got %r"
                raise ValueError(msg % repr(light_direction.shape))
            if light_intensity.ndim != 2 or light_intensity.shape[1] != 3:
                msg = "Expected light intensity to have shape (N, 3); got %r"
                raise ValueError(msg % repr(light_intensity.shape))

            self.light_direction = light_direction
            self.light_intensity = light_intensity

    def generate_maps(self):
        """Generate shape and texture map.

        Returns:
            mesh_batch (pytorch3d.structures.meshes):
                A batch of face-vertex mesh objects consisting of shape map, texture map, and faces. batch size N.
        """

        triangles = self.triangles
        shape_map = super().get_shape_map_19(
            self.shape_coefs, self.expr_coefs, self.angles
        )  # [N, 58203, 3]
        tex_map = super().get_texture_map(self.tex_coefs)  # [N, 58203, 3]

        mesh_batch = []
        for i_batch, (batch_size, device) in enumerate(
            zip(self.batch_size, self.device)
        ):
            # print(f"{batch_size}, {triangles.shape}")
            faces_list = torch.stack([triangles] * batch_size, 0).to(
                device
            )  # [N, 116160, 3]
            textures_list = TexturesVertex(
                tex_map[self.cumsum[i_batch] : self.cumsum[i_batch + 1]].to(device)
            )
            mesh_batch.append(
                Meshes(
                    verts=shape_map[self.cumsum[i_batch] : self.cumsum[i_batch + 1]].to(
                        device
                    ),
                    faces=faces_list,
                    textures=textures_list,
                )
            )

        return mesh_batch

    def render_face(
        self,
        cam_dist=1200.0,
        im_dir: str = None,
        im_filenames: range = None,
        imsize: Union[int, Tuple[int, int]] = 256,
        binsize=None,
        rgba=False,
    ):
        """Render face.

        Args:
            cam_dist (float):
                default works for realistic face size.
                increase this argument if expecting face size to be unrealistically big (large first PC coef)
            im_dir (str):
                root directory for saving the images.
            imsize (int):
                Size in pixels of the output image to be rasterized.
                Can optionally be a tuple of (H, W) in the case of non square images.

        Returns:
            ims (torch.Tensor):
                (N,3,H,W). Batch size N. A batch of RGB images (rendered face meshes).
        """

        mesh_batch = self.generate_maps()
        ims = []
        for i_batch, (mesh, device) in enumerate(zip(mesh_batch, self.device)):
            if self.light_direction.shape[0] > 1:
                light_direction = self.light_direction[
                    self.cumsum[i_batch] : self.cumsum[i_batch + 1]
                ].to(device)
            else:
                light_direction = self.light_direction

            if self.light_intensity.shape[0] > 1:
                light_intensity = self.light_intensity[
                    self.cumsum[i_batch] : self.cumsum[i_batch + 1]
                ].to(device)
            else:
                light_intensity = self.light_intensity

            renderer = image_renderer(
                cam_dist=cam_dist,
                imsize=imsize,
                light_direction=light_direction,
                ambient_color=light_intensity,
                background=self.background_color,
                device=device,
                binsize=binsize,
                fov=self.fov,
                camera=self.camera,
            )
            batch_ims = image_rendering(
                renderer,
                mesh,
                background_color=self.background_color,
                is_background_random=self.is_background_random,
                device=device,
                rgba=rgba,
            ).to(self.device[0])
            # print(batch_ims.shape)
            ims.append(batch_ims)
        ims = torch.vstack(ims)
        # print(ims.shape)

        if self.grayscale == True:
            ims = (
                ims[..., 0] * 0.2126 + ims[..., 1] * 0.7152 + ims[..., 2] * 0.0722
            )  # conversion to gray images
            ims = torch.stack((ims,) * 3, axis=-1)

        ims = ims.permute(0, 3, 1, 2)

        if im_dir is not None:
            Path(im_dir).mkdir(parents=True, exist_ok=True)

            filenames = im_filenames if im_filenames is not None else range(len(ims))
            start = filenames[0]
            for i in filenames:
                path = os.path.join(im_dir, str(i) + ".png")
                save_image(ims[i - start, ...], path)

        return ims

    def save_face(self, out_path):
        latents = dict()
        for latent_type in self.latent_types:
            if getattr(self, latent_type) is not None:
                latents[latent_type] = (
                    getattr(self, latent_type).clone().detach().cpu().numpy()
                )
        np.savez(out_path, **latents)

    def face_log_probability(self):
        """evaluate the log-likelihood of the latents according to the shape, expression and shape components (i.e., p(\theta) )"""

        log_probability = dict()
        components = ["shape_coefs", "tex_coefs", "expr_coefs"]

        for component in components:
            if getattr(self, component) is not None:
                coefs = getattr(self, component)
                if coefs is not None:
                    k = int(coefs.shape[1])
                    log_probability[component] = (
                        -1
                        / 2
                        * (k * math.log(2 * math.pi) + torch.sum(coefs**2, axis=1))
                    )

                    # # an alternative implementation
                    # mvn = torch.distributions.MultivariateNormal(torch.zeros(k,device=coefs.device), torch.eye(k, device=coefs.device))
                    # log_probability2 = mvn.log_prob(coefs)
                    # assert torch.all(torch.isclose(log_probability[component],log_probability2))
        return log_probability


def load_face(model, device, saved_path, **kwargs):
    """load saved faces"""
    saved_latents = np.load(saved_path)
    latent_types = saved_latents.files
    num_faces = saved_latents[latent_types[0]].shape[0]
    face_batch = Face19(
        model,
        num_faces=num_faces,
        is_shape_random=False,
        is_expression_random=False,
        is_texture_random=False,
        is_angle_random=False,
        is_lighting_random=False,
        seed=None,
        **kwargs,
    )

    for latent_type in latent_types:
        setattr(
            face_batch,
            latent_type,
            torch.tensor(saved_latents[latent_type], device=device[0]),
        )
    return face_batch


def get_im_stats(
    bfm_model_path: str,
    imsize: int = 471,
    batch_size: int = 20,
    num_samples: int = 500,
    device=None,
    current_scale: float = 1.0,
    target_height: float = None,
):
    """Get face statistics; scale to target face height.

    Args:
        bfm_model_path(str):
            BFM loading path.
        imsize(int)
        batch_size(int):
            batch size for face sampling; depending on CUDA memory, a large batch size can lead to OOM error.
        num_samples(int):
            number of samples to get face statistics
        device:
            CPU device leads to slow rendering.
        current_scale(float)
        target_height(float):
            if given target height in pixels, calculate the rough scale so the faces generated have the same mean height in pixels.
            note that pixel height depends on image size, so the 'imsize' argument needs to match with target image size.
    """

    new_scale = None
    model = load_BFM(bfm_model_path, device=device)
    face_tensors = []
    for chunk in chunks(range(num_samples), batch_size):
        face_batch = Face19(
            model,
            batch_size,
            True,
            True,
            True,
            scale=current_scale,
            is_angle_random=False,
            is_lighting_random=True,
            seed=None,
            device=device,
            is_background_random=False,
        )
        face_tensor = face_batch.render_face(imsize=imsize)
        face_tensors.append(face_tensor)
    face_tensors = torch.cat(face_tensors)

    im_stats = {}
    im_stats["im_W"], im_stats["im_H"] = [face_tensors.shape[1]] * num_samples, [
        face_tensors.shape[2]
    ] * num_samples
    ims = (
        face_tensors.permute(0, 2, 3, 1)
        != torch.tensor(face_batch.background_color, device=face_batch.device)
    ).all(3)

    h = torch.any(ims, keepdim=False, axis=2)
    w = torch.any(ims, keepdim=False, axis=1)

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

    im_stats["bb_W"] = (w_pts[:, 1] - w_pts[:, 0]).float().cpu()
    im_stats["bb_H"] = (h_pts[:, 1] - h_pts[:, 0]).float().cpu()

    if current_scale and target_height:
        current_height = im_stats["bb_H"].mean().item()
        new_scale = current_scale * (target_height / current_height)

    face_stats = pd.DataFrame(im_stats)
    face_summary_stats = pd.Series(
        {
            "width_mean": im_stats["bb_W"].mean().item(),
            "width_std": im_stats["bb_W"].std().item(),
            "height_mean": im_stats["bb_H"].mean().item(),
            "height_std": im_stats["bb_H"].std().item(),
        }
    )

    return face_stats, face_summary_stats, new_scale


def mean_squared_coefs_from_saved_latents(latent_paths_list):
    """Calculate the average squared coefs of each type of face latents (shape/expression/texture) from saved latent npz files.

    Args:
        latent_paths_list (:obj:`list`)

    Returns:
        mean_squared_latents (dict):
            a dictionary that contains the mean squared latents of each type of coefficients for all
    """
    latent_types = ["shape_coefs", "tex_coefs", "expr_coefs"]
    mean_squared_latents = dict.fromkeys(latent_types, 0)
    total_num_faces = dict.fromkeys(latent_types, 0)
    num_batches = len(latent_paths_list)

    for i_batch, latent_path in enumerate(latent_paths_list):
        saved_latents = np.load(latent_path)
        saved_latent_types = saved_latents.files
        latent_type_set = list(set(saved_latent_types) & set(latent_types))

        for latent_type in latent_type_set:
            total_num_faces[latent_type] += saved_latents[latent_type_set[0]].shape[0]
            sum_squared_latent_per_face = np.sum(
                saved_latents[latent_type] ** 2, axis=1
            )  # (num_faces,)
            mean_squared_latents[latent_type] += sum_squared_latent_per_face.sum()

            if i_batch == num_batches - 1:
                mean_squared_latents[latent_type] = (
                    mean_squared_latents[latent_type] / total_num_faces[latent_type]
                )

    print(mean_squared_latents)

    return mean_squared_latents


def mean_squared_coefs_from_face_batch(face_batch):
    """Calculate the mean squared coefs of each type of face latents (shape/expression/texture) in a batch of faces.
    Args:
        face_batch:
            a batch of Face19 objects.
    Returns:
        mean_squared_latents (dict):
            a dictionary that contains the mean squared latents of each type of coefficients.
    """
    mean_squared_latents = dict()
    components = ["shape_coefs", "tex_coefs", "expr_coefs"]
    num_faces = face_batch.num_faces

    for component in components:
        if getattr(face_batch, component) is not None:
            coefs = getattr(face_batch, component).detach().cpu().numpy()
            sum_squared_latents_per_face = np.sum(coefs**2, axis=1)
            mean_coef = sum_squared_latents_per_face.sum() / num_faces
            mean_squared_latents[component] = mean_coef
    print(mean_squared_latents)

    return mean_squared_latents


def scale_coefs(current_batch, target):
    """In-place scaling of each type of face coefficients in the current batch of faces,
    where the mean squared latents matches with the target batch.

    Args:
        current_batch:
            current batch of Face19 objects.
        target:
            a target batch of Face19 objects, or a list of saved target latents paths.
    """
    cur_squared_latents = mean_squared_coefs_from_face_batch(current_batch)
    if type(target) == list:
        target_squared_latents = mean_squared_coefs_from_saved_latents(target)
    else:
        target_squared_latents = mean_squared_coefs_from_face_batch(target)

    for component in cur_squared_latents.keys():
        if component in target_squared_latents.keys():
            scale = np.sqrt(
                target_squared_latents[component] / cur_squared_latents[component]
            )
            attr = getattr(current_batch, component)
            attr *= scale
    cur_squared_latents = mean_squared_coefs_from_face_batch(
        current_batch
    )  # just to double check it's scaled to the right thing


def random_faces_scaled(
    num_faces: int,
    model_path: str,
    is_angle_random: bool = False,
    is_lighting_random: bool = False,
    is_background_random: bool = False,
    scale: float = 0.8,
    seed: Optional[int] = None,
    grayscale: bool = False,
    optimized_batch=None,
    device: str = None,
):
    """Generate a batch of random faces, with the mean squared latents of their coefficients scaled
    if another batch of face objects are given as target batch.

    Args:
        optimized_batch (Face_19.Face19, optional):
            If given, the coefficients of the random faces generated are scaled
            so the mean squared latents of each type of coefficients match with the mean squared latents in the optimized_batch.
        For all other function arguments, please refer to docstrings of Face_19.Face19 objects.

    Returns:
        random_faces (Face_19.Face19)
    """

    BFM_19 = load_BFM(model_path)
    random_faces = Face19(
        BFM_19,
        num_faces,
        True,
        True,
        True,
        is_angle_random,
        is_lighting_random=is_lighting_random,
        is_background_random=is_background_random,
        scale=scale,
        seed=seed,
        device=device,
        grayscale=grayscale,
    )

    if optimized_batch is not None:
        scale_coefs(random_faces, optimized_batch)

    return random_faces
