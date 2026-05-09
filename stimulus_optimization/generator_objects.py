"""Define face generator object."""

import os, sys

# add project root path to pythonpath
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from pathlib import Path
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torchvision.utils import save_image

from differentiable_faces.load_bfm import load_BFM
from differentiable_faces.Face_19 import Face19
from differentiable_faces.img_preprocess import crop_head_torch
from stimulus_optimization.contrast_fixer import IntensityStatsFixer
from torchvision.transforms import CenterCrop
from utils import constraint_factory, chunks, jitter_images


HF_MODEL_REPO_ID = "wenx-guo/controversial-face-model-checkpoints"
STYLEGAN3_FILENAME = "stylegan3-r-ffhq_128.pkl"
DEFAULT_CHECKPOINT_DIR = "model_checkpoints"
REPO_ROOT = Path(__file__).resolve().parents[1]


class FACE_GENERATOR:
    def __init__(self, cfg, devices):
        self.cfg = cfg
        self.devices = devices
        self.optimized_latent_types = cfg.coefs

        if cfg.fix_intensity_stats:
            self.contrast_fixer = IntensityStatsFixer(
                max_steps=20,
                lr=1.0,
                target_mean_intensity=128 / 255,
                target_std_intensity=cfg.target_std_intensity / 255,
            )
        else:
            self.contrast_fixer = None

    def set_up_generator(self):
        raise NotImplementedError

    def load_latents(self):
        raise NotImplementedError

    def initialize_constraints(self):
        raise NotImplementedError

    def set_up_optimization_params(self):
        raise NotImplementedError

    def step_optimization(self):
        raise NotImplementedError

    def generate_images(self):
        raise NotImplementedError

    def jitter_and_crop(self):
        raise NotImplementedError

    def _noise_images(self, transformed_face_images, enable_noise):
        if self.cfg.grayscale:
            transformed_face_images = rgb_to_grayscale(
                transformed_face_images,
                contrast_fixer=self.contrast_fixer,
                bg_intensity=self.cfg.bg_intensity,
            )

        if enable_noise:
            if (
                "gaussian_noise_sigma" in self.cfg
                and self.cfg.gaussian_noise_sigma is not None
                and self.cfg.gaussian_noise_sigma > 0
            ):
                transformed_face_images[:, :3, :, :] += (
                    torch.randn_like(transformed_face_images[:, :3, :, :])
                    * self.cfg.gaussian_noise_sigma
                )
                transformed_face_images[:, :3, :, :] = torch.clamp(
                    transformed_face_images[:, :3, :, :], 0, 1
                )
        return transformed_face_images

    def save_high_quality_images(self, im_path, im_set_fname=None, rgba=False):
        """Render the face batch and optionally save the images to disk.
        This function produces stimuli in their presentation form (i.e., high resolution, no jittering).
        """
        if im_set_fname is not None:
            im_dir = os.path.join(im_path, im_set_fname)
        else:
            im_dir = None

        face_images = self.generate_images(
            im_dir=im_dir, imsize=self.cfg.saving_imsize, rgba=rgba
        )
        face_images = self.jitter_and_crop(
            face_images, self.cfg.saving_imsize, enable_noise=False
        )
        if self.cfg.grayscale:
            if rgba:
                face_images = rgba_to_grayscale(
                    face_images, contrast_fixer=self.contrast_fixer
                )
            else:
                face_images = rgb_to_grayscale(
                    face_images,
                    contrast_fixer=self.contrast_fixer,
                    bg_intensity=self.cfg.bg_intensity,
                )

        if im_set_fname is not None:
            for i_face, face_image in enumerate(face_images):
                if (
                    "file_naming_scheme" not in self.cfg
                    or self.cfg.file_naming_scheme == "single_file"
                ):
                    fname = im_set_fname + str(i_face) + "_cropped.png"
                elif self.cfg.file_naming_scheme == "pairwise":
                    i_pair = i_face // 2
                    i_face_in_pair = i_face % 2
                    fname = (
                        f"{im_set_fname}_pair_{i_pair:03}_{i_face_in_pair}_cropped.png"
                    )
                elif self.cfg.file_naming_scheme == "pairwise_trial":
                    assert (
                        "pairs_per_correlation" in self.cfg
                        and self.cfg.pairs_per_correlation is not None
                    )
                    i_pair = i_face // 2
                    i_face_in_pair = i_face % 2
                    i_trial = i_pair // self.cfg.pairs_per_correlation
                    i_pair_in_trial = i_pair % self.cfg.pairs_per_correlation
                    fname = f"{im_set_fname}_trial_{i_trial:02}_pair_{i_pair_in_trial:02}_{i_face_in_pair}_cropped.png"
                else:
                    raise ValueError(
                        f"unknown file naming scheme: {self.cfg.file_naming_scheme}"
                    )
                Path(im_dir + "_cropped").mkdir(parents=True, exist_ok=True)
                save_image(
                    face_image,
                    os.path.join(im_dir + "_cropped", fname),
                )

        return face_images

    def save_latents(self):
        raise NotImplementedError

    def save_images(self):
        raise NotImplementedError


class ModelParallel(torch.nn.Module):
    def __init__(self, model, devices, output_device):
        super(ModelParallel, self).__init__()
        self.num_replicas = len(devices)
        self.devices = devices
        self.output_device = output_device

        self.model_replicas = []
        for device in devices:
            model_replica = copy.deepcopy(model).to(device)
            self.model_replicas.append(model_replica)

    def forward(self, input, *args, **kwargs):
        out = []
        if isinstance(input, dict):
            input_size = len(input[list(input.keys())[0]])
            batch_size = input_size // self.num_replicas + (
                input_size % self.num_replicas != 0
            )
            for i_replica in range(self.num_replicas):
                input_batch = dict()
                for key, value in input.items():
                    input_batch[key] = value[
                        i_replica * batch_size : (i_replica + 1) * batch_size
                    ].to(self.devices[i_replica])
                out.append(
                    self.model_replicas[i_replica](
                        None, input_batch, *args, **kwargs
                    ).to(self.output_device)
                )
        else:
            batch_size = len(input) // self.num_replicas + (
                len(input) % self.num_replicas != 0
            )
            for i_replica, input_batch in enumerate(chunks(input, batch_size)):
                input_batch = input_batch.to(self.devices[i_replica])
                out.append(
                    self.model_replicas[i_replica](input_batch, *args, **kwargs).to(
                        self.output_device
                    )
                )
        out = torch.vstack(out)
        return out


class BFMGenerator(FACE_GENERATOR):
    def __init__(self, cfg, devices):
        super().__init__(cfg, devices)

    def set_up_generator(self, **kwargs):
        basel_face_model = load_BFM(self.cfg.generator_path, device=self.devices[0])

        # get eigenvalues for each latent type if optimize the coordinate space
        latent_types = [
            latent_type.split("_")[0] for latent_type in self.optimized_latent_types
        ]
        if getattr(self.cfg, "optimize_coordinate_space", False):
            self.EV = {
                f"{latent_type}_coefs": torch.sqrt(basel_face_model[f"{latent_type}EV"])
                for latent_type in latent_types
            }
            self.eps = 1e-2
        else:
            self.EV = {f"{latent_type}_coefs": None for latent_type in latent_types}
            self.eps = 1e-3

        self.generator = Face19(
            basel_face_model,
            self.cfg.nFaces,
            is_shape_random=True,
            is_expression_random=self.cfg.enable_expressions,
            is_texture_random=True,
            is_angle_random=False,
            angles_list=kwargs.get("angles_list", None),
            is_lighting_random=False,
            is_background_random=False,
            background_color=tuple([self.cfg.bg_intensity] * 3),
            seed=self.cfg.random_seed,
            device=self.devices,
            **kwargs,
        )

    def load_latents(self, latent_path, latent_types=None):
        latents = np.load(latent_path)
        latent_types = latents.files if latent_types is None else latent_types
        for latent_type in latent_types:
            cur_latent = getattr(self.generator, f"{latent_type}")
            if cur_latent.shape == latents[latent_type].shape:
                cur_latent[...] = torch.tensor(
                    latents[latent_type],
                    dtype=cur_latent.dtype,
                    device=cur_latent.device,
                )
            else:  # for lighting and views, the shape might be mismatching
                setattr(
                    self.generator,
                    f"{latent_type}",
                    torch.tensor(
                        latents[latent_type],
                        dtype=cur_latent.dtype,
                        device=cur_latent.device,
                    ),
                )

    def initialize_constraints(self):
        self.constraints = {}
        is_initialized = False

        for latent_type in self.optimized_latent_types:
            cur_latent = getattr(self.generator, latent_type)
            assert cur_latent is not None

            if getattr(self.cfg, f"optimize_only_n_{latent_type}", None) is not None:
                num_dims = int(getattr(self.cfg, f"optimize_only_n_{latent_type}"))

                if not is_initialized:
                    self.fixed_params = {}
                    self.param_mask = {}
                    is_initialized = True

                self.fixed_params[latent_type] = cur_latent.clone()
                assert num_dims <= cur_latent.shape[1]
                self.param_mask[latent_type] = torch.unsqueeze(
                    torch.arange(cur_latent.shape[1]) < num_dims, 0
                ).to(device=self.fixed_params[latent_type].device)

            else:
                num_dims = cur_latent.shape[1]

            if self.cfg.bound_latents:
                self.constraints[latent_type] = constraint_factory(
                    ball_radius=getattr(self.cfg, f"{latent_type}_ball_radius"),
                    box_SD=getattr(self.cfg, f"{latent_type}_box_SDs"),
                    original_latent=cur_latent,
                    center_around_original_latent=self.cfg.constraint_relative_to_individual_faces,
                    dim=1,
                    verbose=self.cfg.verbose,
                    latent_name=latent_type,
                    ev=self.EV[latent_type],
                    eps=self.eps,
                    num_dims=num_dims,
                )
                assert (
                    self.constraints[latent_type].num_dims
                    == int(getattr(self.cfg, f"optimize_only_n_{latent_type}"))
                    if getattr(self.cfg, f"optimize_only_n_{latent_type}", None)
                    is not None
                    else cur_latent.shape[1]
                )

                param_project = self.constraints[latent_type].project(
                    cur_latent[:, :num_dims]
                )
                cur_latent[:, :num_dims] = param_project  # TODO: is this necessary?

                # store unbounded latent (decompress to original space)
                unbounded_latents = self.constraints[latent_type].decompress(
                    param_project
                )
                setattr(
                    self.generator,
                    f"unbounded_{latent_type}",
                    unbounded_latents,
                )

    def set_up_optimization_params(self):
        self.optimized_params = {}
        for latent_type in self.optimized_latent_types:
            if self.cfg.bound_latents:
                latent_type = f"unbounded_{latent_type}"
            latents = getattr(self.generator, latent_type)
            latents.requires_grad_(True)
            self.optimized_params[latent_type] = latents

    def step_optimization(self):
        # TODO: is this necessary?
        # update face_batch so it uses the optimized parameters
        for latent_type, latents in self.optimized_params.items():
            setattr(self.generator, latent_type, latents)

        for latent_type in self.optimized_latent_types:
            # transform unbounded params to bounded params
            if self.cfg.bound_latents:
                constraint = self.constraints[latent_type]
                # TODO: test if this works correctly
                unbounded_latents = getattr(self.generator, f"unbounded_{latent_type}")
                param_compress = constraint.compress(unbounded_latents)
                num_dims = unbounded_latents.shape[1]
                bounded_latents = getattr(self.generator, latent_type)
                bounded_latents[:, :num_dims] = param_compress

            # fix last N-n latents
            if getattr(self.cfg, f"optimize_only_n_{latent_type}", None) is not None:
                setattr(
                    self.generator,
                    latent_type,
                    torch.where(
                        self.param_mask[latent_type],
                        getattr(self.generator, latent_type),
                        self.fixed_params[latent_type],
                    ),
                )

    def generate_images(self, imsize=256, im_dir=None, rgba=False):
        face_images = self.generator.render_face(
            imsize=imsize,
            binsize=getattr(self.cfg, "binsize", None),
            im_dir=im_dir,
            rgba=rgba,
        )
        return face_images

    def jitter_and_crop(self, face_images, crop_size, enable_noise=True):
        transformed_face_images = crop_head_torch(
            face_tensors=face_images,
            enable_jitter=enable_noise,
            newdim=crop_size,  # self.cfg.optimization_crop_size,
            extend=self.cfg.cropping_extend,
            overflow_method=self.cfg.cropping_overflow_method,
            crop_tightness=self.cfg.cropping_tightness,
            bg_intensity=self.cfg.bg_intensity,
            max_jitter=self.cfg.max_jitter,  # these arguments will be ignored if enable_jitter is False
            max_scale_jitter=self.cfg.max_scale_jitter,
        )

        return self._noise_images(transformed_face_images, enable_noise)

    def save_latents(self, save_path):
        self.generator.save_face(save_path)


def get_generator_path(
    generator_path=None,
    checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
    force_download=False,
):
    """
    Return a local path to the StyleGAN3 generator checkpoint.

    If generator_path is provided, use it directly.
    Otherwise, download the checkpoint from Hugging Face into:
        model_checkpoints/stylegan3-r-ffhq_128.pkl

    If the file already exists locally, reuse it.
    """
    from huggingface_hub import hf_hub_download

    if generator_path is not None:
        return str(Path(generator_path).expanduser())

    checkpoint_dir = Path(checkpoint_dir).expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = REPO_ROOT / checkpoint_dir
    local_path = checkpoint_dir / STYLEGAN3_FILENAME

    if local_path.exists() and not force_download:
        return str(local_path)

    print("downloading StyleGAN3-r-ffhq_128 weights from HuggingFace...")

    downloaded_path = hf_hub_download(
        repo_id=HF_MODEL_REPO_ID,
        filename=STYLEGAN3_FILENAME,
        repo_type="model",
        local_dir=str(checkpoint_dir),
        force_download=force_download,
    )

    return downloaded_path


class StyleGAN3(FACE_GENERATOR):
    def __init__(self, cfg, devices):
        super().__init__(cfg, devices)
        self.devices = devices
        self.param_device = self.devices[0]
        assert len(self.optimized_latent_types) == 1
        if cfg.generator_path is None:
            cfg.generator_path = get_generator_path(force_download=False)

    def set_up_generator(self):
        from stylegan3 import legacy

        with open(self.cfg.generator_path, "rb") as f:
            G = legacy.load_network_pkl(f)["G_ema"].to("cuda")

        if self.cfg.random_seed is not None:
            torch.manual_seed(self.cfg.random_seed)

        if self.optimized_latent_types[0] == "z":
            self.generator = ModelParallel(
                G, devices=self.devices, output_device=self.param_device
            )
            self.generator.z = torch.randn([self.cfg.nFaces, G.z_dim]).to(
                self.param_device
            )

        elif self.optimized_latent_types[0] == "w":
            print("optimizing w")
            self.generator = ModelParallel(
                G.synthesis, devices=self.devices, output_device=self.param_device
            )
            self.num_ws = G.num_ws
            self.w_avg = G.mapping.w_avg
            if self.cfg.coef_init_method == "average":
                self.generator.w = torch.tile(self.w_avg, (self.cfg.nFaces, 1)).to(
                    self.param_device
                )
            elif self.cfg.coef_init_method == "random":
                z = torch.randn([self.cfg.nFaces, G.z_dim]).to(self.param_device)
                w = G.mapping(
                    z,
                    None,
                    truncation_psi=self.cfg.truncation_psi,
                    truncation_cutoff=self.cfg.truncation_cutoff,
                )
                self.generator.w = w[:, 0, :]  # w is repeated for each layer
            elif self.cfg.coef_init_method == "pairwise_random":
                z = torch.randn([self.cfg.nFaces // 2, G.z_dim]).to(self.param_device)
                w = G.mapping(
                    z,
                    None,
                    truncation_psi=self.cfg.truncation_psi,
                    truncation_cutoff=self.cfg.truncation_cutoff,
                )
                w = w[:, 0, :]  # w is repeated for each layer; [nFaces//2, 512]
                w = w.tile((1, 2))
                self.generator.w = w.reshape(self.cfg.nFaces, G.z_dim)

            # full_w stores the full w vector, including the part that is not being optimized
            self.generator.full_w = self.generator.w.clone().to(self.param_device)
            # parameters being optimized
            self.generator.w = self.generator.w[:, : self.cfg.truncation_cutoff]
        else:
            raise ValueError(
                f"optimized latent type {self.optimized_latent_types[0]} not supported for StyleGAN3 generator"
            )

        # elif self.optimized_latent_types[0] == "s":
        #     self.generator = ModelParallel(
        #         G.synthesis, devices=self.devices, output_device=self.param_device
        #     )
        #     w_avg = G.mapping.w_avg
        #     w_avg = torch.tile(w_avg, (self.cfg.nFaces, 16, 1)).to(
        #         self.generator.devices[0]
        #     )
        #     s_init = self.generator.model_replicas[0].W2S(w_avg)
        #     self.optimized_latent_types = list(s_init.keys())
        #     for k in s_init.keys():
        #         setattr(
        #             self.generator,
        #             k,
        #             s_init[k].detach().clone().to(self.param_device),
        #         )

    def load_latents(self, latent_path, latent_types=None):
        latents = np.load(latent_path)
        assert len(latents.files) == 1
        latent_types = latents.files if latent_types is None else latent_types
        for k in latent_types:
            assert k in latents.files
            assert latents[k].shape == getattr(self.generator, k).shape
            getattr(self.generator, k)[...] = torch.tensor(
                latents[k], dtype=torch.float32, device=self.param_device
            )

    def initialize_constraints(self):
        if self.cfg.constraint_type == "clamp_w_NLL":
            raise NotImplementedError
        else:
            pass

    def set_up_optimization_params(self):
        self.optimized_params = {}
        for latent_type in self.optimized_latent_types:  # z, w, or s
            latents = getattr(self.generator, latent_type)
            assert latents.device == self.param_device
            latents.requires_grad_(True)
            self.optimized_params[latent_type] = latents

    def step_optimization(self):
        pass

    def generate_images(self, imsize=None, im_dir=None, rgba=None):  # rgba is not used
        if self.optimized_latent_types[0] == "z":
            face_images = self.generator(
                self.generator.z,
                None,
                truncation_psi=self.cfg.truncation_psi,
                truncation_cutoff=self.cfg.truncation_cutoff,
                noise_mode=self.cfg.noise_mode,
                force_fp32=True,
            )
        elif self.optimized_latent_types[0] == "w":
            # truncation trick
            self.generator.full_w[:, : self.cfg.truncation_cutoff] = self.w_avg[
                : self.cfg.truncation_cutoff
            ].lerp(
                self.optimized_params["w"],
                self.cfg.truncation_psi,
            )
            w_param = self.generator.full_w.unsqueeze(1).repeat([1, self.num_ws, 1])
            face_images = self.generator(
                w_param,
                noise_mode=self.cfg.noise_mode,
                force_fp32=True,
            )
        elif "input" in self.optimized_latent_types:
            face_images = self.generator(
                self.optimized_params,
                noise_mode=self.cfg.noise_mode,
                force_fp32=True,
            )
        face_images = self.postprocess_images(face_images, imsize, im_dir)
        return face_images

    def postprocess_images(self, face_images, imsize=None, im_dir=None):
        face_images = (face_images * 127.5 + 128) / 255.0
        face_images = torch.clamp(face_images, 0, 1)
        assert face_images.dtype == torch.float32
        if imsize is not None and face_images.shape[-1] != imsize:
            face_images = interpolate(
                face_images,
                size=(imsize, imsize),
                mode="bilinear",
                align_corners=True,
            )

        if im_dir is not None:
            Path(im_dir).mkdir(parents=True, exist_ok=True)
            for i_file in range(len(face_images)):
                path = os.path.join(im_dir, str(i_file) + ".png")
                save_image(face_images[i_file, ...], path)

        return face_images

    def jitter_and_crop(self, face_images, crop_size, enable_noise=True):
        if enable_noise:
            transformed_face_im_tensors = jitter_images(
                face_images, max_jitter=self.cfg.max_jitter
            )
        else:
            crop_size = (
                crop_size - self.cfg.max_jitter * 2
            )  # self.cfg.optimization_crop_size
            transformed_face_im_tensors = CenterCrop(crop_size)(face_images)
            transformed_face_im_tensors = interpolate(
                transformed_face_im_tensors,
                size=(
                    self.cfg.optimization_crop_size,
                    self.cfg.optimization_crop_size,
                ),
                mode="bilinear",
                align_corners=True,
            )
        return self._noise_images(transformed_face_im_tensors, enable_noise)

    def save_latents(self, save_path):
        latents = {}
        for latent_type in self.optimized_latent_types:
            latents[latent_type] = (
                getattr(self.generator, latent_type).clone().detach().cpu().numpy()
            )
        np.savez(save_path, **latents)


def stylegan_postprocess_torch(face_images):
    face_images = face_images.clone()
    face_images = (face_images * 127.5 + 128) / 255.0
    face_images = torch.clamp(face_images, 0, 1)
    assert face_images.dtype == torch.float32
    return face_images


def rgb_to_grayscale(im_tensor, bg_intensity=0.5, contrast_fixer=None):
    """Convert RGB face images to grayscale, optionally adjusting intensity contrast.
    tensor shape: (N, C, H, W)
    """
    fix_intensity_stats = contrast_fixer is not None
    if fix_intensity_stats:
        alpha_channel = torch.logical_not(
            torch.all(im_tensor == bg_intensity, dim=1, keepdims=False)
        )
    assert im_tensor.shape[1] == 3
    im_tensor = (
        im_tensor[:, 0] * 0.2126 + im_tensor[:, 1] * 0.7152 + im_tensor[:, 2] * 0.0722
    )
    if fix_intensity_stats:
        im_tensor = contrast_fixer(im_tensor, alpha=alpha_channel)
    im_tensor = torch.stack([im_tensor] * 3, dim=1)
    return im_tensor


def rgba_to_grayscale(im_tensor, contrast_fixer=None):
    """Convert RGBA images to grayscale, optionally adjusting intensity contrast"""
    assert im_tensor.shape[1] == 4
    fix_intensity_stats = contrast_fixer is not None
    if fix_intensity_stats:
        alpha_channel = im_tensor[:, -1] > 0
    alpha = im_tensor[:, -1]
    im_tensor = (
        im_tensor[:, 0] * 0.2126 + im_tensor[:, 1] * 0.7152 + im_tensor[:, 2] * 0.0722
    )
    if fix_intensity_stats:
        im_tensor = contrast_fixer(im_tensor, alpha=alpha_channel)
    im_tensor = torch.stack([im_tensor] * 3, dim=1)
    im_tensor = torch.cat([im_tensor, alpha.unsqueeze(1)], dim=1)
    return im_tensor
