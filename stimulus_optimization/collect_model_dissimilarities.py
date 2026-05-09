"""collect model dissimilarities to optimized/randomly sampled stimuli
for bfm stimuli, we saved in higher resolution (417 x 417)
so when we re-load and resize the stimuli to 128 x 128 for model input, the model
dissimilarities slightly differ from directly rendered 128 x 128 stimuli.
however, model predicted rankings of face pairs are preserved and we use spearman's
correlation, so this should not affect the results. can turn on or off
the `re-render_faces` flag.
"""

import os, sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import glob
import re
import argparse
import re

import PIL
import numpy as np
import torch
import pandas as pd
from omegaconf import OmegaConf

from stimulus_optimization import generator_objects
from stimulus_optimization.stim_optim import (
    prepare_readouts_and_models,
    setup_faces_to_representations,
    setup_representation_to_loss,
)


def find_generator(text):
    # Use a regex pattern that prioritizes "bfm_pose" over "bfm"
    pattern = r"(stylegan3|bfm-pose|bfm)"

    match = re.search(pattern, text)
    if match:
        return match.group()
    else:
        return "bfm"


def load_images(cfg, stimulus_set_path, stimulus_set_base_path):
    image_files = sorted(glob.glob(os.path.join(stimulus_set_path, "*.png")))

    image_tensor = []
    im_meta = []

    for image_file in image_files:
        # load image using pil and transform it to torch tensor
        image_tensor.append(
            torch.from_numpy(
                np.asarray(PIL.Image.open(image_file), dtype=np.float32) / 255.0,
            ).permute(2, 0, 1)
        )  # permute to CHW
        if cfg.file_naming_scheme == "pairwise_trial":
            # use regex to match named groups optimized_trial_xx_pair_yy_z_cropped
            match = re.match(
                rf"{stimulus_set_base_path}_trial_(?P<i_trial>\d+)_pair_(?P<i_pair>\d+)_(?P<i_image>\d+)_cropped",
                os.path.basename(image_file),
            )
            im_meta.append(
                {
                    "fname": os.path.basename(image_file),
                    "i_trial": int(match.group("i_trial")),
                    "i_pair": int(match.group("i_pair")),
                    "i_image": int(match.group("i_image")),
                }
            )
        elif cfg.file_naming_scheme == "pairwise":
            # use regex to matched named groups optimized_pair_000_1_cropped.png
            match = re.match(
                rf"{stimulus_set_base_path}_pair_(?P<i_pair>\d+)_(?P<i_image>\d+)_cropped",
                os.path.basename(image_file),
            )
            im_meta.append(
                {
                    "fname": os.path.basename(image_file),
                    "i_pair": int(match.group("i_pair")),
                    "i_image": int(match.group("i_image")),
                }
            )
        elif cfg.file_naming_scheme == "single_trial":
            raise NotImplementedError
        else:
            raise NotImplementedError
    image_tensor = torch.stack(image_tensor)  # NCHW
    im_meta = pd.DataFrame(im_meta)

    def is_sorted(l):
        return all(l[i] <= l[i + 1] for i in range(len(l) - 1))

    if cfg.file_naming_scheme == "pairwise_trial":
        assert is_sorted(im_meta.i_trial.values)
        for i_trial in im_meta.i_trial.unique():
            assert is_sorted(im_meta.i_pair.values[im_meta.i_trial == i_trial])
            for i_pair in im_meta.i_pair.unique():
                assert is_sorted(
                    im_meta.i_image.values[
                        np.logical_and(
                            im_meta.i_trial == i_trial, im_meta.i_pair == i_pair
                        )
                    ]
                )
    elif cfg.file_naming_scheme == "pairwise":
        assert is_sorted(im_meta.i_pair.values)
        for i_pair in im_meta.i_pair.unique():
            assert is_sorted(im_meta.i_image.values[im_meta.i_pair == i_pair])

    return image_tensor, im_meta


def render_images(
    cfg, stimulus_set_path, stimulus_set_base_path, face_generator_devices
):
    def generator_class(generator):
        if "bfm" in generator:
            generator = "BFMGenerator"
        elif generator == "stylegan3":
            generator = "StyleGAN3"
        return generator

    generator = generator_class(find_generator(stimulus_set_path))
    latent_path = os.path.join(
        stimulus_set_path, "..", "..", "latents", "optimized.npz"
    )
    image_tensors, im_meta = load_images(cfg, stimulus_set_path, stimulus_set_base_path)

    # stylegan3 images are optimized and saved in the same resolution, can directly load
    # only need to render faces for BFM
    if generator == "BFMGenerator":
        face_generator = getattr(generator_objects, generator)(
            cfg, devices=face_generator_devices
        )
        face_generator.set_up_generator()
        face_generator.load_latents(latent_path)
        image_tensors = face_generator.generate_images(
            imsize=cfg.optimization_imsize, im_dir=None, rgba=False
        )
        image_tensors = face_generator.jitter_and_crop(
            image_tensors,
            crop_size=cfg.optimization_crop_size,
            enable_noise=False,
        )

    return image_tensors, im_meta


def measure_and_save_optimized_stimuli_response(
    cfg,
    stimulus_set_path,
    stimulus_set_base_path,
    models,
    readouts,
    render_faces=False,
    face_generator_devices=None,
):
    if render_faces:
        image_tensor, im_meta = render_images(
            cfg, stimulus_set_path, stimulus_set_base_path, face_generator_devices
        )
    else:
        image_tensor, im_meta = load_images(
            cfg, stimulus_set_path, stimulus_set_base_path
        )

    representation_to_loss = setup_representation_to_loss(cfg)
    representation_type = representation_to_loss.domain
    cfg.dissimilarity_gradient_checkpoint = False
    faces_to_representations = setup_faces_to_representations(cfg, representation_type)
    assert (
        representation_type == "pairwise_dissimilarities"
    ), "currently only pairwise_dissimilarities is supported"

    representation_path = os.path.join(
        stimulus_set_path, "..", "..", "representation_statistics"
    )
    os.makedirs(representation_path, exist_ok=True)

    with torch.inference_mode():
        dissimilarities = faces_to_representations(image_tensor, readouts, models)
        dissimilarity_df = []
        models = dissimilarities.keys()

        for model in models:
            for instance in dissimilarities[model].keys():
                cur_model_dissimilarities = dissimilarities[model][instance]
                if cfg.file_naming_scheme == "pairwise_trial":
                    pair_idx = 0
                    for i_trial in im_meta.i_trial.unique():
                        for i_pair in im_meta.i_pair.unique():
                            for i_layer in range(cur_model_dissimilarities.shape[0]):
                                dissimilarity_df.append(
                                    {
                                        "model": model,
                                        "instance": instance,
                                        "i_layer": i_layer,
                                        "i_trial": i_trial,
                                        "i_pair": i_pair,
                                        "dissimilarity": cur_model_dissimilarities[
                                            i_layer, pair_idx
                                        ].item(),
                                    }
                                )
                            pair_idx += 1
                    assert pair_idx == cur_model_dissimilarities.shape[1]
                elif cfg.file_naming_scheme == "pairwise":
                    for i_pair in im_meta.i_pair.unique():
                        for i_layer in range(cur_model_dissimilarities.shape[0]):
                            dissimilarity_df.append(
                                {
                                    "model": model,
                                    "instance": instance,
                                    "i_layer": i_layer,
                                    "i_pair": i_pair,
                                    "dissimilarity": cur_model_dissimilarities[
                                        i_layer, i_pair
                                    ].item(),
                                }
                            )
                else:
                    raise NotImplementedError

        dissimilarity_df = pd.DataFrame(dissimilarity_df)

        uq_instances = dissimilarity_df.instance.unique()
        # save dissimilarities, separately for each instance
        for instance in uq_instances:
            cur_instance_dissimilarities = dissimilarity_df[
                dissimilarity_df.instance == instance
            ]
            cur_instance_dissimilarities.to_parquet(
                os.path.join(
                    representation_path,
                    f"dissimilarities_instance_{instance}.parquet",
                )
            )
            cur_instance_dissimilarities.to_csv(
                os.path.join(
                    representation_path,
                    f"dissimilarities_instance_{instance}.csv",
                )
            )


if __name__ == "__main__":
    str2bool = lambda x: x.lower() in ("true", "yes", "t", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimulus_set_parent_folder", type=str)
    parser.add_argument("--stimulus_set_base_path", type=str, default="optimized")
    parser.add_argument("--render_faces", type=str2bool, default=True)
    parser.add_argument("--instances", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--layer_selectors", type=str, nargs="+", default=["conv", "fc"]
    )
    args = parser.parse_args()

    stimulus_set_parent_folder = args.stimulus_set_parent_folder
    stimulus_set_base_path = args.stimulus_set_base_path
    instances = args.instances
    layer_selectors = args.layer_selectors
    render_faces = args.render_faces

    stimulus_set_paths = glob.glob(
        os.path.join(
            stimulus_set_parent_folder, "**", f"{stimulus_set_base_path}_cropped"
        ),
        recursive=True,
    )
    if len(list(stimulus_set_paths)) == 0:
        raise ValueError(
            "No stimulus sets found in the specified folder. Please make sure that the stimulus sets are in the specified folder."
        )
    else:
        print(f"Found {len(stimulus_set_paths)} stimulus sets.")

    for i_set, stimulus_set_path in enumerate(sorted(stimulus_set_paths)):
        print(f"Processing stimulus set: {stimulus_set_path}")
        conf_path = os.path.join(stimulus_set_path, "..", "..", ".hydra", "config.yaml")

        # to re-use the functions from stim_optim.py
        # we need to use the corresponding controversial config to calculate dissimilarities
        # this would automatically configure the e.g. dist_fun, loss, etc.
        if "random" in stimulus_set_path:
            conf_path = conf_path.replace("random", "controversial")
        # load config file into omegaconf config object
        cfg = OmegaConf.load(conf_path)

        cfg.verbose = 2
        cfg.n_gpus_for_rendering = 0
        cfg.sampling_device = "cpu"
        cfg.instance_id = instances
        cfg.layer_selector = ",".join(layer_selectors)  # record all selected layers
        if "bfm" in stimulus_set_path and render_faces:
            cfg.n_gpus_for_rendering = 2
            cfg.generator_path = os.path.join(
                "model_checkpoints", "model2019_fullHead.h5"
            )
        import pdb

        pdb.set_trace()
        if i_set == 0:
            readouts, models, face_generator_devices, _ = prepare_readouts_and_models(
                cfg
            )

        measure_and_save_optimized_stimuli_response(
            cfg,
            stimulus_set_path,
            stimulus_set_base_path,
            models,
            readouts,
            render_faces=render_faces,
            face_generator_devices=face_generator_devices,
        )
