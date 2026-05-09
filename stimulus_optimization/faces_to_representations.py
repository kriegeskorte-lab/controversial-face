"""Generate model RDMs to be optimized"""

from typing import Callable, Optional
import contextlib

import torch
from .activation import get_layerwise_activations

from stimulus_optimization.rdm_utils import rdm_euclidean, pairwise_euclidean


def images_to_activations_to_rdms_simple(
    face_tensors: torch.Tensor,
    readouts: list,
    models: dict,
    rdm_fun: Callable,
    amp: Optional[bool] = False,
    gradient_checkpoint=True,
    keys_are_model_instance_tuples=False,
):
    """go from images to activations to rdms
    This function evaluates the model response seperately for each readout.

    args:
    face_tensor (torch.Tensor) 4D image tensor
    readouts (list of dicts)
    models (dict of VisionModels)
    rdm_fun (function) transforms activations to RDMs
    amp (bool) whether to run the forward pass in mixed precision mode
    gradient_checkpoint (bool) whether to use gradient checkpointing (checkpointing the RDM). This reduces GPU memory usage when backpropagating through multiple models, but it slows down the computation.
    keys_are_model_instance_tuples (bool) whether the keys of the models dict are model instance tuples (e.g. (model_name, model_instance_idx)) or just model names (e.g. model_name)

    returns a list of rdm tensors (or - a list of n_layers x n_distances rdm tensors when multiple layers are requested)
    """

    def get_rdm_for_one_readout(
        face_tensors: torch.Tensor,
        readout: list,
        models: dict,
        rdm_fun: Callable,
        gradient_checkpoint=False,
    ):
        if gradient_checkpoint:
            model_device = readout["model_device"]

            def custom_forward(*inputs):
                return get_rdm_for_one_readout(
                    *inputs, readout, models, rdm_fun, gradient_checkpoint=False
                )

            return torch.utils.checkpoint.checkpoint(
                custom_forward, face_tensors.to(model_device), use_reentrant=True
            )

        model_name = readout["model_name"]
        instance_id = readout["instance_id"]
        layer_required = readout["layer_idx"]
        # print(layer_required)

        if keys_are_model_instance_tuples:
            key = (model_name, instance_id)
        else:
            key = model_name

        # get all required layers from a model
        with contextlib.ExitStack() as stack:
            if amp:  # use mixed CUDA precision for model evaluation.
                stack.enter_context(torch.cuda.amp.autocast())
            layer_activations = get_layerwise_activations(
                models[key],
                layer_required,
                face_tensors,
                forward=True,
                in_place=False,
            )
        # AMP creates issues for RDM calculations in some cases, so the following commands are outside the AMP context
        if isinstance(layer_activations, torch.Tensor):
            layer_activations = layer_activations.float()
            rdm = rdm_fun(layer_activations)
            rdm.k = layer_activations[0].flatten().shape[0]
        else:  # layer_activations is a list
            rdm = torch.stack(
                [
                    rdm_fun(layer_activation.float())
                    for layer_activation in layer_activations
                ],
                dim=0,
            )
            k = torch.tensor(
                [
                    layer_activation[0].flatten().shape[0]
                    for layer_activation in layer_activations
                ],
                device=rdm.device,
            )
            rdm.k = k
        return rdm

    rdms = []
    for readout in readouts:
        rdms.append(
            get_rdm_for_one_readout(
                face_tensors,
                readout,
                models,
                rdm_fun,
                gradient_checkpoint=gradient_checkpoint,
            )
        )

    return rdms


def images_to_activations_to_pairwise_dissimilarities(
    image_tensors: torch.Tensor,
    readouts: list,
    models: dict,
    amp: Optional[bool] = False,
    gradient_checkpoint=True,
    keys_are_model_instance_tuples=True,
):
    """transform images to activations to pairwise distances

    args:
    image_tensor (torch.Tensor) 4D image tensor (NCHW)
    readouts (list of dicts)
    models (dict of VisionModels)
    amp (bool) whether to run the forward pass in mixed precision mode
    gradient_checkpoint (bool) whether to use gradient checkpointing (checkpointing the RDM). This reduces GPU memory usage when backpropagating through multiple models, but it slows down the computation.
    keys_are_model_instance_tuples (bool) whether the keys of the models dict are model instance tuples (e.g. (model_name, model_instance_idx)) or just model names (e.g. model_name)

    returns an n-models long list of lists
    each list contains dissimilarity tensors for each of the instances (realizations) of one model

    if a single layer is requested, the tensor is an n_distances long vector
    if multiple layers are requested, the tensor is an n_layers x n_distances tensor

    In both cases, the first distance returned is the distance between the image_tensors[0] and the image_tensors[1],
    the second distance is the distance between the image_tensors[2] and the image_tensors[3], and so on.

    """

    assert (
        image_tensors.shape[0] % 2 == 0
    ), "image_tensors must have an even number of images"

    def get_pairwise_dissimilarities_for_one_readout(
        image_tensors: torch.Tensor,
        readout: dict,
        models: dict,
        gradient_checkpoint=False,
    ):
        if gradient_checkpoint:
            model_device = readout["model_device"]

            def custom_forward(*inputs):
                return get_pairwise_dissimilarities_for_one_readout(
                    *inputs, readout, models, gradient_checkpoint=False
                )

            return torch.utils.checkpoint.checkpoint(
                custom_forward, image_tensors.to(model_device), use_reentrant=True
            )

        model_name = readout["model_name"]
        instance_id = readout["instance_id"]
        layer_required = readout["layer_idx"]
        # print(layer_required)

        if keys_are_model_instance_tuples:
            key = (model_name, instance_id)
        else:
            key = model_name

        # get all required layers from a model
        with contextlib.ExitStack() as stack:
            if amp:  # use mixed CUDA precision for model evaluation.
                stack.enter_context(torch.cuda.amp.autocast())
            layer_activations = get_layerwise_activations(
                models[key],
                layer_required,
                image_tensors,
                forward=True,
                in_place=False,
            )

        # AMP creates issues for RDM calculations in some cases, so the following commands are outside the AMP context
        if isinstance(layer_activations, torch.Tensor):  # a single layer is evaluated
            d = pairwise_euclidean(
                layer_activations[::2].float(), layer_activations[1::2].float()
            )
            d.k = layer_activations[0].flatten().shape[0]
        else:  # layer_activations is a list
            d = torch.stack(
                [
                    pairwise_euclidean(
                        layer_activation[::2].float(), layer_activation[1::2].float()
                    )
                    for layer_activation in layer_activations
                ],
                dim=0,
            )
            k = torch.tensor(
                [
                    layer_activation[0].flatten().shape[0]
                    for layer_activation in layer_activations
                ],
                device=d.device,
            )
            d.k = k
        return d

    disimilarity_vectors = []
    for readout in readouts:
        disimilarity_vectors.append(
            get_pairwise_dissimilarities_for_one_readout(
                image_tensors,
                readout,
                models,
                gradient_checkpoint=gradient_checkpoint,
            )
        )
    return disimilarity_vectors


class ImagesToRdms:
    """A wrapper class for images_to_activations_to_rdms_simple()."""

    def __init__(self, cfg):
        self.domain = "images"
        self.codomain = "RDMs"
        self.normalize_first_level_rdm = cfg.normalize_first_level_rdm
        self.amp = cfg.amp
        self.first_level_dist_fun = lambda activations: rdm_euclidean(
            activations, normalize=cfg.normalize_first_level_rdm
        )
        self.gradient_checkpoint = cfg.dissimilarity_gradient_checkpoint
        self.objective = cfg.loss

    def __call__(self, face_tensors, readouts, models):
        # given images, readouts and models, return rdms
        l = images_to_activations_to_rdms_simple(
            face_tensors=face_tensors,
            readouts=readouts,
            models=models,
            rdm_fun=self.first_level_dist_fun,
            amp=self.amp,
            gradient_checkpoint=self.gradient_checkpoint,
            keys_are_model_instance_tuples=True,
        )

        if (
            self.objective == "multilayer"
            or self.objective == "expected_difference"
            or self.objective == "raw_corr"
        ):
            return l
        elif "multiinstance" in self.objective:
            # multi-instance single_reference_utility function (CachedCorrelationCalc) requires RDMs in dict format
            d = {}
            for i_readout, readout in enumerate(readouts):
                model_name = readout["model_name"]
                instance_id = readout["instance_id"]
                if model_name not in d:
                    d[model_name] = {}
                d[model_name][instance_id] = l[i_readout]
            return d
        else:
            raise NotImplementedError


class ImagesToPairwiseDissimilarities:
    """A wrapper class for images_to_activations_to_pairwise_dissimilarities()."""

    def __init__(self, cfg):
        self.domain = "images"
        self.codomain = "pairwise_dissimilarities"
        self.normalize_first_level_rdm = cfg.normalize_first_level_rdm
        self.amp = cfg.amp
        self.gradient_checkpoint = cfg.dissimilarity_gradient_checkpoint
        self.objective = cfg.loss

    def __call__(self, face_tensors, readouts, models):
        # given images, readouts and models, return pairwise dissimilarities
        l = images_to_activations_to_pairwise_dissimilarities(
            image_tensors=face_tensors,
            readouts=readouts,
            models=models,
            amp=self.amp,
            gradient_checkpoint=self.gradient_checkpoint,
            keys_are_model_instance_tuples=True,
        )
        # l is a list, each element corresponds to one readout in readouts
        # and contains either a n_layers x n_pairs tensor or a n_pairs tensor of dissimilarities
        if "expected_difference" in self.objective:
            return l
        # to ease downstream processing, we return a dictionary with model name as key
        # and nested dictionary with instance id as key and dissimilarity tensor as value
        elif "multiinstance" in self.objective:
            d = {}
            for i_readout, readout in enumerate(readouts):
                model_name = readout["model_name"]
                instance_id = readout["instance_id"]
                if model_name not in d:
                    d[model_name] = {}
                d[model_name][instance_id] = l[i_readout]
            return d
        else:
            raise NotImplementedError


class cloneLatents:
    """simply grab the ground-truth BFM latents of the faces (shape, texture, and optionally, expression)"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.domain = "latents"
        self.codomain = "latents"

    def __call__(self, latent_dict):
        """
        returns a dictionary of coefficient matrices (shape, texture, and optionally, expression)
        """
        latent_dict = {
            latent_type: getattr(latent_dict, latent_type)
            for latent_type in self.cfg.coefs
        }
        return latent_dict
