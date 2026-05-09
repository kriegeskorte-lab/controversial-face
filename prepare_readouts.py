import os, re
import pandas as pd


def get_one_model_instance(
    *args,
    instance_id=None,
    **kwargs,
):
    # scale is taken care of in the get_layerwise_activations function

    if isinstance(instance_id, list):
        assert len(instance_id) == 1, "only single instance is supported"
        instance_id = instance_id[0]
    elif instance_id is None:
        instance_id = 0
    else:
        assert isinstance(instance_id, int), "instance_id must be an integer or None"

    readouts = [
        {
            "model_name": "VGG16_VGGFace2_128",
            "input_layer": "vggface2_transform",
            "label": "VGG16_VGGFace2_identification",
            "task": "identification",
            "architecture": "VGG16",
            "dataset": "VGGFace2",
        },
        {
            "model_name": "VGG16_BFM_identity_128",
            "input_layer": "bfm_transform",
            "label": "VGG16_BFM_identification",
            "task": "identification",
            "architecture": "VGG16",
            "dataset": "BFM",
        },
        {
            "model_name": "VGG16_ImageNet_128",
            "input_layer": "torchvision",
            "label": "VGG16_ImageNet_classification",
            "task": "classification",
            "architecture": "VGG16",
            "dataset": "ImageNet",
        },
        {
            "model_name": "VGG16_BFM_128",
            "input_layer": "bfm_transform",
            "label": "VGG16_BFM latents prediction",
            "task": "latent_reconstruction",
            "architecture": "VGG16",
            "dataset": "BFM",
        },
        {
            "model_name": "VGG16_VGGFace2_VAE_encoder_128",
            "input_layer": "vggface2_transform",
            "label": "VGG16_VGGFace2_VAE",
            "task": "generation",
            "architecture": "VGG16",
            "dataset": "VGG16",
        },
        {
            "model_name": "VGG16_BFM_VAE_encoder_128",
            "input_layer": "bfm_transform",
            "label": "VGG16_BFM_VAE",
            "task": "generation",
            "architecture": "VGG16",
            "dataset": "BFM",
        },
    ]

    for readout in readouts:
        readout["instance_id"] = instance_id

    return readouts


def prepare_readouts(
    *args,
    instance_id=None,
    **kwargs,
):
    """a version of the pilot2 six models, with multiple instances per model.
    args:
        model_comparison (str)
        instance_id (list) a list of instances to load
        activation_comrpession (str)
        load_best_layers (bool)
    """

    assert isinstance(instance_id, list), "instance_id must be a list"

    readouts = []
    for cur_instance_id in instance_id:
        cur_readouts = get_one_model_instance(
            *args,
            instance_id=cur_instance_id,
            **kwargs,
        )
        readouts.extend(cur_readouts)
    return readouts
