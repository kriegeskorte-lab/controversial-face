"""Use hook to extract intermediate layer activations."""

import torch
from typing import Optional


class Hook:
    """A hook class that returns the input and output of an intermediate layer during forward/backward pass."""

    def __init__(self, module, forward: bool = True):
        """
        Args:
            module: A layer that is an instance of torch.nn.Module class.
            forward (bool): If True, register a forward hook.
                            Otherwise register a backward hook after backpropagation. Layer order is reversed.
        """
        if forward == True:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        # self.input = input
        self.output = output

    def remove(self):
        self.hook.remove()


def get_layerwise_activations(
    model,
    layer_indices: list,
    ims: torch.Tensor,
    forward: bool = True,
    in_place=False,
):
    """get the activations from one or more DNN layers

    Args:
        model (VisionModel)
        layer_indices (list): a list of intermediate layer indices to extract activations,
                              where layer indices are determined by list(model.modules()) (or int for a single layer)
        ims (torch.Tensor)
        forward (bool): Defaults to True, which registers a forward hook. Otherwise register a backward hook.

    Returns:
        layer_out (:obj:`list` of :obj:`torch.Tensor`): a list of output activations by the layers specified.
            each element in the list is a tensor of shape (batch_size, n_channels, height, width) or (batch_size, n_channels)
            if layer_indices is a single int, a single tensor (not a list) is returned.

    Raises:
        ValueError: currently only supports no compression or sigmoid compression with precomputed model scaling factors.
                    raises ValueError otherwise.
    """

    if isinstance(layer_indices, int):
        single_layer_mode = True
        layer_indices = [layer_indices]
    else:
        single_layer_mode = False

    hooks = []
    for i_layer in layer_indices:
        layer = list(model.modules())[i_layer]
        hooks.append(Hook(layer, forward))

    out = model(ims)
    del out

    layer_out = []
    for i_layer, hook_i in zip(layer_indices, hooks):
        out = hook_i.output
        layer_out.append(out)
        hook_i.remove()

    if single_layer_mode:
        layer_out = layer_out[0]
    return layer_out
