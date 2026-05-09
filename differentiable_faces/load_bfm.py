"""Load the Basel Face Model (BFM) parameters.

Example:
    model=load_BFM(MODEL_PATH, device=torch.device('cuda')) # initialize the BFM model
"""

import torch
from typing import Optional
import h5py


def load_BFM(model_path: str, device: Optional[str] = None):
    """loading BFM on the specified torch device.

    Args:
        model_path (str):
            Local path to the Basel Face Model 2019. Submit a request at https://faces.dmi.unibas.ch/bfm/bfm2019.html to download the model.
        device (:obj: `str`, optional):
            Device where model parameters are loaded. Defaults to None, in which case GPU is used if available.

    Returns:
        model (dict): a dictionary of relevant model parameters.
    """
    f = h5py.File(model_path, "r")

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    model = dict()

    data_types = ["shape", "expression", "color"]

    for data_type in data_types:
        try:
            data = f[data_type]
            if data_type == "expression":
                data_type = "expr"
            elif data_type == "color":
                data_type = "tex"

            model[data_type + "MU"] = torch.tensor(
                data["model"]["mean"], dtype=torch.float32, device=device
            )
            model[data_type + "PC"] = torch.tensor(
                data["model"]["pcaBasis"], dtype=torch.float32, device=device
            ).permute(1, 0)

            Var = torch.tensor(
                data["model"]["pcaVariance"], dtype=torch.float32, device=device
            )
            model[data_type + "EV"] = (
                Var**0.5
            )  # Eigenvalues ; use standard deviations for Karhunan-Loeve expansion
            model[data_type + "_dims"] = model[data_type + "EV"].shape[0]

        except:
            continue

    model["Cells"] = torch.tensor(
        f["shape"]["representer"]["cells"], dtype=int, device=device
    ).transpose(0, 1)
    model["Points"] = torch.tensor(
        f["shape"]["representer"]["points"], dtype=torch.double, device=device
    ).transpose(0, 1)
    model["points_dims"] = model["Points"].shape

    return model


def load_attribute(attribute_path):
    import scipy.io as sio

    attribute_data = sio.loadmat(attribute_path)

    return attribute_data
