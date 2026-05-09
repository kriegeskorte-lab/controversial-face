import os, sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
from differentiable_faces.load_bfm import load_BFM


def calc_avg_fov(model_path, camera_dist=90.0, device="cpu", unit="deg"):
    """calculate field of view for the average face, given the camera distance

    Args:
        model_path (str)
        camera_dist (float, optional): camera distance to face in cm (to mimic real life view distance). Defaults to 90.0 (3 feet
        device (str, optional): Defaults to 'cpu'.
        unit(str, optional): 'deg' or 'rad'. Defaults to 'deg'.
    """
    basel_face_model = load_BFM(model_path=model_path, device=device)
    dims = basel_face_model["points_dims"]
    shape_mean = basel_face_model["shapeMU"].reshape(dims)
    y = shape_mean[:, 1]
    face_height = (y.max() - y.min()).item() / 10  # convert to cm

    # calculate field of view
    fov = np.arctan(face_height / (2 * camera_dist))
    if unit == "deg":
        fov = np.rad2deg(fov)

    return fov * 2  # return full field of view
