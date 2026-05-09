import os, sys

# add project root to path for importing libraries
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from tqdm import tqdm
import numpy as np
import xarray as xr
import itertools
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import matplotlib.font_manager as fm

from data_analysis.compute_model_performance_by_seeds import (
    load_model_dissimilarities_from_path,
)
from data_analysis.analysis_utils import xr_spearmanrho_rae
from vis.style import (
    model_label_map,
    model_palette,
    model_order,
)

import matplotlib.font_manager as fm

font_path = "HelveticaNeue.ttc"
fm.fontManager.addfont(font_path)

prop = fm.FontProperties(fname=font_path)
print("registered name:", prop.get_name())

plt.rcParams["font.family"] = prop.get_name()


def compute_similarity_matrix(
    stimuli_info_dir,
    models,
):
    """compute similarity matrix between all model layers for all generators and conditions"""
    generators = ["bfm", "bfm-pose", "stylegan3"]
    conditions = ["random", "controversial"]
    model_layers = [f"{model}_{i_layer}" for model in models for i_layer in range(16)]
    coords = {
        "generator": generators,
        "condition": conditions,
        "i_seed": np.arange(12),
        "model1": model_layers,
        "model2": model_layers,
    }
    model_similarity = xr.DataArray(
        np.zeros(
            (len(generators), len(conditions), 12, len(model_layers), len(model_layers))
        ),
        coords=coords,
        dims=list(coords.keys()),
    )

    for generator in generators:
        for condition in conditions:
            cond_root_dir = os.path.join(stimuli_info_dir, generator, condition)
            seed_dirs = sorted(
                [
                    seed_dir
                    for seed_dir in os.listdir(cond_root_dir)
                    if "seed_" in seed_dir
                ]
            )
            for i_seed, seed_dir in enumerate(seed_dirs):
                cond_exp_dir = os.path.join(cond_root_dir, seed_dir)
                model_dissimilarity_dict = load_model_dissimilarities_from_path(
                    cond_exp_dir,
                    instance_id=0,
                )
                # combinations with replacement: for the same model we need to compute layer-wise similarities
                for model1, model2 in itertools.combinations_with_replacement(
                    models, 2
                ):
                    model1_dissimilarities = model_dissimilarity_dict[
                        model1
                    ].dissimilarity.rename({"model": "model1", "i_layer": "i_layer1"})
                    model2_dissimilarities = model_dissimilarity_dict[
                        model2
                    ].dissimilarity.rename({"model": "model2", "i_layer": "i_layer2"})
                    assert model1_dissimilarities.shape == model2_dissimilarities.shape
                    comp = xr_spearmanrho_rae(
                        model1_dissimilarities, model2_dissimilarities, dim="i_pair"
                    )
                    assert np.all(np.abs(comp) <= 1.0)
                    comp = (
                        comp.drop_vars(("instance", "model1", "model2"))
                        .squeeze()
                        .mean("i_trial")
                    )
                    model_layers1 = [f"{model1}_{i_layer}" for i_layer in range(16)]
                    model_layers2 = [f"{model2}_{i_layer}" for i_layer in range(16)]

                    # Add the (model1, model2) block
                    coords = dict(
                        generator=generator,
                        condition=condition,
                        i_seed=i_seed,
                        model1=model_layers1,
                        model2=model_layers2,
                    )
                    model_similarity.loc[coords] = comp.values

                    if model1 != model2:
                        # Add the (model2, model1) block (symmetric)
                        coords = dict(
                            generator=generator,
                            condition=condition,
                            i_seed=i_seed,
                            model1=model_layers2,
                            model2=model_layers1,
                        )
                        model_similarity.loc[coords] = comp.values.T
    return model_similarity


def procrustes(X, Y, scaling=False, reflection="best"):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.0).sum()
    ssY = (Y0**2.0).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != "best":

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {"rotation": T, "scale": b, "translation": c}

    return d, Z, tform


def apply_tform(P, tform):
    # Your procrustes() returns tform that maps Y -> X as:  b * Y @ T + c
    return tform["scale"] * (P @ tform["rotation"]) + tform["translation"]


def dot(label, base_color_name):
    return Line2D(
        [],
        [],
        linestyle="None",
        marker="o",
        markersize=6.5,
        markerfacecolor=base_color_name,
        markeredgecolor="white",
        markeredgewidth=1,
        label=label,
    )


def header(label):
    return Line2D(
        [],
        [],
        linestyle="None",
        marker="o",
        markersize=6.5,
        markerfacecolor="none",
        markeredgecolor="none",
        label=label,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, default="exp_data")
    args = parser.parse_args()

    exp_root_dir = args.experiment_dir
    fig_dir = os.path.join(exp_root_dir, "figures")
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    analysis_root_dir = os.path.join(exp_root_dir, "analysis")
    stimuli_info_dir = os.path.join(exp_root_dir, "stimuli_info")
    # exclude "leave1out" because it is not used to optimize controversial stimuli
    model_order.remove("leave1out")
    model_layers = [
        f"{model}_{i_layer}" for model in model_order for i_layer in range(16)
    ]

    model_similarity = compute_similarity_matrix(
        stimuli_info_dir,
        model_order,
    )
    panel_title_fontsize = 9
    legend_fontsize = 7

    generator_map = {
        "bfm": "frontal BFM",
        "bfm-pose": "pose-varied BFM",
        "stylegan3": "StyleGAN3",
    }

    fig, ax = plt.subplots(
        2,
        3,
        figsize=(7, 4.7),  # Keep the figsize taller to make room for the legend
        # constrained_layout=True,
        gridspec_kw={"wspace": 0, "hspace": 0},
        dpi=200,
    )
    for i_gen, generator in tqdm(enumerate(model_similarity.generator.values)):
        for i_cond, condition in enumerate(model_similarity.condition.values):
            model_similarity_matrix = (
                model_similarity.loc[
                    dict(
                        generator=generator,
                        condition=condition,
                        model1=model_layers,
                        model2=model_layers,
                    )
                ]
                .mean("i_seed")
                .values
            )
            dissimilarity_matrix = 1 - model_similarity_matrix

            mds = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=42,
                metric=True,
                n_jobs=4,
            )
            mds_coords = mds.fit_transform(
                dissimilarity_matrix,
            )

            if i_gen == 0 and i_cond == 0:
                reference_embedding = mds_coords
                aligned_coords = mds_coords
            else:
                _, aligned_coords, _ = procrustes(reference_embedding, mds_coords)

            for model in model_order:
                model_layer_indices = [
                    i
                    for i, layer in enumerate(model_similarity.model1.values)
                    if layer.startswith(model)
                ]
                points_to_plot = aligned_coords[model_layer_indices]
                color = model_palette[model]
                num_layers = len(points_to_plot)
                rgb_color = mcolors.to_rgb(color)
                alpha_values = np.linspace(0.4, 1.0, num_layers)
                rgba_colors = np.zeros((num_layers, 4))
                rgba_colors[:, :3] = rgb_color
                rgba_colors[:, 3] = alpha_values

                ax[i_cond, i_gen].scatter(
                    points_to_plot[:, 0],
                    points_to_plot[:, 1],
                    label=model_label_map[model],
                    color=rgba_colors,
                    s=24,
                    edgecolors="white",
                    linewidths=0.5,
                )

            if i_cond == 0:
                ax[i_cond, i_gen].set_title(
                    f"{generator_map[generator]}", fontsize=panel_title_fontsize
                )
            if i_gen == 0:
                ax[i_cond, i_gen].set_ylabel(
                    f"{condition}", fontsize=panel_title_fontsize
                )
                ax[i_cond, i_gen].yaxis.set_label_coords(-0.07, 0.5)
            ax[i_cond, i_gen].set_xticks([])
            ax[i_cond, i_gen].set_yticks([])
            ax[i_cond, i_gen].set_ylim([-1, 1])
            ax[i_cond, i_gen].set_xlim([-1, 1])

    model_palette = {
        model_label_map[model]: color for model, color in model_palette.items()
    }

    col2_handles = [
        header("trained on real photographs"),
        dot(
            "face identification (faceID-VGGFace2)",
            model_palette["face identification (VGGFace2)"],
        ),
        dot(
            "autoencoding (autoenc-VGGFace2)", model_palette["autoencoding (VGGFace2)"]
        ),
        dot(
            "object classification (objCat-ImageNet)",
            model_palette["object classification (ImageNet)"],
        ),
    ]
    col1_handles = [
        header("trained on synthetic faces"),
        dot(
            "face identification (faceID-BFM)",
            model_palette["face identification (BFM)"],
        ),
        dot("autoencoding (autoenc-BFM)", model_palette["autoencoding (BFM)"]),
        dot(
            "inverse rendering (invRend-BFM)", model_palette["inverse rendering (BFM)"]
        ),
    ]

    model_handles = col1_handles + col2_handles
    model_labels = [h.get_label() for h in model_handles]
    num_model_rows = len(col1_handles)

    # Create the first legend
    model_legend = fig.legend(
        model_handles,
        model_labels,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        title="Model",
        title_fontsize=panel_title_fontsize,
        fontsize=legend_fontsize,
        columnspacing=4.0,
        handletextpad=0.5,
        labelspacing=0.5,
        handlelength=1.0,
        borderpad=0.0,
        numpoints=1,
        frameon=False,
    )

    # Style its titles and headers
    title = model_legend.get_title()
    title.set_weight("bold")
    texts = model_legend.get_texts()
    texts[0].set_weight("bold")
    texts[num_model_rows].set_weight("bold")

    shift_amount = -10  # You may need to tweak this exact pixel value slightly
    texts[0].set_position((shift_amount, 0))
    texts[num_model_rows].set_position((shift_amount, 0))

    # Create 16 colored patches to represent the gradient blocks
    num_blocks = 16
    gradient_patches = []
    alpha_values = np.linspace(0.25, 1.0, num_blocks)
    for alpha in alpha_values:
        rgba_color = (0.2, 0.2, 0.2, alpha)
        gradient_patches.append(
            Patch(facecolor=rgba_color, edgecolor="black", linewidth=0.5)
        )

    # Create the second legend using the patches
    # We use HandlerTuple to display the patches horizontally with no padding.
    bar_legend = fig.legend(
        [tuple(gradient_patches)],
        [""],  # Handle is a tuple of patches, label is empty
        title="Layer Index",
        title_fontsize=panel_title_fontsize,
        fontsize=panel_title_fontsize - 1,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        borderpad=0.0,
        handletextpad=0,  # No space between bar and (empty) label
        handlelength=14.0,  # Length of the entire bar
    )
    bar_legend.get_title().set_weight("bold")
    fig.canvas.draw()

    # Get the bounding boxes
    bbox_model = model_legend.get_window_extent().transformed(
        fig.transFigure.inverted()
    )
    bbox_bar = bar_legend.get_window_extent().transformed(fig.transFigure.inverted())

    # Calculate the position to place the bar legend just below the model legend
    # We use the bar legend's height to know how far down to move it.
    bar_y_pos = bbox_model.y0 - bbox_bar.height - 0.01  # 0.01 is a small padding
    bar_legend.set_bbox_to_anchor((0.5, bar_y_pos), transform=fig.transFigure)

    # Render again to get the final combined position
    fig.canvas.draw()
    bbox_model = model_legend.get_window_extent().transformed(
        fig.transFigure.inverted()
    )
    bbox_bar = bar_legend.get_window_extent().transformed(fig.transFigure.inverted())

    # Calculate the total bounding box for the combined legend
    combined_x0 = min(bbox_model.x0, bbox_bar.x0)
    combined_y0 = min(bbox_model.y0, bbox_bar.y0)
    combined_x1 = max(bbox_model.x1, bbox_bar.x1)
    combined_y1 = max(bbox_model.y1, bbox_bar.y1)
    combined_width = combined_x1 - combined_x0
    combined_height = combined_y1 - combined_y0

    # Calculate the horizontal shift needed to center the entire block
    center_x = combined_x0 + combined_width / 2
    shift = 0.5 - center_x

    # Apply the shift to both legends' positions
    model_legend.set_bbox_to_anchor(
        (0.5 + shift, bbox_model.y0 - 0.01), transform=fig.transFigure
    )
    bar_legend.set_bbox_to_anchor(
        (0.5 + shift, bar_y_pos - 0.018), transform=fig.transFigure
    )

    # Place 0 and 16 under the gradient bar
    # Render to get the updated, final positions after the shift
    fig.canvas.draw()
    final_bbox_bar = bar_legend.get_window_extent().transformed(
        fig.transFigure.inverted()
    )

    # Place the text just below the y0 (bottom) of the bar legend's bounding box
    text_y = final_bbox_bar.y0 - 0.002
    fig.text(
        final_bbox_bar.x0, text_y, "0", ha="center", va="top", fontsize=legend_fontsize
    )
    fig.text(
        final_bbox_bar.x1, text_y, "15", ha="center", va="top", fontsize=legend_fontsize
    )

    fig.savefig(
        os.path.join(fig_dir, "fig2_model_mds.pdf"), dpi=300, bbox_inches="tight"
    )
