import os, sys

# add project root to path for importing libraries
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import glob
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import scipy.stats
import statsmodels.stats.multitest
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


def summarize_layer_performance(model_performance_xr):
    """Summarize layer performance across participants and trials; estimate standard error"""
    model_performance_avg_xr = model_performance_xr.mean(dim="i_trial")
    model_performance_se = model_performance_avg_xr.std(dim="subj_id") / np.sqrt(
        model_performance_avg_xr.count(dim="subj_id")
    )  # using count() is robust to missing data, len() is not
    model_performance_se_df = (
        model_performance_se.to_dataframe().reset_index().drop(columns="instance")
    )
    model_performance_avg_df = (
        model_performance_avg_xr.mean("subj_id")
        .to_dataframe()
        .reset_index()
        .drop(columns="instance")
    )

    return model_performance_avg_df, model_performance_se_df


def get_cv_best_layer_accuracy(df_model_performance):

    cv_model_perfomance = []
    uq_participants = df_model_performance["subj_id"].unique()
    for subj_id in uq_participants:
        # omit one participant
        df_all_but_one = df_model_performance[
            df_model_performance["subj_id"] != subj_id
        ]

        # average model performance across participants and trials
        df_all_but_one_avg = (
            df_all_but_one.groupby(["model", "i_layer"])["trial_corr"]
            .mean()
            .reset_index()
        )

        df_all_but_one_avg["best_trial_corr"] = df_all_but_one_avg.groupby("model")[
            "trial_corr"
        ].transform("max")

        best_layer = df_all_but_one_avg[
            df_all_but_one_avg["trial_corr"] == df_all_but_one_avg["best_trial_corr"]
        ]
        best_layer = best_layer.drop(
            columns=set(best_layer.columns) - {"model", "i_layer", "best_trial_corr"}
        )

        # for each model, get model performance in the best layer for the omitted participant
        df_held_out = df_model_performance[df_model_performance["subj_id"] == subj_id]
        for model in df_held_out.model.unique():
            if len(best_layer[best_layer["model"] == model]["i_layer"]) == 1:
                cur_model_best_i_layer = best_layer[best_layer["model"] == model][
                    "i_layer"
                ].item()
            else:
                cur_model_best_i_layer = np.random.choice(
                    best_layer[best_layer["model"] == model]["i_layer"], 1
                )[0]
            cur_model_cv_performance = df_held_out[
                np.logical_and(
                    df_held_out["model"] == model,
                    df_held_out["i_layer"] == cur_model_best_i_layer,
                )
            ]["trial_corr"].mean()
            cv_model_perfomance.append(
                {
                    "subj_id": subj_id,
                    "model": model,
                    "best_layer": cur_model_best_i_layer,
                    "cv_corr": cur_model_cv_performance,
                }
            )

    return pd.DataFrame(cv_model_perfomance)


def MCP_corrected_between_model_statistical_testing(df_cv_model_performance):
    # statistical testing

    # apply Fisher tranformation to the correlation coefficients
    df_cv_model_performance["z"] = df_cv_model_performance["cv_corr"].transform(
        np.arctanh
    )

    # iterate over unique model pairs

    pairwise_comparisons = []
    model_pairs = itertools.combinations(df_cv_model_performance["model"].unique(), 2)
    for model1, model2 in model_pairs:
        model1_z = df_cv_model_performance[df_cv_model_performance["model"] == model1][
            "z"
        ]
        model2_z = df_cv_model_performance[df_cv_model_performance["model"] == model2][
            "z"
        ]

        # # perform paired t-test
        statistic, p = scipy.stats.ttest_rel(
            model1_z, model2_z, alternative="two-sided"
        )

        # perform effect size estimate (using G*power)
        cohens_d = np.abs(model1_z.mean() - model2_z.mean()) / np.sqrt(
            (model1_z.var() + model2_z.var())
            - 2
            * np.corrcoef(model1_z, model2_z)[0, 1]
            * model1_z.std()
            * model2_z.std()
        )

        pairwise_comparisons.append(
            {
                "level1": model1,
                "level2": model2,
                "t": statistic,
                "effect_direction": np.sign(statistic),
                "p": p,
                "cohens_d": cohens_d,
                # 'model1_mean':model1_z.mean(),
                # 'model2_mean':model2_z.mean(),
                # 'model1_sd':model1_z.std(),
                # 'model2_sd':model2_z.std(),
                "corr": np.corrcoef(model1_z, model2_z)[0, 1],
            }
        )

    pairwise_comparisons = pd.DataFrame(pairwise_comparisons)

    # # apply FDR_BH multiple comparison correction
    pairwise_comparisons["is_sig"], pairwise_comparisons["p_corrected"] = (
        statsmodels.stats.multitest.fdrcorrection(
            pairwise_comparisons["p"], alpha=0.05, is_sorted=False
        )
    )

    # family-wise error:
    # (
    #     pairwise_comparisons["is_sig"],
    #     pairwise_comparisons["p_corrected"],
    #     _,
    #     _,
    # ) = statsmodels.stats.multitest.multipletests(
    #     pvals=pairwise_comparisons["p"],
    #     alpha=0.05,
    #     method="hs",
    #     is_sorted=False,
    #     returnsorted=False,
    # )

    return pairwise_comparisons


def plot_layer_performance(
    plot_ax,
    model_performance_avg_df,
    model_performance_se_df,
    model_label_map,
    model_order,
    model_palette,
    layer_names,
    lower_noise_ceiling,
    upper_noise_ceiling,
    alpha=0.3,
    bfm_performance_avg=None,
    linewidth=0.5,
):

    common_layer_names = None
    for model in model_order:
        if model == model_label_map["BFM"]:
            plot_ax.axhline(
                bfm_performance_avg, color=model_palette[model], linewidth=linewidth
            )
            continue
        if model == model_label_map["leave1out"]:
            continue
        x = range(len(layer_names[model]))
        y = model_performance_avg_df[model_performance_avg_df.model == model].trial_corr
        se = model_performance_se_df[model_performance_se_df.model == model].trial_corr
        plot_ax.plot(x, y, color=model_palette[model], label=model, linewidth=linewidth)
        plot_ax.fill_between(
            x=x,
            y1=y + se,
            y2=y - se,
            alpha=alpha,
            color=model_palette[model],
            linewidth=0,
        )
        if common_layer_names is None:
            common_layer_names = layer_names[model]
        else:
            assert common_layer_names == layer_names[model]

    if lower_noise_ceiling is not None and upper_noise_ceiling is not None:
        plot_ax.fill_between(
            x=x,
            y1=upper_noise_ceiling,
            y2=lower_noise_ceiling,
            color="grey",
            label="noise ceiling bound",
            alpha=alpha,
            linewidth=0,
        )

    plot_ax.set_xlim([0, len(layer_names) - 1])
    # despine top and left
    plot_ax.spines["top"].set_visible(False)
    plot_ax.spines["right"].set_visible(False)


def plot_cv_model_performance_figure(
    plot_ax,
    df_cv_model_performance,
    upper_noise_ceiling,
    lower_noise_ceiling,
    model_order,
    model_palette,
    alpha=0.3,
    dotsize=1.5,
    edgewidth=0.05,
    vertsize=6,
):
    if lower_noise_ceiling is not None and upper_noise_ceiling is not None:
        plot_ax.axvspan(
            lower_noise_ceiling,
            upper_noise_ceiling,
            alpha=alpha,
            color="gray",
            linewidth=0,
        )
    sns.stripplot(
        data=df_cv_model_performance,
        y="model",
        x="cv_corr",
        hue="model",
        order=model_order,
        palette=model_palette,
        size=dotsize,
        linewidth=edgewidth,
        edgecolor="white",
    )

    df_avg_cv_model_performance = (
        df_cv_model_performance.groupby(["model"])["cv_corr"].mean().reset_index()
    )
    verts = [(-1, 4.8), (-1, -4.8), (1, -4.8), (1, 4.8), (-1, 4.8)]
    sns.stripplot(
        data=df_avg_cv_model_performance,
        y="model",
        x="cv_corr",
        hue="model",
        edgecolors="k",
        order=model_order,
        palette=model_palette,
        size=vertsize,
        jitter=0,
        linewidth=edgewidth,
        zorder=3,
        marker=verts,
        alpha=1.0,
    )

    # turn off legend:
    plot_ax.get_legend().remove()
    plot_ax.set_xlim([-0.1, 1])


def reconstruct_trial(
    face_pair_images, ratings, horizontal_locations, ratings_SDs=None, ax=None, zoom=0.5
):
    """Visually reconstruct a trial
    args:
        face_pair_images: a list of filenames or a list of PIL images
        ratings: a list of mean ratings (0 - identical, 1 - maximally different)
        ratings_SDs: a list of standard deviations of ratings
        ax: a matplotlib axis to plot on (if None, a new figure is created)
    """

    assert len(face_pair_images) == len(ratings)
    if ratings_SDs is not None:
        assert len(ratings_SDs) == len(ratings)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2.18, 1.61), dpi=1000)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(-0.12, 1.12)

    # load images as RGBA numpy arrays
    for i_pair, face_pair_image in enumerate(face_pair_images):
        if isinstance(face_pair_image, str):
            face_pair_image = Image.open(face_pair_image)
        if isinstance(face_pair_image, Image.Image):
            face_pair_image = np.array(face_pair_image)
        assert isinstance(face_pair_image, np.ndarray)
        assert face_pair_image.ndim == 3
        assert face_pair_image.shape[2] == 4, "must be RGBA image"
        face_pair_images[i_pair] = face_pair_image
    # determine positions
    x = np.asarray(horizontal_locations)
    y = np.asarray(ratings)

    # plot images using the OffsetImage class
    for i, face_pair_image in enumerate(face_pair_images):
        # images are scaled such that their height equals
        # %10 of the figure width
        fig_width_in_pixels = fig.get_figwidth() * fig.get_dpi()
        image_height = 0.15 * fig_width_in_pixels
        zoom = 72 / fig.dpi * image_height / face_pair_image.shape[0]
        imagebox = OffsetImage(face_pair_image, zoom=zoom)
        xy = (x[i], y[i])
        ab = AnnotationBbox(imagebox, xy=xy, xycoords="data", frameon=False)
        ax.add_artist(ab)


def test_reconstruction():
    """Test the reconstruction function"""

    # load JSON file
    import json

    with open("pilot1/demo_subj.json", "r") as f:
        data = json.load(f)
    data = pd.DataFrame(data["closing-grizzly"]["tasks"][8]["positions"])
    horizontal_locations = data["x"]
    ratings = data["y"]
    stim_folder = os.path.join(
        "pilot1", "controversial", "1567f4c8", "stimuli_info", "paired_stims"
    )
    face_pair_images = [
        os.path.join(stim_folder, fname + ".png") for fname in data["name"]
    ]
    for fname in face_pair_images:
        assert os.path.isfile(fname)

    reconstruct_trial(face_pair_images, ratings, horizontal_locations)
    plt.savefig("test_reconstruction.png")


def load_model_dissimilarities_xr(stimulus_set_folder, instance_id):
    folder = os.path.join(
        stimulus_set_folder,
        "representation_statistics",
    )

    parquet_file = os.path.join(
        folder, f"dissimilarities_instance_{instance_id}.parquet"
    )
    assert os.path.exists(parquet_file)
    dissimilarities = pd.read_parquet(parquet_file)

    model_dissimilarities = []
    for model in dissimilarities.model.unique():
        cur_model_df = dissimilarities[dissimilarities.model == model]
        cur_model_df.set_index(
            ["model", "instance", "i_layer", "i_trial", "i_pair"], inplace=True
        )
        model_dissimilarities.append(xr.Dataset.from_dataframe(cur_model_df))

    model_dissimilarities = xr.concat(model_dissimilarities, dim="model")

    return model_dissimilarities


def get_dissimilarity_data(exp_root_dir, inference_method="spearman_corr"):
    """load human data and model predictions for one trial"""
    analysis_root_dir = os.path.join(exp_root_dir, "analysis")
    stim_info_root_dir = os.path.join(exp_root_dir, "stimuli_info")

    condition_code = exp_root_dir.split("/")[-1]

    # load model predictions
    model_xr_path = os.path.join(
        analysis_root_dir,
        inference_method,
        f"model_performance_{inference_method}_{condition_code}.nc",
    )
    model_performance_xr = xr.load_dataarray(model_xr_path)

    model_dissimilarities = load_model_dissimilarities_xr(
        stim_info_root_dir, instance_id=0
    )

    # load human data
    sub_dissimilarities = xr.load_dataarray(
        os.path.join(analysis_root_dir, f"subj_data_{condition_code}.nc")
    )

    return model_performance_xr, model_dissimilarities, sub_dissimilarities


def get_best_layer(model_performance_xr):
    return (
        model_performance_xr.mean("i_trial")
        .mean("subj_id")
        .argmax(dim="i_layer")
        .drop("instance")
    )


def get_one_trial(
    trial_idx, model_dissimilarities, best_layers, sub_dissimilarities, model_label_map
):
    """mean human order and model predictions for one trial"""

    model_layer_trial = model_dissimilarities.sel({"i_trial": trial_idx}).drop(
        "instance"
    )["dissimilarity"]
    model_trial = []
    for model in model_layer_trial.model:
        best_layer = best_layers.sel({"model": model}).item()
        model_trial.append(
            model_layer_trial.sel({"model": model, "i_layer": best_layer})
        )
    model_trial = xr.concat(model_trial, dim="model")
    model_trial_rank = model_trial.rank(dim="i_pair")
    xr_model_order = [model_label_map[model] for model in model_trial_rank.model.values]
    model_trial_rank["model"] = xr_model_order

    sub_trial = sub_dissimilarities.mean(dim="i_rep").sel({"i_trial": trial_idx})
    sub_trial_rank = sub_trial.rank(dim="i_pair").mean(dim="subj_id").rank(dim="i_pair")
    sub_trial_rank = sub_trial_rank.expand_dims()
    sub_trial_rank["model"] = "mean human ranking"

    trial_rank = xr.concat(
        [sub_trial_rank, model_trial_rank.drop("i_layer")], dim="model"
    )

    return trial_rank.squeeze()


def face_column(
    trial_idx,
    dissimilarity_ratings,
    experiment_name="pilot1",
    stimulus_set_name="controversial",
    stimulus_set_code="dee5cdde",
    ax=None,
):
    """plot face pairs ranked by ratings

    args:
        trial_idx (int) which trial to plot
        dissimilarity_ratings (list) of ratings for each face pair (higher is more dissimilar)
        experiment_name (str) name of experiment (e.g. pilot1)
        stimulus_set_name (str) name of stimulus set (e.g. controversial)
        stimulus_set_code (str) code of stimulus set (e.g. dee5cdde)
        ax (matplotlib axis) axis to plot on
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(1, 3), dpi=300)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    # load all figure files
    stim_folder = os.path.join(
        experiment_name,
        stimulus_set_name,
        stimulus_set_code,
        "stimuli_info",
        "stimuli",
        "paired_stims",
    )
    files = glob.glob(
        os.path.join(stim_folder, f"optimized_trial_{trial_idx:02d}_pair_*.png")
    )
    assert len(files) == len(
        dissimilarity_ratings
    ), f"Number of files ({len(files)}) does not match number of ratings ({len(dissimilarity_ratings)})"
    im_arrays = []
    for file in files:
        im_arrays.append(np.array(Image.open(file)))

    # rank so most dissimilar is first
    positions = (
        scipy.stats.rankdata(-np.asarray(dissimilarity_ratings), method="ordinal") - 1
    )

    # concatenate images on top of each other
    concat_im_array = [im_arrays[i] for i in positions]
    concat_im_array = np.concatenate(concat_im_array, axis=0)

    ax.imshow(concat_im_array)
    ax.axis("off")


# face_column(trial_idx=0, dissimilarity_ratings=[7, 2,3,4,5,6])
# plt.show()
