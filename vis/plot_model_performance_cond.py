import os, sys

# add project root to path for importing libraries
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import glob
import itertools
import xarray as xr
import argparse
import numpy as np
import pandas as pd
import json
import PIL.Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from vis.metroplot import metroplot
from vis.style import (
    model_label_map,
    model_palette,
    model_order,
    layer_names,
    model_abbrev_map,
)
from vis.plot_utils import (
    plot_cv_model_performance_figure,
    plot_layer_performance,
)
from data_analysis.analysis_utils import normalize_subj_data, get_best_rank_spearman
import warnings

warnings.filterwarnings("ignore")

import matplotlib.font_manager as fm

font_path = "HelveticaNeue.ttc"
fm.fontManager.addfont(font_path)

prop = fm.FontProperties(fname=font_path)
print("registered name:", prop.get_name())

plt.rcParams["font.family"] = prop.get_name()


def metroplot(
    df,
    level_to_location_y,
    level_to_location_x,
    metroplot_element_order,
    dominating_effect_direction=1,
    ax=None,
    level_pallete=None,
    level_axis_ylim=None,
    level_axis_xlim=None,
    element_axis_lim=None,
    open_dot_fill_color="w",
    marker="o",
    linewidth=0.5,
    markeredgewidth=0.5,
    markersize=8,
):
    """Plot a 'metroplot' pairwise comparisons significance plot.

    Each row in df should describe the outcome of one pair-wise comparison.

    args:
    df (pd.DataFrame) with the columns: level1 (str), level2 (str), effect_direction (1|-1), is_sig (bool).
    level_to_location (dict) a dictionary mapping level names (as in level1 and level2 in df) to locations on level_axis.
        Generally, this should correspond with the tick locations of the categories in the main plot.
    metroplot_element_order (list) list of strings - the order in which the metroplot elements should be plotted.
    level_axis (str) 'x'|'y' which axis should be used to plot levels. For example, use 'x' for horizontal bar plots.
    dominating_effect_direction (int) -1 or 1. Changing this flips the roles of open and closed markers.
    ax (matplotlib.axes._subplots.AxesSubplot) axes handle for plotting the metroplot.
    level_pallete (dict) a dictionary mapping levels to colors. Alternatively, you pass a single color.
    level_axis_lim (tuple) the axis limits of level_axis. Typically this should match the limits of the main plot.
    dot_fill_color (color) the fill color of open ("dominated") levels
    marker, linewidth, markeredgewidth and markersize are fed to plt.plot and control the elements' appearance
    """

    if ax is None:
        ax = plt.gca()

    # eliminate comparisons between conditions that don't appear in level_to_location
    df = df[df.level1.isin(level_to_location_y) | df.level2.isin(level_to_location_y)]

    # eliminate non-significant comparisons
    df = df[df.is_sig]

    for dominating_level in metroplot_element_order:
        if dominating_level is None:
            continue

        # find dominated levels
        row_filter = (
            (df.level1 == dominating_level)
            & (df.level2.isin(level_to_location_y))
            & (df["effect_direction"] == dominating_effect_direction)
        )
        dominated_levels_list = list(df[row_filter].level2)
        row_filter = (
            (df.level2 == dominating_level)
            & (df.level1.isin(level_to_location_y))
            & (df["effect_direction"] == -dominating_effect_direction)
        )
        dominated_levels_list.extend(list(df[row_filter].level1))

        if len(dominated_levels_list) == 0:
            continue

        # the following notation assumes the level_axis is y.
        x = []
        y = []
        c_fill = []
        c_edge = []

        if isinstance(level_pallete, dict):
            element_color = level_pallete[dominating_level]
        elif level_pallete is not None:
            element_color = level_pallete
        else:
            element_color = "k"

        # add points to represent the dominated level
        for dominated_level in dominated_levels_list:
            y.append(level_to_location_y[dominated_level])
            x.append(level_to_location_x[dominating_level])
            c_fill.append(open_dot_fill_color)
            c_edge.append(element_color)

        # add a point to represent the dominating level
        y.append(level_to_location_y[dominating_level])
        x.append(level_to_location_x[dominating_level])
        c_fill.append(element_color)
        c_edge.append(element_color)

        ax.plot(
            x, y, "-", color=element_color, clip_on=False, linewidth=linewidth
        )  # plot line
        for i in range(len(x)):  # plot dots
            ax.plot(
                x[i],
                y[i],
                marker,
                markerfacecolor=c_fill[i],
                markeredgecolor=c_edge[i],
                clip_on=False,
                markeredgewidth=markeredgewidth,
                markersize=markersize,
            )  # plot markers

    ax.set_ylim(level_axis_ylim)
    ax.set_xlim(level_axis_xlim)

    ax.axis("off")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, default="exp_data")
    parser.add_argument("--method", type=str, default="spearman_corr")
    parser.add_argument(
        "--stats_result_fn", type=str, default="cond_pairwise_satterthwaite.csv"
    )
    args = parser.parse_args()
    method = args.method
    experiment_dir = args.experiment_dir
    analysis_dir = os.path.join(experiment_dir, "analysis")

    layer_ylim_map = {
        "bfm": [-0.1, 0.7],
        "bfm-pose": [-0.3, 0.7],
        "stylegan3": [-0.1, 0.7],
    }
    fig_map = {
        "bfm": "fig4",
        "bfm-pose": "fig5",
        "stylegan3": "fig6",
    }

    # some seed's example trials have the same ranking
    # so we need to manually adjust the order
    seed_map = {"bfm-pose": 2}

    layer_names_raw = layer_names
    layer_selector = ["conv", "fc"]
    for model in layer_names_raw.keys():
        layer_names_raw[model] = [
            layer
            for layer in layer_names_raw[model]
            if layer_selector[0] in layer or layer_selector[1] in layer
        ]

    model_order_main = [model_abbrev_map[model] for model in model_order]
    model_order = [model_label_map[model] for model in model_order]
    model_palette = {
        model_label_map[model]: color for model, color in model_palette.items()
    }
    layer_names = {}
    for model, model_layers in layer_names_raw.items():
        layer_names[model_label_map[model]] = [
            model_layer.split("_")[0] + " " + model_layer.split("_")[1]
            for model_layer in model_layers
        ]

    ## DATA
    noise_ceiling_df = pd.read_csv(
        os.path.join(analysis_dir, f"noise_ceiling_{method}.csv"),
        index_col=0,
    )

    # model performance xarray: ('generator', 'condition', 'seed', 'model', 'i_layer', 'i_trial', 'subj_id')
    model_performance = xr.load_dataarray(
        os.path.join(analysis_dir, f"model_performance_{method}.nc")
    )
    model_labels = [model_label_map[model] for model in model_performance.model.values]
    model_performance = model_performance.assign_coords({"model": model_labels})

    # CV performance
    df_cv_model_performance = pd.read_csv(
        os.path.join(analysis_dir, f"cv_model_performance_{method}.csv")
    )
    df_cv_model_performance = (
        df_cv_model_performance.groupby(["generator", "condition", "seed", "model"])[
            "cv_corr"
        ]
        .mean()
        .reset_index()
    )
    df_cv_model_performance = (
        df_cv_model_performance.groupby(["generator", "condition", "seed", "model"])[
            "cv_corr"
        ]
        .mean()
        .reset_index()
    )
    df_cv_model_performance["model"] = df_cv_model_performance.apply(
        lambda x: model_label_map[x.model], axis=1
    )

    # statistical results on model comparison
    statistical_results = pd.read_csv(os.path.join(analysis_dir, args.stats_result_fn))
    statistical_results["level1"] = statistical_results.apply(
        lambda x: x.contrast.split(" - ")[0], axis=1
    )
    statistical_results["level2"] = statistical_results.apply(
        lambda x: x.contrast.split(" - ")[1], axis=1
    )
    statistical_results["level1"] = statistical_results.apply(
        lambda x: model_label_map[x.level1], axis=1
    )
    statistical_results["level2"] = statistical_results.apply(
        lambda x: model_label_map[x.level2], axis=1
    )

    statistical_results = statistical_results.drop(columns="contrast")
    statistical_results = statistical_results.rename(
        columns={"t.ratio": "t", "p.value.fdr": "p_corrected"}
    )
    statistical_results["effect_direction"] = np.sign(statistical_results.estimate)
    statistical_results["is_sig"] = statistical_results.apply(
        lambda x: True if x.p_corrected <= 0.05 else False, axis=1
    )

    conditions = sorted(model_performance.condition.values)[::-1]
    for generator in model_performance.generator.values:
        generator_model_performance = model_performance.loc[generator]
        generator_cv_model_performance = df_cv_model_performance[
            df_cv_model_performance.generator == generator
        ]
        generator_noise_ceiling = noise_ceiling_df[
            noise_ceiling_df.generator == generator
        ]
        generator_statistical_results = statistical_results[
            statistical_results.generator == generator
        ]

        sig_results = generator_statistical_results[
            generator_statistical_results.is_sig == True
        ]
        dominating_models = set(
            sig_results[sig_results.effect_direction == 1].level1.tolist()
        )
        dominating_models.update(
            sig_results[sig_results.effect_direction == -1].level2.tolist()
        )
        dominating_models = [
            model for model in model_order if model in dominating_models
        ]

        level_to_location_y = {
            cond: i for i, cond in enumerate(model_order)
        }  # map categories to y axis locations.
        level_to_location_x = {model: i for i, model in enumerate(dominating_models)}

        stim_root_dir = os.path.join(experiment_dir, "stimuli_info", generator)
        # panel 1: example trial panel
        example_trials = [0, 5, 11]
        n_example_trials = len(example_trials)
        n_pairs = 6
        width_crop = 70
        height_crop = 30
        panel_1_col_order = list(
            itertools.chain(
                *[
                    (f"trial_{i_trial}_pair", "trial_space")
                    for i_trial in range(n_example_trials)
                ]
            )
        )
        panel_1_row_order = list(
            itertools.chain(
                *[
                    (f"{cond}_pair_{i_pair}", "pair_space")
                    for cond in conditions
                    for i_pair in range(n_pairs)
                ]
            )
        )
        panel_1_row_order[n_pairs * 2 - 1] = "middle_margin"
        # panel 2: cv plot
        panel_2_col_order = ["cv_dot_plot", "cv_margin", "metroplot"]
        # panel 3: layerwise plot
        panel_3_col_order = ["layerwise_plot"]

        col_elements = (
            ["left_margin"]
            + panel_1_col_order[:-1]
            + ["panel_1_margin"]
            + panel_2_col_order
            + ["panel_2_margin"]
            + panel_3_col_order
            + ["right_margin"]
        )
        row_elements = ["top_margin"] + panel_1_row_order[:-1] + ["bottom_margin"]

        left_margin = 0.4
        # arrow_w = 0.2
        panel1_margin = 1.15
        panel2_margin = 0.5
        right_margin = 0.03
        face_w = 0.45
        trial_space = 0.05
        layerwise_w = 1.3
        cv_dot_w = 1.45
        cv_margin = 0.1
        metroplot_w = 0.7

        top_margin = 0.2
        cond_pair_h = face_w / 2
        pair_space = 0.02
        middle_margin = 0.25
        bottom_margin = 0.5

        trial_widths = [face_w, trial_space] * n_example_trials
        column_widths = (
            [left_margin]
            + trial_widths[:-1]
            + [
                panel1_margin,
                cv_dot_w,
                cv_margin,
                metroplot_w,
                panel2_margin,
                layerwise_w,
                right_margin,
            ]
        )
        trial_heights = [cond_pair_h, pair_space] * n_pairs
        row_heights = (
            [top_margin]
            + trial_heights[:-1]
            + [middle_margin]
            + trial_heights[:-1]
            + [bottom_margin]
        )
        fig_w = np.sum(column_widths)
        fig_h = np.sum(row_heights)
        assert fig_w <= 7.08, fig_w  # 180mm -> 7.08 inches
        assert len(column_widths) == len(col_elements) and len(row_heights) == len(
            row_elements
        )
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
        fig.set_size_inches(fig_w, fig_h)
        print("figure size: ", fig.get_size_inches())
        gs = GridSpec(
            ncols=len(col_elements),
            nrows=len(row_elements),
            figure=fig,
            width_ratios=column_widths,
            height_ratios=row_heights,
            hspace=0,
            wspace=0,
            top=1,
            bottom=0,
            left=0,
            right=1,
        )

        panel_title_fontsize = 9
        axes_label_fontsize = 7.5
        not_important_tick_label_fontsize = 7

        for i_cond, condition in enumerate(conditions):
            layerwise_cond_xr = (
                generator_model_performance.loc[condition]
                .mean("i_trial")
                .mean("subj_id")
            )
            layerwise_avg_df = (
                layerwise_cond_xr.mean("seed").to_dataframe().reset_index()
            )
            layerwise_se_df = (
                (layerwise_cond_xr.std("seed") / np.sqrt(12))
                .to_dataframe()
                .reset_index()
            )

            cond_cv_df = generator_cv_model_performance[
                generator_cv_model_performance.condition == condition
            ]

            noise_ceiling = generator_noise_ceiling[
                generator_noise_ceiling.condition == condition
            ]
            lower_noise_ceiling = noise_ceiling.lower_noise_ceiling.mean().item()
            upper_noise_ceiling = noise_ceiling.upper_noise_ceiling.mean().item()
            model_comparison = generator_statistical_results[
                generator_statistical_results.condition == condition
            ]

            stim_dir = sorted(
                glob.glob(os.path.join(stim_root_dir, condition, "seed_**"))
            )[3]
            seed = int(stim_dir.split("/")[-1].split("_")[-1])
            stim_dir = os.path.join(stim_dir, "ims", "paired_stims")
            subject_data_path = f"{experiment_dir}/analysis/subj_data/{generator}_{condition}_seed_{seed}_subj_data.nc"
            subject_data = xr.load_dataarray(subject_data_path)

            for i_trial, trial_id in enumerate(example_trials):
                trial_data = normalize_subj_data(
                    subject_data.loc[dict(i_trial=trial_id)]
                )
                trial_data = trial_data.mean("i_rep")
                trial_rank = get_best_rank_spearman(trial_data)

                current_ranks = set()
                for i_pair in range(n_pairs):
                    pair_rank = n_pairs - int(trial_rank[i_pair])  # zero index
                    if pair_rank in current_ranks:
                        pair_rank -= 1
                    current_ranks.add(pair_rank)
                    stim_path = os.path.join(
                        stim_dir,
                        f"{generator}_{condition}_seed_{seed}_optimized_trial_{str(trial_id).zfill(2)}_pair_{str(i_pair).zfill(2)}.png",
                    )
                    pair_plot_ax = fig.add_subplot(
                        gs[
                            row_elements.index(f"{condition}_pair_{pair_rank}"),
                            col_elements.index(f"trial_{i_trial}_pair"),
                        ]
                    )
                    stim = PIL.Image.open(stim_path)
                    stim_size = stim.size[0]
                    if generator == "stylegan3":
                        stim.crop((50, 0, stim_size - 50, 0))
                    pair_plot_ax.imshow(stim)
                    pair_plot_ax.axis("off")
                    if (
                        i_trial == n_example_trials // 2
                        and i_cond == 0
                        and pair_rank == 0
                    ):
                        pair_plot_ax.set_title(
                            "(a) three example trials",
                            fontsize=panel_title_fontsize,
                            x=0.5,
                        )
                    if i_trial == 0 and pair_rank == 0:
                        y = -3.75 if condition == "controversial" else -3.1
                        x = -0.9
                        if generator == "stylegan3":
                            if condition == "controversial":
                                y -= 2.25
                            else:
                                y -= 1.8
                            x += 0.2
                        pair_plot_ax.annotate(
                            condition,
                            xy=(x, y),
                            xycoords="axes fraction",
                            ha="center",
                            rotation=90,
                            fontsize=panel_title_fontsize,
                        )
                        # face_plot_ax.set_title(f"example trial {i_trial}", fontsize=not_important_tick_label_fontsize, loc="left")

            row_i = row_elements.index(f"{condition}_pair_0")
            row_j = row_elements.index(f"{condition}_pair_{n_pairs-1}")

            # draw an upward arrow
            arrow_ax = fig.add_subplot(
                gs[row_i : row_j + 1, col_elements.index("trial_0_pair")]
            )
            arrow_ax.axis("off")
            arrow_ax.annotate(
                "",
                xy=(-0.15, 0.95),
                xycoords="axes fraction",  # Arrow head (top)
                xytext=(-0.15, 0),
                textcoords="axes fraction",  # Arrow tail (bottom)
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            )
            arrow_ax.text(
                -0.25,
                0.5,
                "more dissimilar",
                transform=arrow_ax.transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=axes_label_fontsize,  # or panel_title_fontsize depending on your preference
            )

            layerwise_plot_ax = fig.add_subplot(
                gs[row_i : row_j + 1, col_elements.index("layerwise_plot")]
            )
            plot_layer_performance(
                layerwise_plot_ax,
                layerwise_avg_df,
                layerwise_se_df,
                model_label_map,
                model_order,
                model_palette,
                layer_names,
                lower_noise_ceiling,
                upper_noise_ceiling,
                alpha=0.3,
            )
            model = list(layer_names.keys())[0]
            layerwise_plot_ax.set_xticks(range(len(layer_names[model])))
            layerwise_plot_ax.set_xticks([])
            if i_cond == 1:
                layerwise_plot_ax.set_xticks(range(len(layer_names[model])))
                layerwise_plot_ax.set_xticklabels(
                    layer_names[model],
                    rotation=90,
                    fontsize=not_important_tick_label_fontsize,
                )
            layerwise_plot_ax.set_ylim(layer_ylim_map[generator])
            layerwise_plot_ax.set_ylabel(
                "prediction accuracy\n" + r"(Spearman's $\mathit{\rho}$)",
                fontsize=axes_label_fontsize,
            )
            layerwise_plot_ax.yaxis.set_label_coords(-0.17, 0.5)
            layerwise_plot_ax.tick_params(
                axis="y", labelsize=not_important_tick_label_fontsize
            )
            if i_cond == 0:
                layerwise_plot_ax.set_title(
                    "(c) layerwise correlation", fontsize=panel_title_fontsize
                )

            cv_plot_ax = fig.add_subplot(
                gs[row_i : row_j + 1, col_elements.index(f"cv_dot_plot")]
            )
            plot_cv_model_performance_figure(
                cv_plot_ax,
                cond_cv_df,
                upper_noise_ceiling,
                lower_noise_ceiling,
                model_order,
                model_palette,
                alpha=0.3,
                dotsize=4,
                edgewidth=0.5,
                vertsize=10,
            )
            cv_plot_ax.set_yticklabels([])
            cv_plot_ax.set_ylabel("", fontsize=axes_label_fontsize)
            if i_cond == 1:
                cv_plot_ax.set_xlabel(
                    "best layer prediction accuracy\n"
                    + r"(Spearman's $\mathit{\rho}$, cross-validated)",
                    fontsize=axes_label_fontsize,
                    labelpad=2,
                )
            else:
                cv_plot_ax.set_title(
                    "(b) cross-validated model performance",
                    fontsize=panel_title_fontsize,
                    x=0.7,
                    y=1,
                )
                cv_plot_ax.set_xlabel("")
            cv_plot_ax.set_xticks(np.linspace(0, 1, 5))
            cv_plot_ax.set_xticklabels(
                np.linspace(0, 1, 5), fontsize=not_important_tick_label_fontsize
            )
            cv_plot_ax.set_xlim([-0.1, 0.8])

            x_main = -0.05
            x_sub = -0.05
            offset = 0.17

            yticks = range(len(model_order))
            for y, main_label, small_label in zip(
                yticks,
                model_order_main,
                model_order,
            ):
                cv_plot_ax.text(
                    x_main,
                    y - offset,  # vertical offset in data coordinates; tune this
                    main_label,
                    transform=cv_plot_ax.get_yaxis_transform(),  # x in axes coords, y in data coords
                    ha="right",
                    va="center",
                    fontsize=not_important_tick_label_fontsize,
                    fontweight="bold",
                )

                cv_plot_ax.text(
                    x_sub,
                    y + offset,  # vertical offset in data coordinates; tune this
                    small_label,
                    transform=cv_plot_ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=not_important_tick_label_fontsize * 0.75,
                    fontweight="normal",
                    color="dimgray",
                )

            metro_plot_ax = fig.add_subplot(
                gs[row_i : row_j + 1, col_elements.index("metroplot")]
            )

            metroplot(
                model_comparison,
                level_to_location_y=level_to_location_y,
                level_to_location_x=level_to_location_x,
                metroplot_element_order=model_order,
                ax=metro_plot_ax,
                dominating_effect_direction=1,
                level_pallete=model_palette,
                level_axis_ylim=cv_plot_ax.get_ylim(),
                level_axis_xlim=[0, len(dominating_models)],
                markersize=4.5,
                linewidth=0.5,
                markeredgewidth=0.5,
            )
            metro_plot_ax.set_xlim([0, 7])

        fig.savefig(
            f"{experiment_dir}/figures/{fig_map[generator]}_{generator}_model_performance.pdf",
            dpi=300,
        )
