import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import xarray as xr
import argparse
import numpy as np
import pandas as pd
import json
import PIL.Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
from vis.plot_model_performance_cond import (
    metroplot,
)
from data_analysis.analysis_utils import normalize_subj_data, get_best_rank_spearman

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="exp_data")
parser.add_argument("--method", type=str, default="spearman_corr")
args = parser.parse_args()

method = args.method
experiment_dir = args.experiment_dir
analysis_dir = os.path.join(experiment_dir, "analysis")

layer_names_raw = layer_names.copy()
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

model_performance = xr.load_dataarray(
    os.path.join(
        analysis_dir,
        f"model_performance_{method}.nc",
    )
)
model_labels = [model_label_map[model] for model in model_performance.model.values]
model_performance = model_performance.assign_coords({"model": model_labels})

# get average layerwise data
layerwise_mean = (
    model_performance.mean(dim=("i_trial", "subj_id", "seed", "generator", "condition"))
    .to_dataframe()
    .reset_index()
)
concat_model_performance = model_performance.mean(dim=("i_trial", "subj_id")).stack(
    concat=("seed", "generator", "condition")
)
layerwise_ste = concat_model_performance.std(dim="concat") / np.sqrt(
    concat_model_performance.concat.size
)
layerwise_ste = layerwise_ste.to_dataframe().reset_index()

noise_ceiling_df = pd.read_csv(
    os.path.join(analysis_dir, f"noise_ceiling_{method}.csv"),
    index_col=0,
)
lower_nc, upper_nc = (
    noise_ceiling_df.lower_noise_ceiling.mean(),
    noise_ceiling_df.upper_noise_ceiling.mean(),
)

# get cv data
df_cv_model_performance = pd.read_csv(
    os.path.join(analysis_dir, f"cv_model_performance_{method}.csv")
)
df_cv_model_performance["model"] = df_cv_model_performance.apply(
    lambda x: model_label_map[x.model], axis=1
)
df_cv_model_performance = (
    df_cv_model_performance.groupby(["generator", "condition", "seed", "model"])[
        "cv_corr"
    ]
    .mean()
    .reset_index()
)

# get stats results
pooled_stats = pd.read_csv(
    os.path.join(analysis_dir, "pooled_pairwise_satterthwaite.csv")
)
pooled_stats.rename(columns={"p.value": "p.value.fdr"}, inplace=True)

condition_stats = pd.read_csv(
    os.path.join(analysis_dir, "cond_pairwise_satterthwaite.csv")
)
for i_df, df in enumerate([pooled_stats, condition_stats]):
    for level in [1, 2]:
        df[f"level{level}"] = df["contrast"].apply(
            lambda x: x.split("-")[level - 1].strip()
        )
        df[f"level{level}"] = df[f"level{level}"].apply(lambda x: model_label_map[x])
    df.drop(columns=["contrast"], inplace=True)
    df.rename(columns={"p.value.fdr": "p_corrected"}, inplace=True)

    mask = np.sign(df.estimate) < 0  # swap level1 and level2 to have positive estimates
    df.loc[mask, ["level1", "level2"]] = df.loc[mask, ["level1", "level2"]].to_numpy()[
        :, ::-1
    ]
    df.loc[mask, "estimate"] *= -1
    df["is_sig"] = df.apply(
        lambda x: True if x["p_corrected"] <= 0.05 else False, axis=1
    )

pooled_stats = pooled_stats[pooled_stats.is_sig == True]
pooled_stats["effect_direction"] = np.sign(pooled_stats.estimate)
pooled_stats = pooled_stats[
    ["level1", "level2", "effect_direction", "is_sig", "estimate", "SE", "p_corrected"]
]
pooled_stats.sort_values(by=["level1", "level2"], inplace=True)
condition_stats = condition_stats[condition_stats.is_sig == True]

col_elements = [
    "panel1_margin",
    "cv_dot_plot",
    "cv_margin",
    "metroplot",
    "panel2_margin",
    "layerwise_plot",
    "right_margin",
]
row_elements = ["top_margin", "plot", "bottom_margin"]

panel1_margin = 1.15
right_margin = 0.03
panel2_margin = 0.5
cv_margin = 0.1  # between cv dot plot and metroplot

cv_dot_w = 1.45
metroplot_w = 0.7
layerwise_w = 1.3

top_margin = 0.4
bottom_margin = 0.5
plot_h = (3.85 - bottom_margin - top_margin) / 2

column_widths = (
    [panel1_margin]
    + [cv_dot_w]
    + [cv_margin]
    + [metroplot_w]
    + [panel2_margin]
    + [layerwise_w]
    + [right_margin]
)
row_heights = [top_margin] + [plot_h] + [bottom_margin]
fig_w = np.sum(column_widths)
fig_h = np.sum(row_heights)
assert fig_w <= 7.08, fig_w  # 180mm -> 7.08 inches
assert len(column_widths) == len(col_elements) and len(row_heights) == len(row_elements)

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

layerwise_plot_ax = fig.add_subplot(
    gs[row_elements.index("plot"), col_elements.index("layerwise_plot")]
)
plot_layer_performance(
    layerwise_plot_ax,
    layerwise_mean,
    layerwise_ste,
    model_label_map,
    model_order,
    model_palette,
    layer_names,
    lower_noise_ceiling=lower_nc,
    upper_noise_ceiling=upper_nc,
    alpha=0.3,
    linewidth=0.8,
)
model = list(layer_names.keys())[0]
layerwise_plot_ax.set_xticks(range(len(layer_names[model])))
layerwise_plot_ax.set_xticklabels(
    layer_names[model], rotation=90, fontsize=not_important_tick_label_fontsize
)
layerwise_plot_ax.set_ylim([0.0, 0.65])
layerwise_plot_ax.set_ylabel(
    "prediction accuracy\n" + r"(Spearman's $\mathit{\rho}$)",
    fontsize=axes_label_fontsize,
    labelpad=0,
)
layerwise_plot_ax.tick_params(axis="y", labelsize=not_important_tick_label_fontsize)
layerwise_plot_ax.set_title(
    "(b) layerwise correlation\npooled across conditions",
    fontsize=panel_title_fontsize,
    x=0.47,
    y=1,
)

cv_plot_ax = fig.add_subplot(
    gs[row_elements.index("plot"), col_elements.index("cv_dot_plot")]
)
plot_cv_model_performance_figure(
    cv_plot_ax,
    df_cv_model_performance,
    upper_nc,
    lower_nc,
    model_order,
    model_palette,
    alpha=0.3,
    dotsize=3,
    edgewidth=0.4,
    vertsize=8,
)
cv_plot_ax.set_xlabel(
    "best layer prediction accuracy\n"
    + r"(Spearman's $\mathit{\rho}$, cross-validated)",
    fontsize=axes_label_fontsize,
    labelpad=2,
)
cv_plot_ax.set_title(
    "(a) cross-validated model performance\npooled across conditions",
    fontsize=panel_title_fontsize,
    x=0.6,
    y=1,
)
cv_plot_ax.set_yticklabels([])
cv_plot_ax.set_ylabel("")
cv_plot_ax.set_xlim([-0.1, 0.8])
cv_plot_ax.set_xticks(np.linspace(0, 0.8, 5))
cv_plot_ax.set_xticklabels(
    [0, 0.2, 0.4, 0.6, 0.8], fontsize=not_important_tick_label_fontsize
)

x_main = -0.07
x_sub = -0.07
offset = 0.18
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
    gs[row_elements.index("plot"), col_elements.index("metroplot")]
)
dominating_models = set(
    pooled_stats[pooled_stats.effect_direction == 1].level1.tolist()
)
dominating_models.update(
    pooled_stats[pooled_stats.effect_direction == -1].level2.tolist()
)
dominating_models = [model for model in model_order if model in dominating_models]

level_to_location_y = {
    cond: i for i, cond in enumerate(model_order)
}  # map categories to y axis locations.
level_to_location_x = {model: i for i, model in enumerate(dominating_models)}
metroplot(
    pooled_stats,
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
    f"{experiment_dir}/figures/fig7_pooled_model_performance.pdf",
    dpi=300,
)
