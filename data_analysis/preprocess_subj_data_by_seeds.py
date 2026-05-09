import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import re
import glob
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import numpy as np
import xarray as xr
import pandas as pd
from data_analysis.analysis_utils import (
    PreprocessSubjData,
    generate_subj_datastruct,
    assign_cond,
    xr_spearmanrho_rae,
    normalize_subj_data,
)
import warnings


def get_best_rank_spearman(sub_data):
    # rank each trial for each subject
    ranked_data = sub_data.rank(dim="i_pair")
    # get normalized ranks before averaging across subjects
    mean_normalized_rank_data = normalize_subj_data(ranked_data).mean(dim="subj_id")
    # rank the average
    ranked_mean_rank = mean_normalized_rank_data.rank(dim="i_pair")

    return ranked_mean_rank


str2bool = lambda x: True if x.lower() == "true" else False

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="exp_data")
parser.add_argument(
    "--time_lowerbound",
    type=int,
    default=8,
    help="lower bound of median duration to be included in the analysis",
)
parser.add_argument(
    "--inference_method",
    type=str,
    default="spearman_corr",
    choices=["spearman_corr", "pearson_corr"],
)
parser.add_argument("--version_to_approve", type=int, required=False, default=None)
args = parser.parse_args()

exp_root_dir = args.experiment_dir
inference_method = args.inference_method
data_root_dir = os.path.join(exp_root_dir, "raw_data")
analysis_root_dir = os.path.join(exp_root_dir, "analysis")
Path(analysis_root_dir).mkdir(parents=True, exist_ok=True)
subj_data_dir = os.path.join(analysis_root_dir, "subj_data")
Path(subj_data_dir).mkdir(parents=True, exist_ok=True)
excluded_subj_data_dir = os.path.join(analysis_root_dir, "excluded_subj_data")
Path(excluded_subj_data_dir).mkdir(parents=True, exist_ok=True)
all_files = glob.glob(os.path.join(data_root_dir, "*"))
meadows_pattern = r"Meadows_face-arrangement_[a-z]_v_v(\d)"
experiments = np.unique(
    [
        re.search(meadows_pattern, os.path.basename(file)).group(0)
        for file in all_files
        if re.match(meadows_pattern, os.path.basename(file))
    ]
).tolist()

(
    metadata,
    excluded_subj,
    position_data,
    debrief,
    within_sub_reliability_df,
    task_data,
) = ([], [], [], [], [], {})

excluded_subjs_from_debrief = set(
    [
        "champion-shepherd",
        "eminent-jackass",
        "saved-puma",
        "wilderness-tortoise",
        "polar-killdeer",
        "known-jay",
        "freaky-alien",
        "legal-cougar",
    ]
)

# two seeds have 13 valid subjects rather than 12.
# one subject each is excluded based on the time order
# (the last subject that was collected in each seed is excluded)
excluded_subjs_from_debrief.update(["renewed-monkey", "assured-filly"])

with warnings.catch_warnings():
    # suppress the warnings.
    # e.g. some participants placed all face pairs at the same position,
    # which would lead to problems in computing correlation
    warnings.simplefilter("ignore")
    for i_exp, experiment in enumerate(experiments):
        task_data_path = os.path.join(
            data_root_dir,
            f"{experiment}_tree.json",
        )
        with open(task_data_path, "r") as f:
            exp_task_data = json.load(f)
        task_data.update(exp_task_data)

        metadata_path = os.path.join(data_root_dir, f"{experiment}_metadata.csv")
        exp_metadata = pd.read_csv(metadata_path)

        position_data_path = os.path.join(
            data_root_dir,
            f"{experiment}_annotations.csv",
        )
        exp_position_data = pd.read_csv(position_data_path)
        exp_metadata = assign_cond(exp_metadata, exp_position_data)
        exp_metadata = exp_metadata[~exp_metadata.generator.isna()]
        metadata.append(exp_metadata)
        position_data.append(exp_position_data)
    metadata = pd.concat(metadata).reset_index(drop=True)
    assert len(metadata.name) == len(set(metadata.name))
    position_data = pd.concat(position_data).reset_index(drop=True)
    metadata = metadata.groupby(["generator", "condition", "seed"])

    print("Check participant reliability...")
    n_subj, n_subj_excluded = 0, 0
    for (generator, condition, seed), cond_metadata in tqdm(metadata):
        arrangement_data, excluded_arrangement_data = [], []
        for i_subj, subj_id in enumerate(cond_metadata.name):
            subj_metadata = cond_metadata[cond_metadata.name == subj_id]
            version = subj_metadata.version.item()
            prolific_id = subj_metadata.PROLIFIC_PID.item()

            if subj_id in task_data.keys():
                subj_task_data = task_data[subj_id]["tasks"]
            else:
                subj_task_data = None

            subj_pos_data = position_data[position_data.participation == subj_id]
            subj_preprocess = PreprocessSubjData(
                subj_id,
                subj_metadata,
                subj_task_data,
                subj_pos_data,
                inference_method=inference_method,
            )
            (
                metadata_exclusion,
                exclusion_reason,
            ) = subj_preprocess.exclusion_metadata(args.time_lowerbound)
            stim2pos, exclusion_code = None, None
            if metadata_exclusion:
                exclusion_code = 3
            if not metadata_exclusion:
                (
                    taskdata_exclusion,
                    exclusion_reason,
                    stim2pos,
                    mean_y_corr,
                    median_y_corr,
                    mean_x_corr,
                    pval_y,
                    exclusion_code,
                ) = subj_preprocess.exclusion_arrangement_data()
                (
                    engaging,
                    difficult,
                    easy_comments,
                    difficult_comments,
                    comments,
                ) = subj_preprocess.get_subj_debrief()
                debrief.append(
                    pd.DataFrame(
                        {
                            "generator": [generator],
                            "condition": [condition],
                            "seed": [seed],
                            "subj_id": [subj_id],
                            "version": version,
                            "prolific_id": [prolific_id],
                            "how engaging (5 is the most engaging)": [engaging],
                            "how difficult (5 is the most difficult)": [difficult],
                            "what kind of images were most easy to rate?": [
                                easy_comments
                            ],
                            "what kind of images were most difficult to rate?": [
                                difficult_comments
                            ],
                            "comments": [comments],
                        }
                    )
                )

            if (
                metadata_exclusion
                or taskdata_exclusion
                or subj_preprocess.insufficient_trials_recorded
                or subj_id in excluded_subjs_from_debrief
            ):
                df = pd.DataFrame(
                    {
                        "generator": [generator],
                        "condition": [condition],
                        "date": subj_metadata.started.item().split(" ")[0],
                        "seed": [seed],
                        "participant_id": [subj_id],
                        "version": version,
                        "prolific_id": [prolific_id],
                        "exclusion_reason": exclusion_reason,
                        "stim2pos": [stim2pos],
                        "median_trial_dur": [subj_preprocess.median_trial_dur],
                        "exclusion_code": [exclusion_code],
                    }
                )
                excluded_subj.append(df)
                n_subj_excluded += 1

                if (
                    taskdata_exclusion
                    and getattr(subj_preprocess, "arrangement_data", None) is not None
                    or subj_id in excluded_subjs_from_debrief
                ):
                    subj_preprocess.arrangement_data["subj_id"] = [subj_id] * len(
                        subj_preprocess.arrangement_data
                    )
                    excluded_arrangement_data.append(subj_preprocess.arrangement_data)
            else:
                subj_preprocess.arrangement_data["subj_id"] = [subj_id] * len(
                    subj_preprocess.arrangement_data
                )
                arrangement_data.append(subj_preprocess.arrangement_data)
            within_sub_reliability_df.append(
                pd.DataFrame(
                    {
                        "generator": [generator],
                        "condition": [condition],
                        "seed": [seed],
                        "subj_id": [subj_id],
                        "version": version,
                        "prolific_id": [prolific_id],
                        "mean_y_corr": [mean_y_corr],
                        "median_y_corr": [median_y_corr],
                        "mean_x_corr": [mean_x_corr],
                        "pval_y": [pval_y],
                        "median_trial_dur": [subj_preprocess.median_trial_dur],
                        "exclude": (
                            True
                            if (taskdata_exclusion and exclusion_code == 2)
                            or subj_id in excluded_subjs_from_debrief
                            else False
                        ),
                    }
                )
            )
        if len(arrangement_data) == 0:
            continue
        arrangement_data = pd.concat(arrangement_data)
        arrangement_data = arrangement_data[
            ["subj_id"] + [col for col in arrangement_data.columns if col != "subj_id"]
        ]
        qualified_subjs = np.unique(arrangement_data.subj_id)
        n_subj += len(qualified_subjs)
        all_subj_data = generate_subj_datastruct(arrangement_data, qualified_subjs)
        assert len(all_subj_data.subj_id) == 12
        all_subj_data.to_netcdf(
            os.path.join(
                subj_data_dir, f"{generator}_{condition}_seed_{int(seed)}_subj_data.nc"
            )
        )

        if len(excluded_arrangement_data) > 0:
            excluded_arrangement_data = pd.concat(excluded_arrangement_data)
            excluded_arrangement_data = excluded_arrangement_data[
                ["subj_id"]
                + [col for col in excluded_arrangement_data.columns if col != "subj_id"]
            ]
            all_subj_data_y = generate_subj_datastruct(
                excluded_arrangement_data, np.unique(excluded_arrangement_data.subj_id)
            )
            all_subj_data_y.to_netcdf(
                os.path.join(
                    excluded_subj_data_dir,
                    f"{generator}_{condition}_seed_{int(seed)}_y_pos_excluded_subj_data.nc",
                )
            )
            all_subj_data_x = generate_subj_datastruct(
                excluded_arrangement_data,
                np.unique(excluded_arrangement_data.subj_id),
                col="x",
            )
            all_subj_data_x.to_netcdf(
                os.path.join(
                    excluded_subj_data_dir,
                    f"{generator}_{condition}_seed_{int(seed)}_x_pos_excluded_subj_data.nc",
                )
            )

    print("Number of subjects:", n_subj)
    print("Number of subjects excluded:", n_subj_excluded)


# run between-subject reliability and do t-tests
def compute_bet_subj_p_val(corrs):
    from scipy.stats import ttest_1samp

    fisher_transformed_bet_subj_y_corrs = np.arctanh(
        np.clip(corrs, -1 + 1e-8, 1 - 1e-8)
    )
    t_stat_between, pval_between = ttest_1samp(
        a=fisher_transformed_bet_subj_y_corrs, popmean=0, alternative="greater"
    )
    return pval_between


within_sub_reliability_df = pd.concat(within_sub_reliability_df)
subj_ids = set(
    within_sub_reliability_df[
        within_sub_reliability_df.exclude == False
    ].subj_id.tolist()
)
exclude_ids = set(
    within_sub_reliability_df[
        within_sub_reliability_df.exclude == True
    ].subj_id.tolist()
)

subj_data_dir = os.path.join(exp_root_dir, "analysis", "subj_data")
exclude_subj_data_dir = os.path.join(exp_root_dir, "analysis", "excluded_subj_data")
pattern = r"([a-z0-9-]+)_(random|controversial)_seed_(\d+)_subj_data.nc"
dfs = []
for subj_data_path in sorted(os.listdir(subj_data_dir)):
    match = re.match(pattern, subj_data_path)
    generator, cond, seed = match.group(1), match.group(2), match.group(3)

    data = xr.load_dataarray(os.path.join(subj_data_dir, subj_data_path))
    exclude_path = os.path.join(
        exclude_subj_data_dir, f"{generator}_{cond}_seed_{seed}_excluded_subj_data.nc"
    )

    for subj_id in data.subj_id.values:
        subj_data = data.loc[subj_id].mean("i_rep").rank("i_pair")
        other_data = data.sel(
            {"subj_id": list(set(data.subj_id.values.tolist()) - set([subj_id]))}
        )
        other_data = get_best_rank_spearman(other_data.mean("i_rep"))
        corr = xr_spearmanrho_rae(subj_data, other_data, dim="i_pair")
        within_sub_reliability_df.loc[
            within_sub_reliability_df.subj_id == subj_id, "between_subj_corr"
        ] = corr.mean("i_trial")
        within_sub_reliability_df.loc[
            within_sub_reliability_df.subj_id == subj_id, "between_subj_pval"
        ] = compute_bet_subj_p_val(corr.values)

    if os.path.exists(exclude_path):
        exclude_subj_data = xr.load_dataarray(exclude_path)
        exclude_subj_data = exclude_subj_data.sel(
            {
                "subj_id": [
                    id for id in exclude_subj_data.subj_id.values if id in exclude_ids
                ]
            }
        )
        for exclude_subj_id in exclude_subj_data.subj_id.values:
            subj_data = (
                exclude_subj_data.loc[exclude_subj_id].mean("i_rep").rank("i_pair")
            )
            other_data = get_best_rank_spearman(data.mean("i_rep"))
            corr = xr_spearmanrho_rae(subj_data, other_data, dim="i_pair")
            within_sub_reliability_df.loc[
                within_sub_reliability_df.subj_id == exclude_subj_id,
                "between_subj_corr",
            ] = corr.mean("i_trial")
            within_sub_reliability_df.loc[
                within_sub_reliability_df.subj_id == exclude_subj_id,
                "between_subj_pval",
            ] = compute_bet_subj_p_val(corr.values)

debrief = pd.concat(debrief)
debrief.to_csv(
    os.path.join(analysis_root_dir, f"debrief_summary.csv"),
    index=False,
)

if len(excluded_subj) > 0:
    excluded_subj = pd.concat(excluded_subj)
    excluded_subj.to_csv(
        os.path.join(analysis_root_dir, f"excluded_subj_{inference_method}.csv"),
        index=False,
    )


within_sub_reliability_df.to_csv(
    os.path.join(analysis_root_dir, f"within_subj_reliability_{inference_method}.csv"),
    index=False,
)
within_sub_reliability_df = within_sub_reliability_df[
    (~within_sub_reliability_df.pval_y.isna())
    & (within_sub_reliability_df.pval_y < 0.05)
]
print(
    "within subject reliability:\n",
    within_sub_reliability_df.groupby(["generator", "condition"]).mean_y_corr.mean(),
)

# get approved subjects for bulk approval
# version_to_approve = args.version_to_approve
# approved = within_sub_reliability_df = within_sub_reliability_df[
#     (~within_sub_reliability_df.pval_y.isna())
#     & (within_sub_reliability_df.pval_y < 0.05)
# ]
# if version_to_approve is not None:
#     approved = approved[approved.version == version_to_approve]
#     excluded_subj = excluded_subj[excluded_subj.version == version_to_approve]
# approved = set(approved.prolific_id.tolist())
# if len(excluded_subj) > 0:
#     approved = approved - set(excluded_subj.prolific_id.tolist())
#     borderline = set(
#         excluded_subj[excluded_subj.exclusion_code == 1].prolific_id.tolist()
#     )
#     approved = approved.union(borderline)
# print(",".join(approved))
