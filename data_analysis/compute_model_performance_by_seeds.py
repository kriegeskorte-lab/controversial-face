import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
import xarray as xr

from data_analysis.analysis_utils import (
    between_subj_reliability_pairs,
    between_subj_reliability_leave1out,
    normalize_subj_data,
    xr_spearmanrho_rae,
)


def load_model_dissimilarities_from_path(stimulus_set_folder, instance_id):
    """load model dissimilarities from parquet, format as a dict of xarray Datasets"""
    folder = os.path.join(
        stimulus_set_folder,
        "representation_statistics",
    )

    parquet_file = os.path.join(
        folder,
        f"dissimilarities_instance_{instance_id}.parquet",
    )
    assert os.path.exists(parquet_file), f"file {parquet_file} does not exist"
    dissimilarities = pd.read_parquet(parquet_file)

    dissimilarity_dict = {}
    for model in dissimilarities.model.unique():
        cur_model_df = dissimilarities[dissimilarities.model == model]
        cur_model_df.set_index(
            ["model", "instance", "i_layer", "i_trial", "i_pair"], inplace=True
        )
        dissimilarity_dict[model] = xr.Dataset.from_dataframe(cur_model_df)

    return dissimilarity_dict


if __name__ == "__main__":
    str2bool = lambda x: True if x.lower() == "true" else False
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, default="exp_data")
    parser.add_argument("--inference_method", type=str, default="spearman_corr")
    args = parser.parse_args()
    instance = 0

    exp_root_dir = args.experiment_dir
    analysis_root_dir = os.path.join(exp_root_dir, "analysis")
    subj_data_dir = os.path.join(analysis_root_dir, "subj_data")
    stimuli_info_dir = os.path.join(exp_root_dir, "stimuli_info")

    generators = ["bfm", "bfm-pose", "stylegan3"]
    conditions = ["random", "controversial"]
    between_subj_df, leave1out_df, noise_ceiling_df, model_performance = [], [], [], []
    for generator in tqdm(generators, desc="Generators"):
        for condition in tqdm(conditions, desc="Sampling strategies", leave=False):
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
                seed = int(seed_dir.split("_")[-1])
                subj_data_path = os.path.join(
                    subj_data_dir, f"{generator}_{condition}_{seed_dir}_subj_data.nc"
                )
                if not os.path.exists(subj_data_path):
                    continue

                sub_dissimilarities = xr.load_dataarray(
                    os.path.join(
                        subj_data_dir,
                        f"{generator}_{condition}_{seed_dir}_subj_data.nc",
                    )
                )
                sub_dissimilarities = normalize_subj_data(sub_dissimilarities)
                sub_dissimilarities = sub_dissimilarities.mean(
                    dim="i_rep"
                )  # average across repetitions; nanmean if one rep missing

                between_subj_reliability_p = between_subj_reliability_pairs(
                    sub_dissimilarities, inference_method=args.inference_method
                )
                (
                    cond_noise_ceiling_df,
                    cond_leave1out_df,
                ) = between_subj_reliability_leave1out(
                    sub_dissimilarities, inference_method=args.inference_method
                )

                cond_between_subj_df = pd.DataFrame(
                    {
                        "generator": [generator],
                        "condition": [condition],
                        "seed": [seed],
                        "pairwise": [between_subj_reliability_p],
                    }
                )
                between_subj_df.append(cond_between_subj_df)

                for df in [cond_noise_ceiling_df, cond_leave1out_df]:
                    df["generator"] = generator
                    df["condition"] = condition
                    df["seed"] = seed

                model_dissimilarity_dict = load_model_dissimilarities_from_path(
                    cond_exp_dir,
                    instance_id=instance,
                )
                comps = []
                for model in model_dissimilarity_dict.keys():
                    model_dissimilarities = model_dissimilarity_dict[model].sel(
                        instance=instance
                    )["dissimilarity"]
                    if args.inference_method == "pearson_corr":
                        comp = xr.corr(
                            model_dissimilarities, sub_dissimilarities, dim=["i_pair"]
                        )
                    elif args.inference_method == "spearman_corr":
                        comp = xr_spearmanrho_rae(
                            model_dissimilarities, sub_dissimilarities, dim="i_pair"
                        )
                    comp.name = "trial_corr"
                    comps.append(comp)

                cond_model_performance = xr.merge(comps).drop_vars(
                    "instance"
                )  # model x i_layer x i_trial x subject_id
                cond_model_performance = cond_model_performance.assign_coords(
                    generator=generator, condition=condition, seed=i_seed
                )
                cond_model_performance = cond_model_performance.expand_dims(
                    {
                        "generator": [generator],
                        "condition": [condition],
                        "seed": [i_seed],
                    }
                )
                subj_ids = cond_model_performance.subj_id.values.tolist()
                cond_leave1out_df["subj_id"] = cond_leave1out_df.apply(
                    lambda x: subj_ids.index(x.subj_id), axis=1
                )
                cond_noise_ceiling_df["subj_id"] = cond_noise_ceiling_df.apply(
                    lambda x: subj_ids.index(x.subj_id), axis=1
                )
                noise_ceiling_df.append(cond_noise_ceiling_df)
                leave1out_df.append(cond_leave1out_df)

                cond_model_performance = cond_model_performance.assign_coords(
                    subj_id=np.arange(len(subj_ids))
                )
                model_performance.append(cond_model_performance["trial_corr"])

    model_performance = xr.combine_by_coords(model_performance, join="outer")
    model_performance = model_performance["trial_corr"]
    assert model_performance.ndim == 7
    assert not np.any(
        np.isnan(model_performance)
    ), "some seed/model/condition have no data"

    best_layer = model_performance.mean("i_trial").copy(deep=True)
    best_layer = best_layer.mean("i_layer")  # drop i_layer dimension
    best_layer[...] = 0
    seeds = model_performance.seed.values

    for generator in model_performance.generator:
        for condition in model_performance.condition:
            # lopo_within_seed and loso_seed
            for seed in seeds:
                seed_data = model_performance.loc[generator, condition, seed].mean(
                    "i_trial"
                )
                for subj_id in seed_data.subj_id:
                    other_subj = seed_data.sel(
                        subj_id=seed_data.subj_id.values[seed_data.subj_id != subj_id]
                    )
                    best_performing_layer = other_subj.mean("subj_id").argmax("i_layer")
                    best_layer.loc[generator, condition, seed, :, subj_id] = (
                        best_performing_layer
                    )
    best_layer = best_layer.astype(np.int8)
    cv_model_performance = model_performance.isel(i_layer=best_layer)

    # check for nans (some conditions still miss a few subjects)
    orig_missing_mask = model_performance.isnull().all(dim=["i_layer", "i_trial"])
    cv_missing_mask = cv_model_performance.isnull().all(dim=["i_trial"])
    assert (orig_missing_mask == cv_missing_mask).all().item()

    df_cv_model_performance = (
        cv_model_performance.to_dataframe().reset_index().drop(columns="i_layer")
    )
    df_cv_model_performance = df_cv_model_performance[
        ~df_cv_model_performance.trial_corr.isna()
    ]
    df_cv_model_performance = df_cv_model_performance.rename(
        columns={"trial_corr": "cv_corr"}
    )

    model_performance.to_netcdf(
        os.path.join(
            analysis_root_dir,
            f"model_performance_{args.inference_method}.nc",
        )
    )
    between_subj_df = pd.concat(between_subj_df)
    between_subj_df.to_csv(
        os.path.join(
            analysis_root_dir,
            f"between_subj_reliability_{args.inference_method}.csv",
        )
    )
    leave1out_df = pd.concat(leave1out_df)
    leave1out_df.to_csv(
        os.path.join(
            analysis_root_dir,
            f"leave1out_{args.inference_method}.csv",
        ),
        index=False,
    )
    df_cv_model_performance = pd.concat([df_cv_model_performance, leave1out_df])
    assert (
        df_cv_model_performance.groupby(
            ["condition", "generator", "seed", "model", "i_trial"]
        )
        .size()
        .unique()
        == 12
    )

    df_cv_model_performance.to_csv(
        os.path.join(
            analysis_root_dir,
            f"cv_model_performance_{args.inference_method}.csv",
        ),
        index=False,
    )
    noise_ceiling_df = pd.concat(noise_ceiling_df)
    noise_ceiling_df.to_csv(
        os.path.join(
            analysis_root_dir,
            f"noise_ceiling_{args.inference_method}.csv",
        ),
        index=False,
    )
