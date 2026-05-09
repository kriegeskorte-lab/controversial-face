import re
from collections import OrderedDict
from tracemalloc import start
import math
import pandas as pd
import numpy as np
import xarray as xr
import itertools
from scipy.stats import ttest_1samp, ttest_rel, spearmanr
from datetime import datetime
import warnings


class PreprocessSubjData:
    def __init__(
        self,
        subj_id,
        metadata,
        task_data,
        pos_data,
        inference_method="spearman",
        num_tasks=12,
    ):
        """
        Args:
            metadata (pandas.dataframe): one subject metadata
            task_data (list): list of task dictionaries
            pos_data (pandas.dataframe): one subject position data for the arrangement tasks
        """
        self.subj_id = subj_id
        self.metadata = metadata
        if task_data is not None:
            self.task_data = task_data
            self.task_names = [task["task"]["name"] for task in task_data]
        self.pos_data = pos_data
        self.median_trial_dur = None
        self.num_tasks = num_tasks
        self.inference_method = inference_method

    def preprocess_position_data(self):
        practice_data = self.pos_data[self.pos_data.task == "practice_check"]
        self.practice_data = practice_data[["stim1_name", "x", "y"]]

        arrangement_data = self.pos_data[
            self.pos_data.task == "face_arrangement_trials"
        ]
        arrangement_data.loc[:, ["trial_start"]] = pd.to_datetime(
            arrangement_data["time_trial_start"]
        )
        arrangement_data.loc[:, ["trial_end"]] = pd.to_datetime(
            arrangement_data["time_trial_response"]
        )
        trial_dur = (
            arrangement_data["trial_end"] - arrangement_data["trial_start"]
        ).dt.total_seconds()
        median_trial_dur = trial_dur[5::6].median()
        self.median_trial_dur = median_trial_dur

        arrangement_data = arrangement_data[["trial", "stim1_name", "x", "y"]]
        pattern = r"optimized_trial_(\d+)_pair_(\d+)"
        arrangement_data["trial_idx"] = arrangement_data.stim1_name.str.extract(
            pattern
        )[0].astype(int)
        arrangement_data["pair"] = arrangement_data.stim1_name.str.extract(pattern)[
            1
        ].astype(int)
        arrangement_data = arrangement_data.drop(["stim1_name"], axis=1)

        # the data is still usable despite missing repetitions
        self.insufficient_trials_recorded = (
            True if len(np.unique(arrangement_data.trial_idx)) != 12 else False
        )

        # check if the data misses repetitions
        num_unique_trials = np.array(
            [
                len(trial)
                for trial in arrangement_data.groupby("trial_idx")["trial"]
                .unique()
                .sort_index()
            ]
        )
        self.insufficient_trial_rep = (
            True if not np.all(num_unique_trials == 2) else False
        )

        def assign_rep(group):
            unique_trials = group["trial"].unique()
            group["rep"] = (group["trial"] == unique_trials[0]).astype(int)
            return group

        arrangement_data = arrangement_data.groupby("trial_idx").apply(assign_rep)
        arrangement_data.reset_index(drop=True, inplace=True)
        self.arrangement_data = arrangement_data

    def exclusion_metadata(self, time_lowerbound):
        """exclude participants based on participation time and completion status."""
        exclusion_reason = ""
        exclusion = True
        tasks_finished = self.metadata.tasks_finished.item()
        started = datetime.fromisoformat(self.metadata.started.item())
        progressed = datetime.fromisoformat(self.metadata.progressed.item())
        dur_mins = (progressed - started).seconds / 60
        if self.metadata.status.item() != "finished" and (
            tasks_finished <= 9 or self.pos_data.empty
        ):
            exclusion_reason += "tasks not finished;"
        elif dur_mins < time_lowerbound:
            exclusion_reason += "low participation time;"
        else:
            exclusion = False
        return exclusion, exclusion_reason

    def exclusion_arrangement_data(self, p_threshold=0.05):
        """exclude participants based on within-subject reliability, practice check, and completion status."""
        exclusion_reason = ""
        stim2pos = None
        exclusion = True
        exclusion_code = None

        self.preprocess_position_data()

        mean_y_corr, median_y_corr, mean_x_corr, pval_y = self.within_subj_reliability()
        if mean_y_corr is None:
            exclusion_reason += "no repetition of any trial;"
            return (
                exclusion,
                exclusion_reason,
                stim2pos,
                mean_y_corr,
                pval_y,
            )
        practice_performance, practice_stim2pos = self.practice_check()
        if (
            practice_performance
            and not np.isnan(pval_y)
            and pval_y <= p_threshold
            and self.insufficient_trials_recorded is False
        ):
            exclusion = False

        if self.insufficient_trials_recorded:
            exclusion_reason += "missing trials;"
        if not practice_performance:
            exclusion_reason += "unreliable practice check performance;"
            exclusion_code = 1
            stim2pos = practice_stim2pos
        if np.isnan(pval_y):
            exclusion_reason += "no record of some trial, nan correlation;"
            exclusion_code = 3
        if pval_y > p_threshold:
            exclusion_reason += "unreliable within-subj reliability;"
            exclusion_code = 2
        exclusion_reason += f"corr: {mean_y_corr}, pval_y:{pval_y};"

        if self.insufficient_trial_rep:
            exclusion_reason += "wrong number of repetitions "
            if exclusion is False:  # we will not exclude this participant
                exclusion_reason += "(participant data still used)"

        return (
            exclusion,
            exclusion_reason,
            stim2pos,
            mean_y_corr,
            median_y_corr,
            mean_x_corr,
            pval_y,
            exclusion_code,
        )

    def get_subj_debrief(self):
        """get debriefing form responses."""
        debrief_index = self.task_names.index("debriefing_form")
        debrief_task = self.task_data[debrief_index]
        if debrief_task["status"] == "finished":
            engaging = debrief_task["engaging"]
            difficult = debrief_task["difficult"]
            easy_comments = debrief_task["easy_describe"]
            difficult_comments = debrief_task["difficult_describe"]
            comments = debrief_task["comments"]
            return engaging, difficult, easy_comments, difficult_comments, comments
        else:
            return None, None, None, None, None

    def check_completion(self, task):
        """Check if the task is completed."""
        return task["status"] == "finished"

    def practice_check(self):
        """Check if the participant understands the practice task."""
        practice_index = self.task_names.index("practice_check")
        practice_task = self.task_data[practice_index]
        assert self.check_completion(practice_task)

        self.practice_data = self.practice_data.sort_values(by="y", ascending=False)

        stim_names = self.practice_data.stim1_name.values
        stim_y = self.practice_data.y.values
        stim_dict = OrderedDict(zip(stim_names, stim_y))

        # very loose criterion
        if "dog_car" in stim_dict:
            if (
                stim_dict["dog_car"] > stim_dict["apple_orange"]
                and stim_dict["apple_orange"] > stim_dict["apple_apple2"]
                and stim_dict["apple_orange"] > stim_dict["strawberry_strawberry2"]
            ):
                return True, " > ".join(stim_names)
            else:
                return False, " > ".join(stim_names)
        else:
            if (
                stim_dict["strawberry_dog"] > stim_dict["apple_orange"]
                and stim_dict["apple_car"] > stim_dict["apple_apple2"]
                and stim_dict["apple_car"] > stim_dict["strawberry_strawberry2"]
                and stim_dict["apple_orange"] > stim_dict["apple_apple2"]
                and stim_dict["apple_orange"] > stim_dict["strawberry_strawberry2"]
            ):
                return True, " > ".join(stim_names)
            else:
                return False, " > ".join(stim_names)

    def within_subj_reliability(self):
        """Calculate within-subject reliability through correlation between two repetitions of each trial."""
        x_corrs, y_corrs = [], []
        for i_trial in self.arrangement_data.trial_idx.unique():
            trial_data = self.arrangement_data[
                self.arrangement_data.trial_idx == i_trial
            ]
            if trial_data.rep.nunique() != 2:
                print(self.subj_id, f"trial idx {i_trial} missing rep data")
                continue
            one_rep = trial_data[trial_data.rep == 0]
            one_rep = one_rep.sort_values(by="pair")
            two_rep = trial_data[trial_data.rep == 1]
            two_rep = two_rep.sort_values(by="pair")
            assert one_rep.pair.values.tolist() == two_rep.pair.values.tolist()
            if self.inference_method == "spearman_corr":
                y_corr = spearmanr(one_rep.y, two_rep.y).statistic
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x_corr = spearmanr(one_rep.x, two_rep.x).statistic
            elif self.inference_method == "pearson_corr":
                y_corr = np.corrcoef(one_rep.y, two_rep.y)[0, 1]
                x_corr = np.corrcoef(one_rep.x, two_rep.x)[0, 1]
            else:
                raise ValueError("unsupported correlation type")
            y_corrs.append(spearman_brown_correction(y_corr))
            x_corrs.append(spearman_brown_correction(x_corr))

        # test if the vertical correlation is significantly different from zero
        if len(y_corrs) == 0:
            return None, None, None, None, None
        y_corrs = np.array(y_corrs)
        x_corrs = np.array(x_corrs)
        median_y_corr = np.median(y_corrs)
        if np.any(np.isnan(y_corrs)):
            mean_y_corr = np.nanmean(y_corrs)
            pval_y = np.nan
        else:
            mean_y_corr = y_corrs.mean()
            # fisher transform correlation
            fisher_transformed_y_corrs = np.arctanh(
                np.clip(y_corrs, -1 + 1e-8, 1 - 1e-8)
            )
            # statistical one sample t test on correlation
            t_stat_y, pval_y = ttest_1samp(
                a=fisher_transformed_y_corrs, popmean=0, alternative="greater"
            )

            # no nan corr in y pos but in x pos
        mean_x_corr = np.nan if np.any(np.isnan(x_corrs)) else x_corrs.mean()

        return mean_y_corr, median_y_corr, mean_x_corr, pval_y


def assign_cond(metadata, position_data):
    """Assign condition based on the stimulus name."""
    for participant_id in metadata.name.tolist():
        participant_data = position_data[
            (position_data.participation == participant_id)
            & (position_data.task == "face_arrangement_trials")
        ]
        if len(participant_data) == 0:
            generator, cond, seed = np.nan, np.nan, np.nan
        else:
            stim_name = participant_data.iloc[0].stim1_name
            parsed = stim_name.split("_")
            idx = parsed.index("seed")
            if idx == 2:
                generator, cond, seed = parsed[0], parsed[1], parsed[3]
            elif idx == 3:
                generator, cond, seed = (
                    "-".join([parsed[0], parsed[1]]),
                    parsed[2],
                    parsed[4],
                )
        metadata.loc[metadata.name == participant_id, "generator"] = generator
        metadata.loc[metadata.name == participant_id, "condition"] = cond
        metadata.loc[metadata.name == participant_id, "seed"] = seed
    return metadata


def generate_subj_datastruct(data, subj_ids, col="y"):
    """Average across two trial repetitions for each subject.

    Return:
    sub_data (xarray.DataArray): (# participants x # trials x # face pairs)
    """

    data = data[data.subj_id.isin(subj_ids)]
    reps = sorted(data.rep.unique())
    trials = sorted(data.trial_idx.unique())
    pairs = sorted(data.pair.unique())

    raw_array = np.empty(
        [len(subj_ids), len(reps), len(trials), len(pairs)], dtype=float
    )
    raw_array[..., :] = np.nan
    dims = ("subj_id", "i_rep", "i_trial", "i_pair")
    coords = {
        "subj_id": subj_ids,
        "i_rep": reps,
        "i_trial": trials,
        "i_pair": pairs,
    }
    subj_data = xr.DataArray(raw_array, dims=dims, coords=coords)

    for subj_id in subj_ids:
        for i_rep in reps:
            for i_trial in trials:
                trial_data = data[
                    (data.subj_id == subj_id)
                    & (data.rep == i_rep)
                    & (data.trial_idx == i_trial)
                ]
                if len(trial_data) == 0:
                    continue
                trial_data = trial_data.sort_values(by="pair")
                pair = trial_data.pair.values
                subj_data.loc[subj_id, i_rep, i_trial, pair] = trial_data[col].values

    return subj_data


def normalize_subj_data(sub_data):
    """
    Args:
        sub_data (xarray.DataArray): (# participants x # trials x # face pairs)
    """
    return sub_data / xr.DataArray.std(sub_data, dim="i_pair")


def get_best_rank_spearman(sub_data):
    # rank each trial for each subject
    ranked_data = sub_data.rank(dim="i_pair")
    # get normalized ranks before averaging across subjects
    mean_normalized_rank_data = normalize_subj_data(ranked_data).mean(dim="subj_id")
    # rank the average
    ranked_mean_rank = mean_normalized_rank_data.rank(dim="i_pair")

    return ranked_mean_rank


def between_subj_reliability_leave1out(sub_data, inference_method):
    """Compute the lower and upper bound of the noise ceiling.

    Args:
        sub_data (xarray.DataArray): (# participants x # trials x # face pairs)
        inference_method (str): 'spearman_corr' or 'pearson_corr'
    Return:
        mean_comp (float): mean correlation between each participant's ratings and the average of other participants' ratings
        comp_df (pandas.core.frame.DataFrame)
    """

    noise_ceiling = []
    leave1out_df = []

    # get upper bound prediction
    if inference_method == "spearman_corr":
        all_subj = get_best_rank_spearman(sub_data)
    elif inference_method == "pearson_corr":
        all_subj = normalize_subj_data(sub_data).mean("subj_id")

    for subj_id in sub_data.subj_id:
        heldout_subj = sub_data.sel(subj_id=subj_id.values)
        other_subj = sub_data.drop_sel(subj_id=subj_id.values)

        if inference_method == "spearman_corr":
            heldout_subj = heldout_subj.rank(dim="i_pair")
            other_subj = get_best_rank_spearman(other_subj)
            lower = xr_spearmanrho_rae(heldout_subj, other_subj, dim="i_pair")
            upper = xr_spearmanrho_rae(heldout_subj, all_subj, dim="i_pair")

        elif inference_method == "pearson_corr":
            heldout_subj = normalize_subj_data(heldout_subj)
            other_subj = normalize_subj_data(other_subj).mean(dim="subj_id")
            lower = xr.corr(heldout_subj, other_subj, dim="i_pair")
            upper = xr.corr(heldout_subj, all_subj, dim="i_pair")

        # preserve trial information
        leave1out_df.append(
            pd.DataFrame(
                {
                    "subj_id": subj_id.values.item(),
                    "model": "leave1out",
                    "i_trial": lower.i_trial.values,
                    "cv_corr": lower.values,
                }
            )
        )
        noise_ceiling.append(
            pd.DataFrame(
                {
                    "subj_id": subj_id.values.item(),
                    "i_trial": lower.i_trial.values,
                    "lower_noise_ceiling": lower.values,
                    "upper_noise_ceiling": upper.values,
                }
            )
        )
    leave1out_df = pd.concat(leave1out_df)
    noise_ceiling_df = pd.concat(noise_ceiling)

    return noise_ceiling_df, leave1out_df


def between_subj_reliability_pairs(sub_data, inference_method):
    """Compute the lower bound of the noise ceiling.

    Args:
        sub_data (xarray.DataArray): (# participants x # trials x # face pairs)
    Return:
        mean_corr (float): mean correlation between each participant's ratings and the average of other participants' ratings
    """
    comps = []
    # normalized_sub_data = normalize_subj_data(sub_data)
    subj_pairs = itertools.combinations(sub_data.subj_id.values, 2)
    for subj_pair in subj_pairs:
        first_subj = sub_data.loc[subj_pair[0]]
        second_subj = sub_data.loc[subj_pair[1]]
        if inference_method == "pearson_corr":
            comps.append(
                xr.corr(first_subj, second_subj, dim="i_pair")
                .mean(dim="i_trial")
                .values.item()
            )
        elif inference_method == "spearman_corr":
            comps.append(
                xr_spearmanrho_rae(
                    first_subj.rank(dim="i_pair"),
                    second_subj.rank(dim="i_pair"),
                    dim="i_pair",
                )
                .mean(dim="i_trial")
                .values.item()
            )
    mean_corr = np.mean(comps)

    return mean_corr


def between_subj_reliability_pairs_xr(sub_data, inference_method):
    coords = {"subj1": sub_data.subj_id.values, "subj2": sub_data.subj_id.values}
    corr_xr = xr.DataArray(
        data=np.ones(
            shape=(len(sub_data.subj_id.values), len(sub_data.subj_id.values))
        ),
        dims=list(coords.keys()),
        coords=coords,
    )
    subj_pairs = itertools.combinations(sub_data.subj_id.values, 2)
    for subj_pair in subj_pairs:
        first_subj = sub_data.loc[subj_pair[0]]
        second_subj = sub_data.loc[subj_pair[1]]
        if inference_method == "pearson_corr":
            corr = (
                xr.corr(first_subj, second_subj, dim="i_pair")
                .mean(dim="i_trial")
                .values.item()
            )
        elif inference_method == "spearman_corr":
            corr = (
                xr_spearmanrho_rae(
                    first_subj.rank(dim="i_pair"),
                    second_subj.rank(dim="i_pair"),
                    dim="i_pair",
                )
                .mean(dim="i_trial")
                .values.item()
            )
        corr_xr.loc[subj_pair[0], subj_pair[1]] = corr
        corr_xr.loc[subj_pair[1], subj_pair[0]] = corr

    return corr_xr


def spearman_brown_correction(rho):
    if rho > 0:
        return 2 * rho / (1 + rho)
    else:
        return rho


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    # https://stackoverflow.com/a/312464
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def xr_pearsonr(x, y, dim, assume_no_nans=False):
    """a slightly faster alternative to xr.corr (if there are no nans)

    args:
        x: xarray.DataArray
        y: xarray.DataArray
        dim: str (across which dimension to compute the correlation)
        assume_no_nans: bool (whether to assume that there are no nans in the data)
    returns
        correlation: xarray.DataArray with dim reduced
    """
    # from https://gist.github.com/kathoef/2fbdfd19f29a03aed561e0f5f56d445a
    x, y = xr.align(x, y)

    # to speed up, we can assume that there are no missing values
    # otherwise, we eliminate pairs where at least one of the variables is missing
    if not assume_no_nans:
        valid_values = x.notnull() & y.notnull()
        x = x.where(valid_values)
        y = y.where(valid_values)

    c = ((x - x.mean(dim=dim)) * (y - y.mean(dim=dim))).mean(dim=dim)
    c /= x.std(dim)
    c /= y.std(dim)
    return c


def xr_spearmanrho_rae(x, y, dim, assume_no_nans=False):
    """xarray version of spearmanrho, with an analytical random-among-equals correction for ties

    args:
        x: xarray.DataArray
        y: xarray.DataArray
        dim: str (across which dimension to compute the correlation)
        assume_no_nans: bool (whether to assume that there are no nans in the data)
    returns
        correlation: xarray.DataArray with dim reduced
    """
    # from https://gist.github.com/kathoef/2fbdfd19f29a03aed561e0f5f56d445a
    x, y = xr.align(x, y)

    # # currently this function does not handle missing values
    if not assume_no_nans:
        valid_values = x.notnull() & y.notnull()
        x = x.where(valid_values)
        y = y.where(valid_values)

    n = (x.notnull() * y.notnull()).sum(dim=dim)
    mean_rank = (1 + n) / 2

    # compute ranked vectors with average ranks for ties:
    a = x.rank(dim=dim, pct=False) - mean_rank  # requires the bottleneck package
    b = y.rank(dim=dim, pct=False) - mean_rank  # requires the bottleneck package

    # Equation 14 in https://doi.org/10.48550/arXiv.2112.09200
    c = 12 / (n**3 - n) * (a * b).sum(dim=dim)
    return c
