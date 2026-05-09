import os
import pandas as pd
import xarray as xr

analysis_dir = "final_exp/analysis"
version = 8

within_subj = pd.read_csv(os.path.join(analysis_dir, "within_subj_reliability.csv"))
excluded_subj = pd.read_csv(os.path.join(analysis_dir, "excluded_subj.csv"))
included_subj = within_subj[~within_subj.subj_id.isin(excluded_subj.participant_id)]
included_subj["seed"] = included_subj["seed"].astype(int)
included_subj = (
    included_subj.groupby(["generator", "condition", "seed"]).size().reset_index()
)
included_subj = included_subj.rename(columns={0: "num_participants"})
included_subj["num_extra"] = included_subj.apply(
    lambda x: 12 - x.num_participants, axis=1
)
condition_size = included_subj.groupby(["generator", "condition"]).num_extra.sum()
print(condition_size)
included_subj.to_csv(
    os.path.join(
        analysis_dir,
        "extra_participants",
        f"extra_participants_to_collect_v{version}.csv",
    )
)
