import os, sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import re
import argparse
import json
import numpy as np
import pandas as pd
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, default="final_exp")
args = parser.parse_args()

participants = []
demographic_info = []
demographic_header = [
    "age",
    "Gender",
    "Self-identified gender",
    "ethnicity",
    "Self-identified ethnicity",
    "Race (check all that apply)",
    "Self-identified race",
    "Highest Level of Education Completed",
]
# get all subdirectories of a directory using os random walk
valid_participants_path = os.path.join(
    args.experiment,
    "analysis",
    f"within_subj_reliability.csv",
)
valid_participants = pd.read_csv(valid_participants_path).subj_id.tolist()
excluded_subj = pd.read_csv(
    os.path.join(args.experiment, "analysis", "excluded_subj.csv")
).participant_id.tolist()
valid_participants = [p for p in valid_participants if p not in excluded_subj]
print(len(valid_participants))

experiment_map = {
    "s": "stylegan3",
    "p": "bfm_pose",
    "b": "bfm",
}
for root, dirnames, filenames in os.walk(args.experiment):
    for filename in filenames:
        if not filename.startswith("."):
            if "tree" in filename:
                data_path = open(os.path.join(root, filename), "r")

                metadata_filename = filename.replace("tree", "metadata").replace(
                    ".json", ".csv"
                )
                annotation_filename = filename.replace("tree", "annotations").replace(
                    ".json", ".csv"
                )

                data = json.load(data_path)
                metadata = pd.read_csv(os.path.join(root, metadata_filename))
                annotation = pd.read_csv(os.path.join(root, annotation_filename))

                experiment = filename.split("_")[2]
                experiment = experiment_map[experiment]

                for participant_id in data.keys():
                    if participant_id in valid_participants:
                        tasks = data[participant_id]["tasks"]
                        task_names = [task["task"]["name"] for task in tasks]

                        participant_info = {}
                        try:
                            demographics_index = task_names.index("demographic_form")
                        except Exception as e:
                            print(
                                "no demographics found for participant", participant_id
                            )
                            continue
                        demographics = tasks[demographics_index]
                        participant_info["subj_id"] = participant_id
                        participant_info["prolific_id"] = metadata[
                            metadata.name == participant_id
                        ].PROLIFIC_PID.item()
                        participant_info["experiment"] = experiment
                        one_stim = (
                            annotation[annotation.participation == participant_id]
                            .iloc[-1]
                            .stim1_name
                        )
                        seed = int(re.findall(r"seed_(\d+)", one_stim)[0])
                        participant_info["condition"] = one_stim.split(
                            f"{experiment}_"
                        )[1].split("_")[0]
                        participant_info["seed"] = seed

                        try:
                            for header in demographic_header:
                                if "Race" in header:
                                    if len(demographics[header]) > 1:
                                        races = ""
                                        for race in demographics[header]:
                                            races += race + ";"
                                        participant_info["Race"] = [races]
                                    elif len(demographics[header]) == 0:
                                        participant_info["Race"] = [""]
                                    else:
                                        participant_info["Race"] = demographics[header]
                                else:
                                    participant_info[header] = demographics[header]
                        except Exception as e:
                            print(
                                "no demographics found for participant", participant_id
                            )
                            continue
                        demographic_info.append(pd.DataFrame(participant_info))
                data_path.close()
demographic_info = pd.concat(demographic_info, join="inner")
demographic_info.to_csv(
    os.path.join(args.experiment, f"{args.experiment}_demographics.csv")
)
# print(np.array(demographic_info.age).astype(int).dtype)
print(Counter(demographic_info["Gender"]))
print(Counter(demographic_info["ethnicity"]))
# print(Counter(demographic_info['Race']))
demographic_info = demographic_info[
    pd.to_numeric(demographic_info.age, errors="coerce").notnull()
]
pd.set_option("display.max_columns", None)
print(
    f"Age: mean:{pd.to_numeric(demographic_info.age).mean()}, std:{pd.to_numeric(demographic_info.age).std()}"
)
# for h in demographic_header[1:]:
