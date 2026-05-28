# Human Face Perception Reflects Inverse-Generative and Naturalistic Discriminative Objectives

This repository contains code for generating face stimuli, collecting model dissimilarities, preprocessing behavioral data, computing model-vs-human performance, and producing the project figures.

## Quick start

If you only want to reproduce the main analysis outputs and plots:

1. Create the conda environment from `environment.yml`.
2. Download `exp_data` from OSF and place it at `./exp_data`.
3. Run behavioral preprocessing and model-performance scripts in `data_analysis/`.
4. Run plotting scripts in `vis/`.

---

## Repository overview

- `stimulus_optimization/`: stimulus generation/optimization pipelines (BFM, BFM-pose, StyleGAN3), loss functions, and representation extraction.
- `differentiable_faces/`: differentiable implementation of BFM face rendering.
- `model_zoo/`: model wrappers and loading utilities.
- `data_analysis/`: subject data preprocessing, reliability/noise-ceiling analysis, model-performance computation, and statistical analysis (in R).
- `vis/`: plotting scripts (MDS and model-performance figures).
- `stylegan3/`, `yolov5_face/`: vendored model code used by optimization/evaluation.
  - StyleGAN3 code was obtained from: https://github.com/nvlabs/stylegan3
  - yolov5-face code was obtained from: https://github.com/deepcam-cn/yolov5-face
- `bfm-pose_init/`: initialization assets for BFM-pose optimization.
---

## Set up environment

The original runs were executed with Conda + CUDA modules. You can install the environment with:

```bash
conda env create -f environment.yml
```

Typical install time is 5--10 minutes on a desktop computer. You may need to adapt this setup for your local machine or HPC cluster.
`environment-axon-full.yml` contains the exact build strings from our GPU cluster installation.

```bash
conda activate controversial-face
# example HPC modules used in original runs
ml gcc/10.4
ml cudnn/8.6.0.163
ml cuda/11.8.0
```
GCC > 7 is required for StyleGAN3. The recommended GCC version depends on your CUDA version.

## Data and model setup

### Download behavioral/stimulus data

Download `exp_data` from OSF and place it at the repository root:

- OSF: https://osf.io/bzx4e
- Expected path after download: `./exp_data/...`

Many scripts assume this folder layout (for example `exp_data/raw_data`, `exp_data/stimuli_info`, and `exp_data/analysis`).

## Randomly sample or optimize faces

### Download model checkpoints

For model-driven controversial-stimulus optimization, download model checkpoints from Hugging Face:
https://huggingface.co/wenx-guo/controversial-face-model-checkpoints
Then place the downloaded checkpoint files in `./model_checkpoints/`.

Example setup:

```bash
pip install -U "huggingface_hub"

# to download faster, get your personal access token.
export HF_TOKEN="your_hf_token_here"

# optionally persist in your shell profile
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
source ~/.bashrc

# check if you are logged in
hf auth login
hf auth whoami 

# then download the required files from the HF repository into model_checkpoints/
mkdir -p model_checkpoints
hf download wenx-guo/controversial-face-model-checkpoints --local-dir model_checkpoints/
```

### Download BFM model

(Optional) If you want to sample or optimize for BFM faces, download BFM 2019 and place:

- `model2019_fullHead.h5` under `./model_checkpoints/`
- BFM site: https://faces.dmi.unibas.ch/bfm/bfm2019.html

### StyleGAN3 optimization constraints

(Optional) If you want to sample or optimize StyleGAN3 stimuli, download:

- If you already completed **Download model checkpoints**, you should already have `stylegan3-r-ffhq_128.pkl` under `./model_checkpoints/`. You can randomly sample StyleGAN3 faces using this model.
- If you want to optimize controversial StyleGAN3 stimuli, download the `yolov5n-face` checkpoint for face/non-face constraints: https://drive.google.com/file/d/1XJ8w55Y9Po7Y5WP4X1Kg1a77ok2tL_KY/view?usp=sharing. The upstream YOLOv5-face repository is: https://github.com/deepcam-cn/yolov5-face.
- The face/non-face classifier `face_nonface_classifier.npz` is also used in this project (it is downloaded automatically when you clone this repository).

Place these models under `./model_checkpoints/`.

## Stimulus optimization

- `stimulus_optimization/stim_optim.py`

This script uses Hydra configs from `stimulus_optimization/conf/`.

Stimulus optimization is GPU-memory intensive. Some GPUs are assigned to face rendering/generation, and the remaining GPUs are used to compute model activations. You may need to adjust the allocation based on available memory.

For the experimental scale used in this project (144 faces = 72 face pairs), the following allocations worked without OOM:

### BFM/BFM-pose recommended GPU allocation

| GPU type | Total GPUs requested | GPUs for rendering (`n_gpus_for_rendering`) | GPUs for model activations (remaining) |
|---|---:|---:|---:|
| GeForce RTX 2080 Ti | 5 | 2 | 3 |
| L40 / A40 | 2 | 1 | 1 |

### StyleGAN3 recommended GPU allocation

| GPU type | Total GPUs requested | GPUs for rendering (`n_gpus_for_rendering`) | GPUs for model activations (remaining) |
|---|---:|---:|---:|
| L40 / A40 | 5 | 4 | 1 |

Notes:
- Request the **total** number of GPUs from your scheduler.
- In the config, `n_gpus_for_rendering` sets how many GPUs are used for rendering, and the script automatically uses the rest for model activations.
- RTX 2080 Ti cards are generally too small for StyleGAN3 at this scale.

Example:
Request 5 GeForce RTX 2080 Ti GPUs, then run this script to generate controversial BFM faces:
```bash
ml gcc/10.4
ml cudnn/8.6.0.163
ml cuda/11.8.0

stimulus_class=bfm # choices: bfm, bfm-pose, stylegan3
cond=controversial # choices: controversial or random
seed=1

python -u stimulus_optimization/stim_optim.py \
  --config-name "${stimulus_class}_${cond}.yaml" \
  random_seed=$seed \
  export_folder="$HOME/${stimulus_class}_${cond}/seed_${seed}" \
  n_gpus_for_rendering=2
```

Notes:

- Optimization is GPU-intensive.
- The script writes outputs such as images, latents, RDMs, and logs to the export folder.

---

## Collect model dissimilarities on stimulus sets

Use:

```bash
python -u stimulus_optimization/collect_model_dissimilarities.py \
  --stimulus_set_parent_folder exp_data/stimuli_info
```

This computes and stores model dissimilarities for the provided stimuli. For BFM-based stimuli, the script can re-render faces before computing dissimilarities (see comments/flags in the script).

---

## Behavioral data preprocessing

Use:

```bash
python data_analysis/preprocess_subj_data_by_seeds.py \
  --experiment_dir exp_data \
  --inference_method spearman_corr
```

High-level workflow:

- Loads raw Meadows task outputs.
- Applies participant exclusion criteria.
- Computes participant-level dissimilarity structures.
- Saves analysis-ready arrays/tables under `exp_data/analysis/`.

Notes:

- Warnings about constant participant inputs can occur if a participant placed many/all pairs at the same location.
- Raw data versions are organized by stimulus family (`b`, `p`, `s`) and version IDs in `exp_data/raw_data`.

---

## Compute model performance and reliability metrics

Use:

```bash
python data_analysis/compute_model_performance_by_seeds.py \
  --experiment_dir exp_data \
  --inference_method spearman_corr
```

Outputs include:

- cross-validated model performance (`model_performance_*.nc`)
- between-subject reliability tables
- leave-one-subject-out and noise-ceiling summaries

Saved under `exp_data/analysis/`.

---

## Statistical analysis in R

From an R console:

```r
setwd("/path/to/controversial-face-proj-build")
source("data_analysis/stats_in_R/analysis_cond.R")
source("data_analysis/stats_in_R/analysis_pooled.R")
```

---

## Plotting

Generate figures with:

```bash
python vis/plot_mds.py
python vis/plot_model_performance_cond.py
python vis/plot_model_performance_pooled.py
```

`vis/plot_mds.py` expects the font file `HelveticaNeue.ttc` (included at the repository root).

---

## Reproducibility tips

- Keep folder names/paths consistent with script defaults (`exp_data`, `model_checkpoints`).
- Run behavioral preprocessing before model-performance evaluation.
- For large runs, log stdout/stderr and keep per-seed export folders.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
