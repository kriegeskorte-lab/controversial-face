import os, sys

# add project root path to pythonpath
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from pathlib import Path
import gc
import numpy as np

import torch
import torch.utils.tensorboard
import torch.utils.tensorboard.summary
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import pandas as pd

from model_zoo import model_zoo
import utils
from utils import omega_conf_to_hparams
from stimulus_optimization.loss import RDM_losses, latent_losses
from stimulus_optimization.generator_objects import BFMGenerator, StyleGAN3
import stimulus_optimization.faces_to_representations
from prepare_readouts import prepare_readouts


def prepare_readouts_and_models(cfg: DictConfig):
    """
    Prepare readouts (specifications of representationa) and models (actual NNs) for the experiment.

    Args:
        cfg: experiment configuration

    returns:
        readouts: list of readout specifications (dicts)
        models: a dict of models (torch.nn.Module)
        face_generator_devices: a list of torch.device-s used for rendering
        sampling_device: a torch.device used for sampling related operations
    """
    load_representational_models = cfg.load_representational_models

    if load_representational_models:
        readouts = prepare_readouts(
            instance_id=list(
                cfg.instance_id
            ),  # convert from OmegaConf list to python list
        )
    else:
        readouts = []

    # automatic GPU allocation
    readouts, face_generator_devices, sampling_device = utils.gpu_allocation(
        readouts,
        sampling_device=cfg.sampling_device if "sampling_device" in cfg else "cpu",
        n_gpus_for_rendering=(
            cfg.n_gpus_for_rendering if "n_gpus_for_rendering" in cfg else 1
        ),
    )

    if cfg.verbose > 0:
        print("readouts:", pd.DataFrame(readouts))
        print("face_generator_device:", face_generator_devices)
        print("sampling_device:", sampling_device)

    # let's load the NN models
    models = dict()
    for readout in readouts:
        cur_model_name = readout["model_name"]
        cur_model_instance_id = readout["instance_id"]
        models[cur_model_name, cur_model_instance_id] = getattr(
            model_zoo, cur_model_name
        )(instance_id=cur_model_instance_id)
        models[cur_model_name, cur_model_instance_id].load(
            device=readout["model_device"]
        )
        if cfg.verbose > 0:
            print(
                f"loaded model {cur_model_name}, realization {cur_model_instance_id} to device {models[cur_model_name, cur_model_instance_id].device}"
            )
        layer_selectors = cfg.layer_selector.split(",")
        selected_layers = np.array(
            [
                (layer_name, layer_num)
                for layer_name, layer_num in zip(
                    models[cur_model_name, cur_model_instance_id].layer_names,
                    models[cur_model_name, cur_model_instance_id].layer_nums,
                )
                for layer_selector in layer_selectors
                if layer_selector in layer_name
            ]
        )
        readout["layer_name_subset"] = selected_layers[:, 0].tolist()
        readout["layer_idx"] = selected_layers[:, 1].astype(int).tolist()
        if cfg.verbose >= 3:
            print(f"{cur_model_name} selected layers:")
            print(readout["layer_idx"])
            print(readout["layer_name_subset"])

    return readouts, models, face_generator_devices, sampling_device


def setup_faces_to_representations(cfg: DictConfig, representation_type: str):
    """setup an object that goes from BFM face object to representations
    args:
        cfg: experiment configuration
        representation_type: type of representation to use (str) RDMs|pairwise_dissimilarities|activations|latents
    returns
        faces_to_representations
    """
    if representation_type == "RDMs":
        faces_to_representations = (
            stimulus_optimization.faces_to_representations.ImagesToRdms(cfg)
        )
    elif representation_type == "pairwise_dissimilarities":
        # this object is a list of dissimilarity tensors (n_layers x n_pairs or just n_pairs)
        # each tensor matches to one readout.
        # to ease the use of the object, we convert it to a dictionary with model name as key as, and
        # a nested dictionary of instances as values.
        faces_to_representations = stimulus_optimization.faces_to_representations.ImagesToPairwiseDissimilarities(
            cfg
        )
    elif representation_type == "latents":
        faces_to_representations = (
            stimulus_optimization.faces_to_representations.cloneLatents(cfg)
        )
    else:
        raise NotImplementedError

    return faces_to_representations


def setup_representation_to_loss(cfg):
    """
    Prepare an object that transforms representations to loss values.

    Args:
        cfg: experiment configuration

    returns a represention_to_loss object
    """

    if hasattr(cfg, "model_prior_prob_file") and cfg.model_prior_prob_file is not None:
        model_prior_probs, layer_prior_probs = utils.read_prior_probs(
            cfg.model_prior_prob_file
        )
    else:
        model_prior_probs = None
        layer_prior_probs = None

    if cfg.loss == "multiinstance_multilayer":
        representation_to_loss = (
            RDM_losses.RDM_MultiInstance_Multilayer_Raw_Correlation_Utility(
                cfg,
                device=None,
                pairs_per_correlation=None,
                model_prior_probs=model_prior_probs,
                layer_prior_probs=layer_prior_probs,
            )
        )
    elif cfg.loss == "multiinstance_pairwise_dissimilarities":
        representation_to_loss = (
            RDM_losses.Pairwise_MultiInstance_Multilayer_Raw_Correlation_Utility(
                cfg, pairs_per_correlation=cfg.pairs_per_correlation
            )
        )
    elif cfg.loss is None:
        representation_to_loss = latent_losses.NullLoss()
    else:
        raise ValueError

    return representation_to_loss


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # start by cleaning the GPU memory
    gc.collect()  # collects unreachable circular references
    torch.cuda.empty_cache()  # empties the GPU memory

    if cfg.verbose:
        # print pytorch allocated GPU memory in each device (used to diagnose memory leaks)
        for i_gpu in range(torch.cuda.device_count()):
            print(
                f"GPU {i_gpu} memory: {torch.cuda.memory_allocated(torch.cuda.device(i_gpu)) / 1e6} MB"
            )

    try:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        export_folder = hydra_cfg["runtime"]["output_dir"]
        run_name = hydra_cfg.job.name
    except:  # hydra is not running
        export_folder = cfg.export_folder
        run_name = cfg.export_folder
    print("export_folder", export_folder)

    if cfg.verbose:
        print(f"created folder {export_folder}")

    if getattr(cfg, "generate_dissimilarity_scatter_plot", False):
        models_selected = [cfg.model1, cfg.model2]

    img_path = os.path.join(export_folder, "ims")
    vis_path = os.path.join(export_folder, "vis")
    latent_npz_dir = os.path.join(export_folder, "latents")
    rdm_path = os.path.join(export_folder, "rdms")
    log_path = os.path.join(export_folder, "logs")

    utils.write_down_git_hush(os.path.join(export_folder, "git_hush"))

    # when the the evaluation is stochastic (e.g., with jittered stimuli), the gradients can be noisy.
    (
        readouts,
        models,
        face_generator_devices,
        sampling_device,
    ) = prepare_readouts_and_models(cfg)
    if cfg.loss is not None:
        reference_readouts = [
            r for r in readouts if r["instance_id"] == cfg.reference_realization_id
        ]
        data_generating_readouts = [
            r for r in readouts if r["instance_id"] != cfg.reference_realization_id
        ]

    Path(img_path).mkdir(parents=True, exist_ok=True)
    Path(latent_npz_dir).mkdir(parents=True, exist_ok=True)
    Path(vis_path).mkdir(parents=True, exist_ok=True)
    if getattr(cfg, "print_rdm_plots", False):
        Path(rdm_path).mkdir(parents=True, exist_ok=True)
    if getattr(cfg, "save_optimization_log", False):
        Path(log_path).mkdir(parents=True, exist_ok=True)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_path)

    # setup for animated GIFs

    if cfg.random_seed is not None:
        torch.manual_seed(cfg.random_seed)

    # Initialize the face generator
    if cfg.verbose > 0:
        print("initializing the face generator and a set of random faces...")

    if cfg.generator == "BFM":
        face_generator = BFMGenerator(cfg, devices=face_generator_devices)
        face_generator.set_up_generator()
    elif cfg.generator == "StyleGAN3":
        face_generator = StyleGAN3(cfg, devices=face_generator_devices)
        face_generator.set_up_generator()
    else:
        raise NotImplementedError

    if getattr(cfg, "load_latents", False):
        if cfg.verbose > 0:
            print("loading latents from the last optimization...")
        load_latent_types = getattr(
            cfg, "load_latent_types", None
        )  # if None, load all latents in npz.
        assert os.path.exists(cfg.latent_path), (
            "latent path not found:" + cfg.latent_path
        )
        face_generator.load_latents(cfg.latent_path, latent_types=load_latent_types)
        print("loaded latents from: " + cfg.latent_path)

    if getattr(cfg, "bound_latents", False) or (
        any(
            getattr(cfg, f"optimize_only_n_{latent_type}", None) is not None
            for latent_type in cfg.coefs
        )
    ):
        if cfg.verbose > 0:
            print("initializing latent constraints...")
        face_generator.initialize_constraints()

    latent_path = os.path.join(latent_npz_dir, "original_faces.npz")
    face_generator.save_latents(latent_path)

    if cfg.verbose > 0:
        print("setting optimization parameters...")
    face_generator.set_up_optimization_params()

    # the general structure of the objects used in the optimization loop:

    # faces_to_representations (faces -> representations) with the following properties:
    #   .domain: (how the input faces are represented)
    #       'images' (image computable representations), OR
    #       'latents' (representations linked directly to the BFM latents)
    #   .codomain: (the format of the representations)
    #      'RDMs' representation dissimilarity matrices, OR
    #      'activations' (NN activations, not currently implemented),
    #      'latents' (just copy the latents)

    # representation_to_loss (representations -> loss) with the following property:
    #   .domain: (a value consistent with the range of the faces_to_representations object)

    # setup loss/accuracy function
    # all of this functions return loss, accuracy (or loss, None)
    # TODO: can we make this if-else more abstract? might require introducing optional argparse argument groups, or replace argparse with a more flexible interface.

    if cfg.max_iters > 0:
        if cfg.verbose > 0:
            print("optimizing...")
        # setting up optimizer
        optimizer_type = getattr(cfg, "optimizer", "Adam")
        lr_adaptive = getattr(cfg, "lr_adaptive", True)
        lr_schedule = getattr(cfg, "lr_schedule", "StepLR")
        conv_check_param = getattr(cfg, "conv_check_param", "loss")
        early_stopping = getattr(cfg, "early_stopping", True)

        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                face_generator.optimized_params.values(),
                lr=cfg.learning_rate,
                eps=1e-14,
                betas=(cfg.beta1, cfg.beta2),
            )
        elif optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                face_generator.optimized_params.values(), lr=cfg.learning_rate
            )
        # setting up convergence check
        if lr_adaptive:
            conv_check = utils.ConvergenceCheck(
                window_length=getattr(cfg, "window_length", 30)
            )
        # setting up learning rate scheduler
        if lr_schedule == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.5
            )
            scheduler_steps = 0
        elif lr_schedule == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=0, last_epoch=-1
            )
            num_iters = np.cumsum(cfg.T_0 * (cfg.T_mult ** np.arange(cfg.num_restarts)))
            cfg.max_iters = int(num_iters[-1]) - 1  # reset max_iters
            print("max iterations:", cfg.max_iters)
            assert lr_adaptive is False

    representation_to_loss = setup_representation_to_loss(cfg)
    # setup image to representation mapping
    representation_type = representation_to_loss.domain
    faces_to_representations = setup_faces_to_representations(cfg, representation_type)
    # ensure domain and codomain are consistent
    assert faces_to_representations.codomain == representation_to_loss.domain

    # add nonface probability loss for stylegan3
    if getattr(cfg, "nonface_loss", False):
        from stimulus_optimization.loss import nonface_losses

        alpha = getattr(cfg, "face_detector_smooth_alpha", 20)
        nonface_probability_loss = nonface_losses.NonFaceProbabilityLoss(
            detector_weight_path=cfg.face_detector_weight_path,
            classifier_weight_path=cfg.classifier_path,
            alpha=alpha,
            model_device=cfg.face_detector_device,
            agg_func=getattr(cfg, "nonface_loss_agg_func", "mean"),
        )

    with torch.no_grad():  # save initial images
        face_generator.save_high_quality_images(
            im_path=img_path, im_set_fname="original", rgba=False
        )

    if cfg.loss:
        disable_reference_im_jitter = getattr(cfg, "disable_reference_im_jitter", False)
        for t in tqdm(range(cfg.max_iters + 1)):
            if cfg.max_iters > 0:
                optimizer.zero_grad()
            accum_loss = 0.0
            accum_accuracy = 0.0
            face_generator.step_optimization()
            representation_to_loss.current_iter = t

            face_im_tensors = face_generator.generate_images(
                imsize=cfg.optimization_imsize, im_dir=None, rgba=False
            )

            for i_accum_step in range(cfg.accumulation_steps):
                data_generating_im_tensors = face_generator.jitter_and_crop(
                    face_im_tensors,
                    crop_size=cfg.optimization_crop_size,
                    enable_noise=cfg.enable_jitter,
                )
                if disable_reference_im_jitter:
                    if i_accum_step == 0:
                        reference_im_tensors = face_generator.jitter_and_crop(
                            face_im_tensors,
                            crop_size=cfg.optimization_crop_size,
                            enable_noise=False,
                        )
                else:  # use the same jittering for reference and data generating images
                    reference_im_tensors = data_generating_im_tensors.clone()

                if not disable_reference_im_jitter or (
                    i_accum_step == 0 and disable_reference_im_jitter
                ):
                    reference_representations = faces_to_representations(
                        reference_im_tensors, reference_readouts, models
                    )
                representations = faces_to_representations(
                    data_generating_im_tensors, data_generating_readouts, models
                )
                for model in representations.keys():
                    representations[model][cfg.reference_realization_id] = (
                        reference_representations[model][cfg.reference_realization_id]
                    )

                loss, accuracy = representation_to_loss(representations)
                if getattr(cfg, "nonface_loss", False):
                    nonface_probability = nonface_probability_loss(
                        reference_im_tensors.to(nonface_probability_loss.model_device)
                    )
                    loss = loss + nonface_probability

                del data_generating_im_tensors, representations
                loss = loss / cfg.accumulation_steps
                if not loss.isfinite().item():
                    print("non-finite loss")
                    continue
                loss.backward(retain_graph=True)

                accum_loss += float(loss)
                if accuracy is not None:
                    accum_accuracy += float(accuracy)
                else:
                    accum_accuracy = None
                del loss

            loss = accum_loss
            del face_im_tensors, reference_im_tensors

            if accum_accuracy is not None:
                accum_accuracy = accum_accuracy / cfg.accumulation_steps

            if cfg.save_optimization_log:
                writer.add_scalar("Loss/train", accum_loss, t)
                if cfg.generator == "StyleGAN3":
                    writer.add_scalar(
                        "Loss/nonface_probability", nonface_probability, t
                    )
                    writer.add_scalar(
                        "Controversiality", -(accum_loss - nonface_probability), t
                    )
                writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], t)
                if accum_accuracy is not None:
                    writer.add_scalar("Accuracy/train", accum_accuracy, t)

            if t % 1 == 0:
                loss_str = f"{t} loss:{accum_loss:.5f}"
                if accum_accuracy is not None:
                    loss_str += f" accuracy:{accum_accuracy:.3f}"
                if cfg.generator == "StyleGAN3":
                    loss_str += f" neg log face prob:{nonface_probability:.4f}"

                if cfg.verbose >= 2:
                    print(loss_str)
                else:
                    print(f"{t} loss:{accum_loss:.3f}")

            if (
                t >= cfg.max_iters
            ):  # don't optimize on the last step, since the loss will not be measured again.
                print("max iters reached")
                break

            if lr_adaptive:
                if conv_check_param == "loss":
                    did_converge = conv_check.update_and_check_convergence(accum_loss)
                elif conv_check_param == "accuracy":
                    did_converge = conv_check.update_and_check_convergence(
                        -accum_accuracy
                    )

                if did_converge:
                    if scheduler_steps < cfg.max_scheduler_steps:
                        scheduler.step()
                        scheduler_steps += 1
                        print("reducing learning rate")
                        conv_check = utils.ConvergenceCheck(
                            window_length=getattr(cfg, "window_length", 30)
                        )
                    elif early_stopping:
                        print("converged.")
                        break
                    elif not early_stopping:
                        # for certain objective, we want the optimization to continue until the maximum iteration is reached.
                        print("converged. continuing optimization...")
            else:
                scheduler.step()

            optimizer.step()

        del (
            loss,
            accuracy,
            readouts,
            models,
            reference_representations,
            reference_readouts,
            data_generating_readouts,
        )
        torch.cuda.empty_cache()

    with torch.no_grad():
        test_accuracy = None
        if getattr(cfg, "test_realization_id", None) is not None:
            print(
                "simulate model recovery accuracy on a heldout model reference instance with jittered stimuli..."
            )
            cfg.instance_id = [cfg.reference_realization_id, cfg.test_realization_id]
            (
                readouts,
                models,
                face_generator_devices,
                sampling_device,
            ) = prepare_readouts_and_models(cfg)
            reference_readouts = [
                readout
                for readout in readouts
                if readout["instance_id"] == cfg.reference_realization_id
            ]
            data_generating_readouts = [
                readout
                for readout in readouts
                if readout["instance_id"] == cfg.test_realization_id
            ]

            face_im_tensors = face_generator.generate_images(
                imsize=cfg.optimization_imsize, im_dir=None, rgba=False
            )
            reference_im_tensors = face_generator.jitter_and_crop(
                face_im_tensors,
                crop_size=cfg.optimization_crop_size,
                enable_noise=False,
            )
            reference_representations = faces_to_representations(
                reference_im_tensors, reference_readouts, models
            )
            test_accuracy = []
            for _ in tqdm(
                range(1000)
            ):  # jitter 1000 times and compute the average accuracy
                data_generating_im_tensors = face_generator.jitter_and_crop(
                    face_im_tensors,
                    crop_size=cfg.optimization_crop_size,
                    enable_noise=True,
                )
                representations = faces_to_representations(
                    data_generating_im_tensors, data_generating_readouts, models
                )
                for model in representations.keys():
                    representations[model][cfg.reference_realization_id] = (
                        reference_representations[model][cfg.reference_realization_id]
                    )
                loss, accuracy = representation_to_loss(representations)
                test_accuracy.append(accuracy)
            test_accuracy = np.array(test_accuracy)
            print("test accuracy:", test_accuracy.mean(), test_accuracy.std())
            np.save(
                os.path.join(export_folder, "logs", "test_accuracy.npy"), test_accuracy
            )

    if cfg.save_optimization_log:
        # save hparams and metrics to enable hyperparameter comparison between configurations
        metrics = {}
        metrics["final_loss"] = accum_loss
        if accum_accuracy is not None:
            metrics["final_accuracy"] = accum_accuracy
        if test_accuracy is not None:
            metrics["test_accuracy_mean"] = test_accuracy.mean()
            metrics["test_accuracy_std"] = test_accuracy.std()
        writer.add_hparams(hparam_dict=omega_conf_to_hparams(cfg), metric_dict=metrics)
        writer.close()

    print("saving post-optimization faces...")
    with torch.no_grad():
        # render and save resulting images (cropped faces)
        face_generator.save_high_quality_images(
            im_path=img_path, im_set_fname="optimized", rgba=False
        )
        latent_path = os.path.join(latent_npz_dir, "optimized.npz")
        face_generator.save_latents(latent_path)  # this saves the non-cropped images


if __name__ == "__main__":
    main()
