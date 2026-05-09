import os
import torch
import numpy as np
from utils import mean_variation, minimum_abs_distance
from ..rdm_utils import rdm_euclidean


class LatentVariation(torch.nn.Module):
    def __init__(self):
        super(LatentVariation, self).__init__()
        self.domain = "latents"

    def forward(self, latent_dict):
        variation = [mean_variation(latent) for latent in latent_dict.values()]
        loss = -torch.sum(torch.stack(variation))
        return loss, None


class NullLoss(torch.nn.Module):
    """always return 0"""

    def __init__(self):
        super(NullLoss, self).__init__()
        self.domain = "latents"

    def forward(self, latent_dict):
        return torch.tensor(0.0), None


class RDMMinAbsDistLoss(torch.nn.Module):
    """maximize the minimum absolute distanceb between RDM dissimilarities"""

    def __init__(self, alpha=1e-2):
        super(RDMMinAbsDistLoss, self).__init__()
        self.domain = "latents"
        self.alpha = alpha

    def forward(self, latent_dict):
        latents = [latent for latent in latent_dict.values() if latent is not None]
        latents = torch.concat(latents, dim=1)
        rdm = rdm_euclidean(latents).unsqueeze(1)
        hard_minimum, smooth_minimum = minimum_abs_distance(rdm, alpha=self.alpha)
        loss = -smooth_minimum
        return loss, None


class DistRMSELoss(torch.nn.Module):
    def __init__(
        self,
        cfg,
        pairs_per_correlation=None,  # if None, compute RDMs
    ):
        super(DistRMSELoss, self).__init__()
        self.domain = "latents"

        optimized_latent_types = cfg.coefs
        self.ball_radius = {
            latent_type: cfg.get(f"{latent_type}_ball_radius", None)
            for latent_type in optimized_latent_types
        }
        self.box_SDs = {
            latent_type: cfg.get(f"{latent_type}_box_SDs", None)
            for latent_type in optimized_latent_types
        }
        self.pairs_per_correlation = pairs_per_correlation
        self.distance_type = cfg.get("distance_type", "euclidean")

    def calc_rmse(self, latent, max_distance):
        """calculate the RMSE between the target dissimilarities and observed dissimilarities"""
        num_faces = latent.shape[0]

        if self.pairs_per_correlation is not None:  # pairwise dissimilarities
            first_face_idx = torch.arange(0, num_faces, 2)
            second_face_idx = first_face_idx + 1
            n_pairs = first_face_idx.shape[0]

            dissimilarities = torch.linalg.norm(
                latent[first_face_idx] - latent[second_face_idx], dim=1
            )

            # span a linear space of euclidean distances within each trial
            div_round_up = lambda x, y: x // y + (x % y != 0)
            n_trials = div_round_up(n_pairs, self.pairs_per_correlation)
            target_dissimilarities = torch.zeros(
                n_pairs, dtype=torch.float32, device=latent.device
            )

            for i_trial in range(n_trials):
                cur_trial_pair_indices = torch.arange(
                    i_trial * self.pairs_per_correlation,
                    min((i_trial + 1) * self.pairs_per_correlation, n_pairs),
                    device=latent.device,
                )
                target_dissimilarities[cur_trial_pair_indices] = torch.linspace(
                    0,
                    max_distance,
                    len(cur_trial_pair_indices),
                    device=latent.device,
                )
        else:
            n_pairs = int((num_faces * (num_faces - 1)) // 2)

            dissimilarities = rdm_euclidean(
                latent
            )  # the dissimilarities are always measured in squared euclidean distances
            assert dissimilarities.shape[0] == n_pairs
            dissimilarities = torch.sort(dissimilarities)[0]

            start_dissimilarity = max_distance / (n_pairs + 1)
            if self.distance_type == "squared_euclidean":
                target_dissimilarities = torch.linspace(
                    start_dissimilarity**2,
                    max_distance**2,
                    n_pairs,
                    device=latent.device,
                )  # this defines the target dissimilarities in squared euclidean distances
            elif self.distance_type == "euclidean":  #
                target_dissimilarities = torch.linspace(
                    start_dissimilarity, max_distance, n_pairs, device=latent.device
                )  # this defines the target dissimilarities in (non-squared) euclidean distances
                target_dissimilarities = (
                    target_dissimilarities**2
                )  # bring the target dissimilarities to the squared euclidean space
            else:
                raise ValueError("invalid distance_type " + self.distance_type)

        rmse = torch.mean((target_dissimilarities - dissimilarities) ** 2)

        return rmse

    def forward(self, latent_dict):
        """

        args:
            latents: dict of torch.Tensors (batch_size, latent_size)
        """
        loss = 0.0
        for latent_name, latent in latent_dict.items():
            # requires one constraint is used to have a maximum RDM / pairwise distance
            assert (self.ball_radius[latent_name] is None) or (
                self.box_SDs[latent_name] is None
            ), "only one kind of constraint can be used"

            num_latents = latent.shape[1]
            # calculate maximum euclidean distance given the constraint
            # the maximal distance within a ball of radius ball_radius
            if self.ball_radius[latent_name] is not None:
                ball_radius = self.ball_radius[latent_name]
                max_distance = (num_latents * ball_radius) ** 0.5 * 2

            # the maximal distance with a hypercube of size 2*SDs
            elif self.box_SDs[latent_name] is not None:
                box_SDs = self.box_SDs[latent_name]
                max_distance = (num_latents**0.5) * 2 * box_SDs

            loss += self.calc_rmse(latent, max_distance)

        return loss, None


class DistRMSELoss_StyleGAN3(DistRMSELoss):
    def __init__(
        self,
        cfg,
        pairs_per_correlation=None,  # if None, compute RDMs
    ):
        super().__init__(cfg, pairs_per_correlation)
        self.domain = "latents"

        # calculate maximum euclidean distance given the reference set
        self.max_distance = {}
        max_dist_path = cfg.get("max_dist_path", None)
        if max_dist_path is None:
            reference_latent_path = os.path.join(
                cfg.reference_set_path, "latents", "optimized.npz"
            )
            for coef in cfg.coefs:
                assert (
                    coef in np.load(reference_latent_path).keys()
                ), "the reference set does not contain the optimized latents"
                reference_latents = np.load(reference_latent_path)[
                    coef
                ]  # (num_faces, num_latents)
                face_pair_dists = np.sqrt(
                    np.sum(
                        (reference_latents[::2] - reference_latents[1::2]) ** 2, axis=1
                    )
                )
                self.max_distance[coef] = np.max(face_pair_dists)
        else:
            max_distance = np.load(max_dist_path)
            for coef in cfg.coefs:
                self.max_distance[coef] = max_distance[coef].item()

    def forward(self, latent_dict):
        loss = 0.0
        for latent_name, latent in latent_dict.items():
            loss += self.calc_rmse(latent, self.max_distance[latent_name])
        return loss, None
