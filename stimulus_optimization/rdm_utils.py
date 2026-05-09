"""First-level Representational Dissimilarity Matrices (RDM), RDM plot, and RDM comparison."""

import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import scipy.special, scipy.cluster, scipy.stats
from torch import nn

from rsatoolbox.vis.colors import rdm_colormap_classic


def rdm_euclidean(activations: torch.Tensor, normalize=False, method="fast"):
    """Generate RDMs using squared euclidean distances.

    Args:
        activations (torch.Tensor):
            (N, ...) model activations (model response) to N conditions, or stimulus images.
        normalize (bool)
            If True, divide RDMs by the number of features.

    Returns:
        rdm (torch.Tensor): flattened upper RDM
    """

    # if bound == 'tanh':
    #     activations = torch.tanh(activations)
    # elif bound == 'sigmoid':
    #     activations = torch.sigmoid(activations)

    if activations.ndim != 2:
        activations = activations.flatten(1, -1)
    n, p = activations.shape

    if method == "fast":
        b = torch.sum(activations**2, axis=1)
    else:
        b = torch.einsum(
            "ij,ij->i", activations, activations
        )  # this might use less memory than the line above , but it's slower.
    rdm = (
        -2 * torch.matmul(activations, activations.T) + b.unsqueeze(1) + b.unsqueeze(0)
    )
    del b

    triu_ind = torch.triu_indices(n, n, offset=1)
    triu_ind = (triu_ind[0], triu_ind[1])
    rdm = rdm[triu_ind]

    if normalize:
        rdm = rdm / p

    return rdm


def pairwise_euclidean(
    activations1: torch.Tensor,
    activations2: torch.Tensor,
    normalize=False,
    method="fast",
):
    """Generate pairwise dissimilarity vectors using squared euclidean distances.

    Args:
        activations1 (torch.Tensor):
            (N, ...) model activations (model response) to N conditions, or stimulus images.
        activations1 (torch.Tensor):
            (N, ...) model activations (model response) to N conditions, or stimulus images.
        normalize (bool)
            If True, divide distances by the number of features.

    Returns:
        pairwise_dissimilarities (torch.Tensor): an N-long vector of pairwise dissimilarities, such that
            pairwise_dissimilarities[i] is ||activations1[i]-activations2[i]||^2
    """

    if activations1.ndim != 2:
        activations1 = activations1.flatten(1, -1)
    if activations2.ndim != 2:
        activations2 = activations2.flatten(1, -1)

    n, p = activations1.shape
    assert activations1.shape == activations2.shape

    d = torch.linalg.vector_norm(activations1 - activations2, axis=1) ** 2

    if normalize:
        d = d / p

    return d


def rdm_euclidean_reduce_space(activations: torch.Tensor, normalize=False):
    """convert a 4D activation tensor to an n_distances x n_channels x 1 x 1 stack of RDMs"""
    N, C, H, W = activations.shape

    a = activations.reshape(N, C, H * W).permute((1, 0, 2))
    a_T = a.permute((0, 2, 1))
    rdm = (
        -2 * torch.matmul(a, a_T)
        + torch.sum(a**2, axis=-1).unsqueeze(-1)
        + torch.sum(a**2, axis=-1).unsqueeze(1)
    )
    triu_ind = torch.triu_indices(N, N, offset=1)
    rdm = rdm[:, triu_ind[0], triu_ind[1]].T.unsqueeze(axis=-1).unsqueeze(axis=-1)
    if normalize:
        rdm = rdm / (H * W)
    return rdm  # n_distances x C x 1 x 1


def rdm_euclidean_reduce_channels(activations: torch.Tensor, normalize=False):
    """convert a 4D activation tensor to an n_distances x 1 x H x W stack of RDMs"""
    N, C, H, W = activations.shape

    a = activations.permute((2, 3, 0, 1))
    a_T = a.permute((0, 1, 3, 2))
    rdm = (
        -2 * torch.matmul(a, a_T)
        + torch.sum(a**2, axis=-1).unsqueeze(-1)
        + torch.sum(a**2, axis=-1).unsqueeze(-2)
    )
    triu_ind = torch.triu_indices(N, N, offset=1)
    rdm = rdm[:, :, triu_ind[0], triu_ind[1]].permute([2, 0, 1]).unsqueeze(1)
    if normalize:
        rdm = rdm / C
    return rdm  # n_distances x 1 x H x W


def rdm_cos_loss(rdm1: torch.Tensor, rdm2: torch.Tensor):
    """Cosine similarity between two RDMs."""
    cos_fn = nn.CosineSimilarity(dim=0)
    rdm1 = rdm1 / rdm1.norm(dim=0)
    rdm2 = rdm2 / rdm2.norm(dim=0)
    cos_sim = cos_fn(rdm1, rdm2)

    return cos_sim


def rdm_corr_loss(rdm1: torch.Tensor, rdm2: torch.Tensor, is_corr: bool = False):
    """Covariance or pearson correlation between two RDMs."""
    E1_dif = rdm1 - torch.mean(rdm1)
    E2_dif = rdm2 - torch.mean(rdm2)

    cov = torch.sum(E1_dif * E2_dif)

    if is_corr == False:
        cost = cov
    else:
        cost = cov / torch.sqrt(torch.sum(E1_dif**2) * torch.sum(E2_dif**2))

    return cost


def rdm_plot(nConds: int, rdm_vector: torch.Tensor, title: str, path: str):
    """Plotting RDM with flattened upper RDM vector."""

    rdm = np.zeros((nConds, nConds))
    cmap = rdm_colormap_classic()

    # rdm_vector = rdm_vector/np.linalg.norm(rdm_vector, axis=0)
    count = 0
    for i in range(0, nConds):
        for j in range(i + 1, nConds):
            rdm[i, j] = rdm_vector[count]
            count = count + 1

    rdm = rdm + rdm.T

    fig = plt.figure(figsize=(10, 10))
    rdm = plt.matshow(rdm, cmap=cmap)
    plt.axis("off")
    plt.colorbar(rdm)

    plt.suptitle("%s" % (title), fontsize=10)

    fig.subplots_adjust(top=0.93)

    plt.savefig(path)
    plt.close()


def rdm_stimuli_plot(
    nConds: int,
    rdm_vector: torch.Tensor,
    stim_path: str,
    title: str,
    save_path: str,
    target: bool = False,
    cropped: bool = False,
    percentile: bool = False,
):

    rdm = np.zeros((nConds, nConds))
    # cmap = rdm_colormap()
    cmap = plt.get_cmap("gray_r")

    # rdm_vector = rdm_vector/np.linalg.norm(rdm_vector, axis=0)

    count = 0
    for i in range(0, nConds):
        for j in range(i + 1, nConds):
            rdm[i, j] = rdm_vector[count]
            count = count + 1

    rdm = rdm + rdm.T

    if percentile:
        rdm = scipy.stats.rankdata(rdm).reshape(rdm.shape)
        rdm = rdm - np.min(rdm)
        rdm = rdm / np.max(rdm)

    inch_per_im = 11 / 48
    titlespace = max((nConds + 1) / 12, 0.5)
    title_font_size = max(round(titlespace * inch_per_im * 0.50 * 72), 6)

    height_ratios = np.asarray([titlespace, nConds + 1, 0.25]) * inch_per_im
    width_ratios = np.asarray([0.25, nConds + 1, 0.25, 0.5, 1.0]) * inch_per_im
    W = np.sum(width_ratios)
    H = np.sum(height_ratios)

    fig = plt.figure(figsize=(W, H))
    gs0 = fig.add_gridspec(
        nrows=3,
        ncols=5,
        wspace=0,
        hspace=0,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        left=0,
        right=1,
        bottom=0,
        top=1,
    )

    left_cell = gs0[1, 1]
    right_cell = gs0[1, 3]

    RDM_grid_gs = gridspec.GridSpecFromSubplotSpec(
        nrows=nConds + 1, ncols=nConds + 1, subplot_spec=left_cell, wspace=0, hspace=0
    )

    fname = "target" if target == True else "current"

    for i in range(nConds):

        filename = os.path.join(stim_path, fname + str(i))

        if cropped:
            filename = filename + "_cropped"

        filename = filename + ".png"
        img = mpimg.imread(filename)

        ax1 = fig.add_subplot(RDM_grid_gs[0, i + 1])
        ax1.imshow(img)
        ax1.axis("off")
        ax1.grid("off")
        ax2 = fig.add_subplot(RDM_grid_gs[i + 1, 0])
        ax2.imshow(img)
        ax2.axis("off")
        ax2.grid("off")

    axbig = fig.add_subplot(RDM_grid_gs[1:, 1:])
    rdm = axbig.matshow(rdm, cmap=cmap)
    axbig.axis("off")

    ax_color = fig.add_subplot(right_cell)
    cbar = fig.colorbar(rdm, cax=ax_color)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0", "25", "50", "75", "100"])
    plt.suptitle("%s" % (title), fontsize=title_font_size)

    # fig.subplots_adjust(top=0.93)

    # fig.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(save_path, dpi=600)
    plt.close()
