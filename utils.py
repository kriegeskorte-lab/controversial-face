"""from the YOLO5Face repo: https://github.com/deepcam-cn/yolov5-face"""

import math
from packaging import version
import os, errno
from collections.abc import Iterable
import warnings
import pickle
import re
import json

import psutil
import numpy as np
import scipy.stats
import torch
import torchvision as tv
import rsatoolbox
import PIL
import git
import h5py


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    # https://stackoverflow.com/a/312464
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def extract_triu(X, diagonal=1):
    """extract upper triangular part of matrix

    args:
    X (torch.tensor) last two dimensions are treated as defining 2D matrices.
    diagonal (int) 1 for excluding major diagonal, 0 for including it
    """

    assert X.shape[-1] == X.shape[-2]
    n_conditions = X.shape[-1]
    mask = torch.triu(
        torch.ones((n_conditions, n_conditions), device=X.device, dtype=torch.bool),
        diagonal=diagonal,
    )
    return X[..., mask]


def triu_to_full(X, diagonal=1):
    """distances matrix to symmetric RDM

    args:
    X (torch.tensor) the last dimension defines distances
    diagonal (int) 1 for excluding major diagonal, 0 for including it
    """

    assert diagonal > 0, "zero diagonal not supported"
    N_dists = X.shape[-1]
    n_conditions = int(0.5 + math.sqrt(1 / 4 + 2 * N_dists))
    new_shape = list(X.shape[:-1]) + [n_conditions, n_conditions]
    full = torch.zeros(new_shape, dtype=X.dtype, device=X.device)
    mask = torch.triu(
        torch.ones((n_conditions, n_conditions), device=X.device, dtype=torch.bool),
        diagonal=diagonal,
    )
    full[..., mask] += X
    full += torch.transpose(full, -1, -2)

    return full


def _test_full_to_triu_and_back():
    # generate non-negative symmetric matrix
    full = torch.randn(3, 4, 5, 50, 50)
    full = (full - torch.transpose(full, -1, -2)) ** 2
    dists = extract_triu(full)
    full2 = triu_to_full(dists)
    assert torch.isclose(full, full2).all()


def numerically_stable_cosine_similarity(a, b, dim, keepdim=False, eps=1e-16):
    # norm_a=torch.clamp(torch.norm(a,p=2,dim=dim,keepdim=True),min=eps)
    # norm_b=torch.clamp(torch.norm(b,p=2,dim=dim,keepdim=True),min=eps)

    norm_a = torch.norm(a, p=2, dim=dim, keepdim=True) + eps
    norm_b = torch.norm(b, p=2, dim=dim, keepdim=True) + eps

    cosine_similarity = ((a / norm_a) * (b / norm_b)).sum(dim=dim, keepdim=keepdim)
    return cosine_similarity


def batched_bilinear_form(x, y, A, dim, keepdim=False):
    """calculate u@A@v where cov is 2d and u and v are subvectors along the dim-th dimension of x and y

    More explicitly:
    batched_bilinear_form(x,y,A,dim)[i,j,k]==x[1,...,dim-1,dim+1..,M] @ A @ y[1,...,dim-1,dim+1..,M]

    args:
    x (torch.tensor) at least 2D tensor
    y (torch.tensor) at least 2D tensor
    A (torch.tensor) must be 2D
    dim (int) dimension to reduce
    keepdim (boolean)
    """

    p = x.shape[dim]
    # move dim to last dimension (required by matmul):
    u = x.transpose(dim, -1).unsqueeze(-2)  # row vector
    v = y.transpose(dim, -1).unsqueeze(-1)  # column vector

    A = A.unsqueeze(0)
    assert A.shape == (1, p, p)

    A_v = torch.matmul(A, v)
    assert A_v.shape == v.shape

    u_A_v = torch.matmul(u, A_v).squeeze(-1).squeeze(-1)
    if keepdim:
        u_A_v = u_A_v.unsqueeze(dim)
    return u_A_v


def _test_batched_bilinear_form():
    from sklearn.neighbors import DistanceMetric
    import numpy as np

    u = torch.rand((30, 40))
    v = torch.rand((30, 40))

    X = torch.rand((1000, 40))

    cov = X.T @ X
    precision = torch.inverse(cov)
    torch_mahal = batched_bilinear_form(u - v, u - v, precision, dim=1)

    dist = DistanceMetric.get_metric("mahalanobis", V=cov.numpy())

    torch_sklearn = np.stack(
        [
            dist.pairwise(u[i : i + 1].numpy(), v[i : i + 1].numpy()).squeeze() ** 2
            for i in range(len(u))
        ]
    )

    print(torch_mahal)
    print(torch_sklearn)

    assert np.all(
        np.isclose(torch_mahal.numpy(), torch_sklearn)
    ), "The numbers don't match :("


def spearman_brown_correction(rho):
    if rho > 0:
        return 2 * rho / (1 + rho)
    else:
        return rho


def whitened_unbiased_cosine_similarity(
    a, b, dim, V=None, inv_V=None, keepdim=False, eps=0
):
    """a torch implementation of eq 12 from Diedrichsen et al., 2020"""

    if inv_V is None:
        assert V is not None, "either V or inv_V must be provided"
        inv_V = torch.inverse(V)

    numerator = batched_bilinear_form(a, b, inv_V, dim=dim, keepdim=keepdim)
    denominator = torch.sqrt(
        batched_bilinear_form(a, a, inv_V, dim=dim, keepdim=keepdim)
        * batched_bilinear_form(b, b, inv_V, dim=dim, keepdim=keepdim)
    )
    WUC = numerator / (denominator + eps)
    return WUC


class QuickWhitenedUnbiasedCosineSimilarity(torch.nn.Module):
    """A PyTorch Module for calculating Whitened Cosine Similarity/Pearson Correlation assuming unit sigma_k"""

    def __init__(self, n_cond=None, n_dist=None):

        super().__init__()
        # make sure specified rdm dimensions make sense
        assert (n_cond is not None) or (
            n_dist is not None
        ), "specify either n_cond or n_dist"
        if n_dist is None:
            n_dist = int(n_cond * (n_cond - 1) / 2)
        elif n_cond is None:
            n_cond = int(np.ceil(np.sqrt(n_dist * 2)))
        assert n_cond == int(np.ceil(np.sqrt(n_dist * 2)))
        assert n_dist == int(n_cond * (n_cond - 1) / 2)
        self.n_cond = n_cond
        self.n_dist = n_dist

        # precalculate sumI
        rowI, colI = rsatoolbox.util.matrix.row_col_indicator_g(n_cond)
        sumI = rowI + colI
        self.register_buffer("sumI", torch.tensor(sumI, dtype=torch.float32))

        self.sqrt2 = math.sqrt(2)

    def _cov_weighting(self, vector):
        """Transforms an array of RDM vectors in to representation
        in which the elements are isotropic. This is a stretched-out
        second moment matrix, with the diagonal elements appended.
        To account for the fact that the off-diagonal elements are
        only there once, they are multipled by 2
        Args:
            vector (numpy.ndarray):
                RDM vectors (2D) N x n_dist
        Returns:
            vector_w:
                weighted vectors (M x n_dist + n_cond)

        (adapted from rsagroup/rsatoolbox)
        """

        N, n_dist = vector.shape
        assert n_dist == self.n_dist
        vector_w = -0.5 * torch.cat(
            [
                vector,
                torch.zeros((N, self.n_cond), dtype=vector.dtype, device=vector.device),
            ],
            1,
        )

        # column and row means
        m = vector_w @ self.sumI.to(vector.device) / self.n_cond
        # Overall mean
        mm = torch.sum(vector_w * 2, axis=1, keepdims=True) / (self.n_cond**2)
        # subtract the column and row means and add overall mean
        vector_w = vector_w - m @ self.sumI.T.to(vector.device) + mm

        # Weight the off-diagnoal terms double
        vector_w[:, :n_dist] = vector_w[:, :n_dist] * self.sqrt2
        return vector_w

    def whitened_cosine_similarity(self, rdms1, rdms2, eps=0):
        """(Adapted from numpy code by Heiko Schutt)
        rdms1 (torch.Tensor) m_RDMs x n_dists
        rdms2 (torch.Tensor) n_RDMs x n_dists

        returns m_RDMS x n_RDMs (torch.Tensor) correlation matrix
        """

        # Compute the extended version of RDM vectors in whitened space
        rdms1_m = self._cov_weighting(rdms1)
        rdms2_m = self._cov_weighting(rdms2)

        # compute the inner products v1^T v2 for all combinations
        numerator = torch.einsum("ij,kj->ik", rdms1_m, rdms2_m)

        # divide by sqrt(v1^T v1)
        denominator1 = (
            torch.sqrt(torch.einsum("ij,ij->i", rdms1_m, rdms1_m))
            .reshape((-1, 1))
            .expand(numerator.shape)
        )
        # divide by sqrt(v2^T v2)
        denominator2 = (
            torch.sqrt(torch.einsum("ij,ij->i", rdms2_m, rdms2_m))
            .reshape((1, -1))
            .expand(numerator.shape)
        )
        WUC = numerator / (denominator1 * denominator2 + eps)
        return WUC

    def whitened_pearson_correlation(self, rdms1, rdms2, eps=0):
        def center(x):
            return x - torch.mean(x, dim=1, keepdim=True)

        return self.whitened_cosine_similarity(center(rdms1), center(rdms2), eps=eps)


def whitened_pearson_correlation(a, b, dim, V=None, inv_V=None, keepdim=False, eps=0):
    """a torch implementation of eq 13 from Diedrichsen et al., 2020"""

    def center(x):
        return x - torch.mean(x, dim=dim, keepdim=True)

    return whitened_unbiased_cosine_similarity(
        center(a), center(b), dim, V=V, inv_V=inv_V, keepdim=keepdim, eps=eps
    )


def _fake_rsatoolbox_rdms(n_obs, n_cond):
    signal = np.random.randn(n_cond, n_obs) * 1
    measurements1 = signal + np.random.randn(n_cond, n_obs) * 1.0
    measurements2 = signal + np.random.randn(n_cond, n_obs) * 1.0

    rdm1 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(measurements1))
    rdm2 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(measurements2))
    return rdm1, rdm2


def _test_whitened_unbiased_cosine_similarity():
    for i in range(100):
        n_obs, n_cond = 200, 30
        rdm1, rdm2 = _fake_rsatoolbox_rdms(n_obs, n_cond)

        import rsatoolbox

        output_rsatoolbox = rsatoolbox.rdm.compare(
            rdm1, rdm2, method="cosine_cov", sigma_k=None
        )

        obj = QuickWhitenedUnbiasedCosineSimilarity()
        V = _get_v(n_cond, None).todense()
        V = torch.tensor(V, dtype=torch.float32)
        a = torch.tensor(rdm1.dissimilarities, dtype=torch.float32)
        b = torch.tensor(rdm2.dissimilarities, dtype=torch.float32)
        output_this = whitened_unbiased_cosine_similarity(a, b, V=V, dim=1)

        assert np.isclose(output_rsatoolbox, output_this)
    print("great success!")


def _test_quick_whitened_unbiased_cosine_similarity():
    for i in range(100):
        n_obs, n_cond = 200, 30
        rdm1, rdm2 = _fake_rsatoolbox_rdms(n_obs, n_cond)

        import rsatoolbox

        output_rsatoolbox = rsatoolbox.rdm.compare(
            rdm1, rdm2, method="corr_cov", sigma_k=None
        )

        obj = QuickWhitenedUnbiasedCosineSimilarity(n_cond=n_cond)

        a = torch.tensor(rdm1.dissimilarities, dtype=torch.float32)
        b = torch.tensor(rdm2.dissimilarities, dtype=torch.float32)

        output_this = obj.whitened_pearson_correlation(a, b)

        assert np.isclose(output_rsatoolbox, output_this)
    print("great success!")

    # now test vectorized compuation
    n_obs, n_cond = 200, 30
    rdms1 = []
    rdms2 = []
    for i in range(5):

        rdm1, rdm2 = _fake_rsatoolbox_rdms(n_obs, n_cond)
        rdms1.append(rdm1)
        rdms2.append(rdm2)
    rdms2 = rdms2[:3]

    rdms1 = rsatoolbox.rdm.concat(rdms1)
    rdms2 = rsatoolbox.rdm.concat(rdms2)

    output_rsatoolbox = rsatoolbox.rdm.compare(
        rdms1, rdms2, method="corr_cov", sigma_k=None
    )

    print("rdmtoolbox:")
    print(output_rsatoolbox)

    print("pytorch port:")

    a = torch.tensor(rdms1.dissimilarities, dtype=torch.float32)
    b = torch.tensor(rdms2.dissimilarities, dtype=torch.float32)

    output_this = obj.whitened_pearson_correlation(a, b)
    print(output_this)


def pearson_correlation(a, b, dim, keepdim=False, eps=1e-16):
    ma = torch.mean(a, dim=dim, keepdim=True)
    mb = torch.mean(b, dim=dim, keepdim=True)

    sa = torch.std(a, dim=dim, keepdim=True, unbiased=False) + eps
    sb = torch.std(b, dim=dim, keepdim=True, unbiased=False) + eps

    r = (((a - ma) / sa) * ((b - mb) / sb)).mean(dim=dim, keepdim=keepdim)
    return r


def covariance(a, b, dim, keepdim=False):
    """univariate covariance within dim"""
    ma = torch.mean(a, dim=dim, keepdim=True)
    mb = torch.mean(b, dim=dim, keepdim=True)

    cov = ((a - ma) * (b - mb)).mean(dim=dim, keepdim=keepdim)
    return cov


def RDM_dim_from_upper_triangular_part_vector_length(n_dists, exclude_diagonal=True):
    """given the length of a vectorized upper triangular part of a matrix, calculate the original size of the matrix

    args:
    n_dists (int) how many distances in the upper triangular part
    exclude_diagonal (bool) True if the major diagonal was excluded when extracting the distances, False if it's included.
    """

    if exclude_diagonal:
        n_conditions = (+1 + math.sqrt(1 + 8 * n_dists)) / 2
    else:
        n_conditions = (-1 + math.sqrt(1 + 8 * n_dists)) / 2
    assert n_conditions == int(
        n_conditions
    ), "distance vector length inconsistent with upper triangular part"
    n_conditions = int(n_conditions)
    return n_conditions


def gpu_allocation(
    readouts,
    gpu_allocation_strategy="simple",
    n_gpus=None,
    n_gpus_for_rendering=1,
    sampling_device="cpu",
    models_and_renderers_share_gpu=None,
):
    """allocate GPUs to differentiable rendering and model evaluation

    args:
    readouts (list) a list of readout dictionaries
    gpu_allocation_strategy (str) currently only 'simple' is implemented
    n_gpus (int) total number of GPUs to use. If None, use all available GPUs.
    n_gpus_for_rendering (int) number of GPUs to use for rendering. Default is 1.
    sampling_device (str) 'cpu' or 'gpu' -  used for sampling based computations, where applicable
    models_and_renderers_share_gpu (bool) if True, models and renderers share the GPUs. Default: determine by GPU availability.

    returns
    readouts (list) with GPU indices added to the readout dictionary

    """

    if n_gpus is None:
        n_gpus = torch.cuda.device_count()

    assert n_gpus_for_rendering <= n_gpus, "n_gpus_for_rendering must be <= n_gpus"

    if models_and_renderers_share_gpu is None:
        models_and_renderers_share_gpu = n_gpus_for_rendering == n_gpus

    if n_gpus > 0:
        print("Using {} GPUs.".format(n_gpus))
    else:
        print("WARNING: CPU only, this will be slow!")

    if gpu_allocation_strategy == "simple":
        # first, pytorch3d gets its own device(s).
        # next, place each model on its own GPU, if possible. Don't reuse input layers.

        n_models = len(readouts)

        if n_gpus > 0:
            available_gpus = list(range(n_gpus))

            # set a GPU for sampling, if requested
            if sampling_device == "cpu":
                sampling_device = torch.device("cpu")
            elif sampling_device == "gpu":
                sampling_device = torch.device(f"cuda:{available_gpus.pop(-1)}")
            else:
                raise ValueError('sampling_device must be "cpu" or "gpu"')

            if not models_and_renderers_share_gpu:
                pytorch3d_devices = [
                    torch.device(f"cuda:{available_gpus.pop(0)}")
                    for _ in range(n_gpus_for_rendering)
                ]
                gpu_allocation_counter = 0
            else:
                pytorch3d_devices = [
                    torch.device(f"cuda:{available_gpus[i_gpu]}")
                    for i_gpu in range(n_gpus_for_rendering)
                ]
                gpu_allocation_counter = n_gpus_for_rendering
        else:
            pytorch3d_devices = torch.device("cpu")
            sampling_device = torch.device("cpu")
            available_gpus = None

        gpus_for_models = set()

        for i_readout, readout in enumerate(readouts):
            if available_gpus is not None:
                cur_gpu_id = available_gpus[
                    gpu_allocation_counter % len(available_gpus)
                ]
                cur_device = torch.device("cuda:{}".format(cur_gpu_id))
                gpu_allocation_counter += 1
                gpus_for_models.add(cur_device)
            else:
                cur_device = torch.device("cpu")

            readout["model_device"] = cur_device
            readout["input_layer_device"] = cur_device

        print(
            f"gpus for pytorch3d:{pytorch3d_devices}, gpus_for_models:{gpus_for_models}, device for sampling:{sampling_device}"
        )

        # determine where are we sampling RDMs
    #             if n_gpus > len(readouts)+1:
    #                 sampling_device = torch.device('cuda:{}'.format(n_gpus-1))
    #             else: # don't place sampling on the same GPU as one of the models or PyTorch3D, it takes too much GPU RAM
    #                 sampling_device = torch.device('cpu')

    elif gpu_allocation_strategy == "only_models":
        assert (
            n_gpus_for_rendering == 0
        ), "n_gpus_for_rendering must be 0 when using gpu_allocation_strategy=only_models"
        gpus_for_models = list(range(torch.cuda.device_count()))
        for i_readout, readout in enumerate(readouts):

            if gpus_for_models is not None:
                cur_gpu_id = gpus_for_models[i_readout % len(gpus_for_models)]
                cur_device = torch.device("cuda:{}".format(cur_gpu_id))
            else:
                cur_device = torch.device("cpu")

            readout["model_device"] = cur_device
            readout["input_layer_device"] = cur_device
        sampling_device = None
        pytorch3d_devices = None
    elif gpu_allocation_strategy == "efficient":
        raise NotImplementedError

    return readouts, pytorch3d_devices, sampling_device


def new_gpu_allocation(
    readouts, n_gpus=None, pytorch3d_gpu_usage="exclusive", sampling_gpu_usage="cpu"
):
    """assign GPUs to pytorch3d, DNN models, and RDM sampling
    args:
        readouts
        n_gpus
        pytorch3d_device: 'cpu'/'shared'/'exclusive'
        sampling_device: 'cpu'/'shared'/'exclusive'
    """

    if n_gpus is None:
        n_gpus = torch.cuda.device_count()

    if n_gpus > 0:
        print("Using {} GPUs.".format(n_gpus))
    else:
        print("WARNING: CPU only, this will be slow!")
        pytorch3d_device = torch.device("cpu")
        sampling_device = torch.device("cpu")
        gpus_for_models = None
    # first, pytorch3d gets its own device.
    # next, place each model on its own GPU, if possible. Don't reuse input layers.

    n_models = len(readouts)
    available_gpus = [g for g in range(n_gpus)]
    if n_gpus > 0:
        if pytorch3d_gpu_usage != "cpu":
            pytorch3d_device = torch.device("cuda:{}".format(available_gpus[0]))
            if pytorch3d_gpu_usage == "exclusive":
                available_gpus.pop(0)
        else:
            pytorch3d_device = torch.device("cpu")

        if sampling_gpu_usage != "cpu":
            sampling_device = torch.device(f"cuda:{available_gpus[-1]}")
            if sampling_gpu_usage == "exclusive":
                available_gpus.pop(-1)
        else:
            sampling_device = torch.device("cpu")

    for i_readout, readout in enumerate(readouts):
        if len(available_gpus) > 0 is not None:
            cur_gpu_id = available_gpus[i_readout % len(available_gpus)]
            cur_device = torch.device("cuda:{}".format(cur_gpu_id))
        else:
            cur_device = torch.device("cpu")

        readout["model_device"] = cur_device

    print(
        f"gpu for pytorch3d:{pytorch3d_device}, gpus_for_models:{available_gpus}, gpu for sampling:{sampling_device}"
    )
    return readouts, pytorch3d_device, sampling_device


class ConvergenceCheck:
    def __init__(self, window_length=20):
        self.window_length = window_length
        self.loss_history = []

    def update_and_check_convergence(self, loss):
        """updates loss history and returns True if convergence criterion has been met"""

        self.loss_history.append(loss)
        if len(self.loss_history) < self.window_length * 2:
            # we haven't seen enough time steps to decide
            return False
        prev_window = np.asarray(
            self.loss_history[(-self.window_length * 2) : (-self.window_length)]
        )
        cur_window = np.asarray(self.loss_history[(-self.window_length) :])

        if np.mean(cur_window) > np.mean(prev_window):
            return True

        # cohens_d = (np.mean(cur_window) - np.mean(prev_window)) / np.sqrt(0.5*np.var(cur_window)+0.5*np.var(prev_window))

        t, p = scipy.stats.ttest_ind(cur_window, prev_window)
        return p >= 0.1


class Constraint:
    # abstract class for constraints
    def __init__(self):
        raise NotImplementedError

    def compress(self, x):
        raise NotImplementedError

    def decompress(self, x):
        raise NotImplementedError

    def project(self, x):
        raise NotImplementedError


class DummyConstraint(Constraint):
    def __init__(self, *args, **kwargs):
        pass

    def compress(self, x):
        return x

    def decompress(self, x):
        return x

    def project(self, x):
        return x


class ConstrainedSquaredL2NormReparameterization(Constraint):
    def __init__(
        self, max_squared_l2_norm, num_dims, dim=-1, eps=1e-6, center=None, ev=None
    ):
        """

        Args:
            max_squared_l2_norm: maximum squared L2 norm of the vector
            dim: dimension over which the constraint is applied
            eps: small number to add to the allowed norm to avoid numerical issues
            center: if not None, the vector is centered around this point. should be broadcastable to the same shape as the x arg for compress and decompress
            ev: eigenvalue for the latents
        """

        self.max_squared_l2_norm = max_squared_l2_norm
        self.dim = dim
        self.center = center
        self.eps = eps
        self.num_dims = num_dims
        if ev is not None:
            self.ev = torch.diag(ev[: self.num_dims])
            assert (
                self.center is None
            ), "centering not supported for constrained coordinate space"
        else:
            self.ev = ev

        if version.parse(torch.__version__) >= version.parse("1.7.0"):  # PyTorch >=1.7
            self.norm_fun = (
                lambda x: torch.linalg.norm(x, ord=2, dim=self.dim, keepdim=True) ** 2
            )
        else:
            self.norm_fun = (
                lambda x: torch.norm(x, p=2, dim=self.dim, keepdim=True) ** 2
            )

    def compress(self, x):
        """compress x such that the squared norm of infinity is compressed to the bound"""
        if self.ev is not None:
            sq_norms = self.norm_fun(x @ self.ev)
        else:
            sq_norms = self.norm_fun(x)  # norm.shape == n x 1

        compressed_sq_norms = torch.tanh(sq_norms) * (
            self.max_squared_l2_norm - self.eps
        )
        scale_factors = torch.sqrt(compressed_sq_norms / sq_norms)
        scale_factors = torch.where(
            sq_norms == 0, torch.ones_like(scale_factors), scale_factors
        )  # deal with the 0/0 case
        compressed_x = x * scale_factors
        if (
            self.center is not None
        ):  # if a center was specified, norm bounding is relative to that center
            compressed_x = compressed_x + self.center
        return compressed_x

    def decompress(self, x):
        """decompress norm-bounded x to unbounded x"""
        if self.ev is not None:
            sq_norms = self.norm_fun(x @ self.ev)
        else:
            if (
                self.center is not None
            ):  # if a center was specified, norm bounding is relative to that center
                x = x - self.center
            sq_norms = self.norm_fun(x)

        assert torch.all(
            sq_norms <= self.max_squared_l2_norm
        ), f"x must be admissible but {sq_norms.max()} > {self.max_squared_l2_norm}."
        decompressed_sq_norms = torch.atanh(
            sq_norms / (self.max_squared_l2_norm + self.eps)
        )
        scale_factors = torch.sqrt(decompressed_sq_norms / sq_norms)
        scale_factors = torch.where(
            sq_norms == 0, torch.ones_like(scale_factors), scale_factors
        )  # deal with the 0/0 case
        decompressed_x = x * scale_factors
        return decompressed_x

    def project(self, x):
        """project x into the norm-bounded ball"""
        if self.ev is not None:
            sq_norms = self.norm_fun(x @ self.ev)
        else:
            if (
                self.center is not None
            ):  # if a center was specified, norm bounding is relative to that center
                x = x - self.center
            sq_norms = self.norm_fun(x)

        clamped_sq_norms = sq_norms.clamp(max=self.max_squared_l2_norm - self.eps)
        scale_factors = torch.sqrt(clamped_sq_norms / sq_norms)
        scale_factors = torch.where(
            sq_norms == 0, torch.ones_like(scale_factors), scale_factors
        )  # deal with the 0/0 case
        projected_x = x * scale_factors
        if (
            self.center is not None
        ):  # if a center was specified, norm bounding is relative to that center
            projected_x = projected_x + self.center

        print("maximum projected sq norm:", self.norm_fun(projected_x).max())
        return projected_x


def _test_ConstrainedSquaredL2NormReparameterization():
    # initializing stimulus parameters (10 stimuli, 199 features)
    x0 = torch.normal(mean=0, std=1, size=[10, 199])

    reparam = ConstrainedSquaredL2NormReparameterization(250.0)

    x0 = reparam.project(x0)  # make sure all x0 examples are admissible

    max_sq_norm = reparam.norm_fun(x0).max()
    print("max sq norm:", max_sq_norm)
    x1 = reparam.decompress(x0)  # this is an unlimited-norm representation of x0

    x1.requires_grad_(True)
    optimizer = torch.optim.Adam([x1], lr=10)

    for i in range(10):
        optimizer.zero_grad()
        loss = -x1.sum()  # just an arbitray loss - maximize the sum of x
        loss.backward()
        optimizer.step()
        print("loss=", loss.item())

    x0 = reparam.compress(
        x1
    )  # go back to the norm-bounded representation of the solution

    max_sq_norm = reparam.norm_fun(x0).max()
    print("sq-norms", reparam.norm_fun(x0).squeeze())
    print("max sq norm:", max_sq_norm)
    assert max_sq_norm <= 250 + 1e-3


def _test_ConstrainedSquaredL2NormReparameterization_EV(shape_EV, device):
    x0 = torch.normal(mean=0, std=1, size=[10, 199], device=device)
    reparam = ConstrainedSquaredL2NormReparameterization(
        90 * 199, ev=shape_EV, eps=1e-2, num_dims=199
    )
    x0 = reparam.project(x0)
    max_sq_norm = reparam.norm_fun(x0 @ reparam.ev).max()
    print("max sq norm:", max_sq_norm)
    x1 = reparam.decompress(x0)
    x1.requires_grad_(True)
    optimizer = torch.optim.Adam([x1], lr=10)

    for i in range(10):
        optimizer.zero_grad()
        loss = -x1.sum()  # just an arbitray loss - maximize the sum of x
        loss.backward()
        optimizer.step()
        print("loss=", loss.item())

    x0 = reparam.compress(
        x1
    )  # go back to the norm-bounded representation of the solution

    max_sq_norm = reparam.norm_fun(x0 @ reparam.ev).max()
    print("sq-norms", reparam.norm_fun(x0 @ reparam.ev).squeeze())
    print("max sq norm:", max_sq_norm)


class LatentBoxConstraintReparameterization(Constraint):
    def __init__(self, max_abs_latent, center=None, eps=1e-6, num_dims=None):
        """
        Args:
            max_abs_latent: maximum absolute value of the latent variable
            center: if not None, the vector is centered around this point. should be broadcastable to the same shape as the x arg for compress and decompress
            eps: small number to add to the allowed range to avoid numerical issues
        """

        self.max_abs_latent = max_abs_latent
        self.center = center
        self.eps = eps
        self.num_dims = num_dims  # top n dims to be used in L-inf constraint.

    def compress(self, x):
        """compress x such that an individual latent value of infinity is compressed to the bound"""
        compressed_x = torch.tanh(x) * (self.max_abs_latent + self.eps)
        if (
            self.center is not None
        ):  # if a center was specified, norm bounding is relative to that center
            compressed_x = compressed_x + self.center
        return compressed_x

    def decompress(self, x):
        """decompress the bounded x to unbounded x"""
        if (
            self.center is not None
        ):  # if a center was specified, norm bounding is relative to that center
            x = x - self.center
        decompressed_x = torch.atanh(x / (self.max_abs_latent + self.eps))
        return decompressed_x

    def project(self, x):
        """project x into the box"""
        if (
            self.center is not None
        ):  # if a center was specified, norm bounding is relative to that center
            x = x - self.center
        projected_x = x.clamp(min=-self.max_abs_latent, max=self.max_abs_latent)
        if self.center is not None:
            projected_x = projected_x + self.center
        return projected_x


def constraint_factory(
    ball_radius=None,
    box_SD=None,
    original_latent=None,
    center_around_original_latent=False,
    eps=1e-3,
    dim=-1,
    verbose=False,
    latent_name="",
    ev=None,
    num_dims=None,
) -> Constraint:
    """
    Returns:
        An object that projects a vector into a box or a ball
        args:
            ball_radius: if not None, the vector is projected into a ball of this radius
            box_SD: if not None, the vector is projected into a box of this max BFM absolute standard deviation
            original_latent (torch.Tensor): the latent tensor to be constrainted
            center_around_original_latent: if True, the center of the box is set to the original latent variable
            eps: small number to add to the allowed range to avoid numerical issues
            dim: the dimension to constrain (defaults to the last dimension)
            verbose: if True, print out the name of the latent variable and the constraint
            latent_name: the name of the latent variable (used only if verbose is True)
            ev: eigenvalues of the BFM latents
            num_dims: number of dimensions of the latents to be used in L2 norm.
    """
    assert (
        ball_radius is not None or box_SD is not None
    ), "must specify either ball_radius or box_SD"
    if center_around_original_latent:
        center = original_latent.clone()
    else:
        center = None

    constrained_top_n_dims = (
        original_latent.shape[dim] if num_dims is None else num_dims
    )

    if ball_radius is not None:
        if verbose:
            print(
                f"Creating ball constraint with radius {ball_radius} * {constrained_top_n_dims} and latent dim {constrained_top_n_dims} for {latent_name}."
            )
        return ConstrainedSquaredL2NormReparameterization(
            max_squared_l2_norm=ball_radius * constrained_top_n_dims,
            center=center,
            eps=eps,
            dim=dim,
            ev=ev,
            num_dims=constrained_top_n_dims,
        )
    elif box_SD is not None:
        if verbose:
            print(
                f"Creating box constraint with SD {box_SD} and latent dim {original_latent.shape[dim]} for {latent_name}."
            )
        return LatentBoxConstraintReparameterization(
            box_SD, center=center, eps=eps, num_dims=constrained_top_n_dims
        )
    else:
        return None


def _test_LatentBoxConstraintReparameterization():
    # initializing stimulus parameters (10 stimuli, 199 features)
    x0 = torch.normal(mean=0, std=1, size=[10, 199])

    reparam = LatentBoxConstraintReparameterization(3.0)

    x0 = reparam.project(x0)  # make sure all x0 examples are admissible

    print("max value:", x0.max(dim=-1)[0], "min value:", x0.min(dim=-1)[0])
    x1 = reparam.decompress(x0)  # this is an unlimited-norm representation of x0

    x1.requires_grad_(True)
    optimizer = torch.optim.Adam([x1], lr=10)

    for i in range(10):
        optimizer.zero_grad()
        loss = -(x1**2).sum()  # just an arbitray loss - maximize the sum of squared x
        loss.backward()
        optimizer.step()
        print("loss=", loss.item())

    x0 = reparam.compress(
        x1
    )  # go back to the norm-bounded representation of the solution

    print("max value:", x0.max(dim=-1)[0], "min value:", x0.min(dim=-1)[0])
    assert torch.max(torch.abs(x0)) <= 3 + 1e-3


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def silent_remove(filename):
    # https://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def log_ndtr(value: torch.Tensor):
    """
    Adapted from: https://github.com/pytorch/pytorch/issues/52973
    Function to compute the log of the normal CDF at value.
    This is based on the TFP implementation.

    TG: The fixed series expansion approximation for the lower domain was removed due to numerical issues
    """
    dtype = value.dtype
    if dtype == torch.float64:
        upper = 8
    elif dtype == torch.float32:
        upper = 5
    else:
        raise TypeError("value needs to be either float32 or float64")

    # When x < lower, then we perform a fixed series expansion (asymptotic)
    # = log(cdf(x)) = log(1 - cdf(-x)) = log(1 / 2 * erfc(-x / sqrt(2))) = log(-1 / sqrt(2 * pi) * exp(-x ** 2 / 2) / x * (1 + sum))
    # When x >= lower and x <= upper, then we simply perform log(cdf(x))
    # When x > upper, then we use the approximation log(cdf(x)) = log(1 - cdf(-x)) \approx -cdf(-x)
    normal = torch.distributions.Normal(0, 1)
    return torch.where(
        value > upper, torch.log1p(-normal.cdf(-value)), torch.log(normal.cdf(value))
    )


def ndtr(value: torch.Tensor):
    normal = torch.distributions.Normal(0, 1)
    return normal.cdf(value)


def reduce_weighted_logsumexp(
    logx, w=None, dim=None, keepdim=False, return_sign=False, name=None
):
    """Computes `log(abs(sum(weight * exp(elements across tensor dimensions))))`.
    A PyTorch version of a TFP function: https://github.com/tensorflow/probability/blob/v0.12.2/tensorflow_probability/python/math/generic.py#L232-L323

    If all weights `w` are known to be positive, it is more efficient to directly
    use `reduce_logsumexp`, i.e., `tf.reduce_logsumexp(logx + tf.log(w))` is more
    efficient than `du.reduce_weighted_logsumexp(logx, w)`.
    Reduces `input_tensor` along the dimensions given in `axis`.
    Unless `keepdim` is true, the rank of the tensor is reduced by 1 for each
    entry in `axis`. If `keepdim` is true, the reduced dimensions
    are retained with length 1.
    If `axis` has no entries, all dimensions are reduced, and a
    tensor with a single element is returned.
    This function is more numerically stable than log(sum(w * exp(input))). It
    avoids overflows caused by taking the exp of large inputs and underflows
    caused by taking the log of small inputs.
    For example:
    ```python
    x = torch.tensor([[0., 0, 0],
                [0, 0, 0]])
    w = torch.tensor([[-1., 1, 1],
                [1, 1, 1]])
    reduce_weighted_logsumexp(x, w)
    # ==> log(-1*1 + 1*1 + 1*1 + 1*1 + 1*1 + 1*1) = log(4)
    reduce_weighted_logsumexp(x, w, dim=0)
    # ==> [log(-1+1), log(1+1), log(1+1)]
    reduce_weighted_logsumexp(x, w, dim=1)
    # ==> [log(-1+1+1), log(1+1+1)]
    reduce_weighted_logsumexp(x, w, dim=1, keepdim=True)
    # ==> [[log(-1+1+1)], [log(1+1+1)]]
    reduce_weighted_logsumexp(x, w, dim=[0, 1])
    # ==> log(-1+5)
    ```
    Args:
    logx: The tensor to reduce. Should have numeric type.
    w: The weight tensor. Should have numeric type identical to `logx`.
    dim: The dimensions to reduce. If `None` (the default), reduces all
    dimensions. Must be in the range `[-rank(input_tensor),
    rank(input_tensor))`.
    keepdim: If true, retains reduced dimensions with length 1.
    return_sign: If `True`, returns the sign of the result.
    name: A name for the operation (optional).
    Returns:
    lswe: The `log(abs(sum(weight * exp(x))))` reduced tensor.
    sign: (Optional) The sign of `sum(weight * exp(x))`.
    """

    if w is None:
        lswe = torch.logsumexp(logx, dim=dim, keepdim=keepdim)
        if return_sign:
            sgn = torch.ones_like(lswe)
            return lswe, sgn
        return lswe

    log_absw_x = logx + torch.log(torch.abs(w))
    if dim is None:
        max_log_absw_x = torch.amax(log_absw_x)
    else:
        max_log_absw_x = torch.amax(log_absw_x, dim, keepdim=True)
    # If the largest element is `-inf` or `inf` then we don't bother subtracting
    # off the max. We do this because otherwise we'd get `inf - inf = NaN`. That
    # this is ok follows from the fact that we're actually free to subtract any
    # value we like, so long as we add it back after taking the `log(sum(...))`.
    max_log_absw_x = torch.where(
        torch.isinf(max_log_absw_x),
        torch.zeros([], dtype=max_log_absw_x.dtype, device=max_log_absw_x.device),
        max_log_absw_x,
    )
    wx_over_max_absw_x = torch.sign(w) * torch.exp(log_absw_x - max_log_absw_x)
    if dim is not None:
        sum_wx_over_max_absw_x = torch.sum(wx_over_max_absw_x, dim=dim, keepdim=keepdim)
    else:
        sum_wx_over_max_absw_x = torch.sum(wx_over_max_absw_x)
    if not keepdim:
        if dim is not None:  # multi-dim squeeze is not natively supported by pytorch
            if isinstance(dim, Iterable):
                for d in sorted(dim, reverse=True):
                    max_log_absw_x = torch.squeeze(max_log_absw_x, dim=d)
            else:
                max_log_absw_x = torch.squeeze(max_log_absw_x, dim=dim)
        else:
            max_log_absw_x = torch.squeeze(max_log_absw_x)
    sgn = torch.sign(sum_wx_over_max_absw_x)
    lswe = max_log_absw_x + torch.log(sgn * sum_wx_over_max_absw_x)
    if return_sign:
        return lswe, sgn
    return lswe


class sigmoid_based_log_norm_cdf(torch.nn.Module):
    # https://www.sciencedirect.com/science/article/pii/0096300395001905

    def __init__(self):
        super().__init__()
        self.log_sigmoid = torch.nn.LogSigmoid()
        self.sqrt_pi = math.sqrt(math.pi)

    def forward(self, z):
        return self.log_sigmoid(
            self.sqrt_pi * (-0.0004406 * (z**5) + 0.0418198 * (z**3) + 0.9000000 * (z))
        )


def _test_sigmoid_based_log_norm_cdf():
    import matplotlib.pyplot as plt

    f1 = sigmoid_based_log_norm_cdf()
    z = torch.linspace(-10.0, 10.0, 1001, dtype=torch.float32)
    z.requires_grad_(True)
    y1 = f1(z)
    for i in range(len(y1)):
        y1[i].backward(retain_graph=True)
    g1 = z.grad.detach().numpy()

    f2 = log_ndtr
    z = torch.linspace(-10.0, 10.0, 1001, dtype=torch.float32)
    z.requires_grad_(True)
    y2 = f2(z)
    for i in range(len(y2)):
        y2[i].backward(retain_graph=True)
    g2 = z.grad.detach().numpy()

    f3 = torch.nn.LogSigmoid()
    z = torch.linspace(-10.0, 10.0, 1001, dtype=torch.float32)
    z.requires_grad_(True)
    y3 = f3(z)
    for i in range(len(y3)):
        y3[i].backward(retain_graph=True)
    g3 = z.grad.detach().numpy()

    plt.figure()
    plt.plot(z.detach().numpy(), y1.detach().numpy())
    plt.plot(z.detach().numpy(), y2.detach().numpy())
    plt.plot(z.detach().numpy(), y3.detach().numpy())
    plt.legend(["sigmoid_based_log_norm_cdf", "log_ndtr", "logsigmoid"])
    plt.show()

    plt.figure()
    plt.plot(z.detach().numpy(), g1)
    plt.plot(z.detach().numpy(), g2)
    plt.plot(z.detach().numpy(), g3)
    plt.legend(["sigmoid_based_log_norm_cdf", "log_ndtr", "logsigmoid"])
    plt.show()


def zoom_crop(
    im,
    x1,
    x2,
    y1,
    y2,
    output_H: int,
    output_W: int,
    mode="bilinear",
    padding_mode="border",
    align_corners=True,
):
    """crops an image stack.
    args:
    im (torch.tensor) NCHW stack of images
    x1 (torch.tensor) N-long vector of top-left crop corner x coordinates
    x2 (torch.tensor) N-long vector of bottom-right crop corner x coordinates
    y1 (torch.tensor) N-long vector of top-left crop corner y coordinates
    y2 (torch.tensor) N-long vector of bottom-right crop corner y coordinates
    output_H (int) height of output images
    output_W (int) width of output images
    mode (str) interpolation mode to calculate output values 'bilinear' | 'nearest' | 'bicubic'.
    padding_mode (str) padding mode for outside grid values 'zeros' | 'border' | 'reflection'.
    align_corners (bool)

    The corner coordinates (x1,x2,y1 and y2) can be fractional.

    See https://stackoverflow.com/a/61973182 for transformation matrix conventions
    """

    N, C, H, W = im.shape

    # transform x1, x2, y1, y2 into normalized coordinates ([-1,1])
    # extreme cases:
    # 0,0 -> -1.0, -1.0
    # H-1,W-1 -> 1.0, 1.0

    x1 = x1 * 2.0 / (W - 1) - 1.0
    x2 = x2 * 2.0 / (W - 1) - 1.0

    y1 = y1 * 2.0 / (H - 1) - 1.0
    y2 = y2 * 2.0 / (H - 1) - 1.0

    # build transformation matrices
    x_scale = (x2 - x1) / 2.0
    y_scale = (y2 - y1) / 2.0
    x_shift = (x1 + x2) / 2.0
    y_shift = (y1 + y2) / 2.0

    theta = torch.stack(
        [
            torch.stack([x_scale, torch.zeros_like(x_scale), x_shift], axis=1),
            torch.stack([torch.zeros_like(y_scale), y_scale, y_shift], axis=1),
        ],
        axis=1,
    )

    # perform sampling
    grid = torch.nn.functional.affine_grid(
        theta, size=[N, C, output_H, output_W], align_corners=align_corners
    )
    cropped_im = torch.nn.functional.grid_sample(
        im, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )
    return cropped_im


def _zoom_crop_sanity_checks():
    import matplotlib.pyplot as plt

    # test that the crop works by cropping a coordinate grid:
    x1 = torch.tensor([22.5, 50])
    x2 = torch.tensor([60, 900])

    y1 = torch.tensor([51, 20])
    y2 = torch.tensor([80, 30])

    grid_y, grid_x = torch_meshgrid(torch.arange(256), torch.arange(256))

    im = torch.stack([grid_y.float(), grid_x.float()], axis=0).unsqueeze(0)

    im = torch.cat([im, im], axis=0)
    output_H = 64
    output_W = 64

    cropped_im = zoom_crop(im, x1, x2, y1, y2, output_H, output_W)

    print("x1=", cropped_im[:, 1, 0, 0])
    print("y1=", cropped_im[:, 0, 0, 0])

    print("x2=", cropped_im[:, 1, -1, -1])
    print("y2=", cropped_im[:, 0, -1, -1])

    print(cropped_im.shape)
    plt.figure()
    plt.imshow(cropped_im[1, 1].numpy())
    plt.colorbar()
    plt.show()


def mean_variation(x):
    """calculate the mean variation of tensor x. all non-batched dimensions are concatneated before the variances a calculated.

    args:
    x (torch.tensor) NxP tensor

    """
    assert x.ndim > 1

    if x.ndim > 2:
        x = x.view((x.shape[0], -1))
    N, P = x.shape

    return x.var(dim=0).mean()


def str2bool(v):  # https://stackoverflow.com/a/43357954
    from argparse import ArgumentTypeError

    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def _expand2square(pil_img, background_color):
    # source: https://github.com/nkmk/python-snippets/blob/0f6b4672097e91b00e51775ae1932aaf47b8977a/notebook/my_lib/imagelib.py
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = PIL.Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = PIL.Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def _rgba2rgb(rgba: torch.Tensor, background: float = 0.5):
    """Add background color to an RGBA image.

    Args:
        rgba (torch.Tensor): (4,H,W)
        background (float):
            uniform background color ranging from 0 to 1.

    Returns:
        rgb (torch.Tensor): (3,H,W)
    """

    ch, row, col = rgba.shape

    if ch == 3:
        return rgba

    rgb, alpha = rgba[:3, :, :], rgba[3, :, :]

    rgb = rgb * alpha.unsqueeze(0) + (1.0 - alpha.unsqueeze(0)) * background
    return rgb


def smoothmax(x, alpha, dim=0):
    return torch.logsumexp(x * alpha, dim=dim) / alpha


def smoothmin(x, alpha, dim=0):
    return -smoothmax(-x, alpha, dim=dim)


def minimum_abs_distance(rdm, alpha=1e-2):
    """compute hard and smooth minimum absolute distance between pairs of elements in a distance matrix

    Args:
        rdm (torch.tensor)
        alpha (float, optional): Defaults to 1e-2.
    """
    if rdm.ndim == 1:
        rdm = rdm.unsqueeze(1)
    num_pairs = rdm.shape[0]
    ind = torch.triu_indices(num_pairs, num_pairs, offset=1)
    abs_ds = torch.abs(rdm - rdm.permute(1, 0))
    abs_ds = abs_ds[ind[0], ind[1]]
    hard_minimum = torch.min(abs_ds)
    smooth_minimum = smoothmin(abs_ds, alpha=alpha)
    return hard_minimum, smooth_minimum


def mse(target_distances, rdm):
    """mean squared error between target distances and distances in a distance matrix"""
    sorted_rdm = torch.sort(rdm)[0]
    return torch.mean((target_distances - sorted_rdm) ** 2)


def load_im(stim_path, bg_color=0.5):
    """load a stimulus image into a torch tensor, potentially changing transparent pixels to background pixels and adding margins to make the image square"""

    if isinstance(stim_path, list):
        return torch.stack([load_im(f) for f in stim_path], axis=0)

    # load a stimulus image from disk
    pil_im = PIL.Image.open(stim_path)

    # expand non-square images to square images
    pil_im = _expand2square(pil_im, bg_color)

    torch_im = tv.transforms.ToTensor()(pil_im)

    # if images are in RGBA, generate RGB images with a specific uniform background color.
    torch_im = _rgba2rgb(torch_im, background=bg_color)
    return torch_im


import torch

_torch_version_meshgrid_indexing = version.parse(torch.__version__) >= version.parse(
    "1.10.0a0"
)


def torch_meshgrid(*tensors):
    """A wrapper of torch.meshgrid to compat different PyTorch versions.
    Since PyTorch 1.10.0a0, torch.meshgrid supports the arguments ``indexing``.
    So we implement a wrapper here to avoid warning when using high-version
    PyTorch and avoid compatibility issues when using previous versions of
    PyTorch.
    Args:
        tensors (List[Tensor]): List of scalars or 1 dimensional tensors.
    Returns:
        Sequence[Tensor]: Sequence of meshgrid tensors.

    adopted from open-mmlab
    """
    if _torch_version_meshgrid_indexing:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)  # Uses indexing='ij' by default


def write_down_git_hush(path):
    """save a text file with the hash of the current git commit.
    This is useful for reproducibility.

    args:
    path (str): path to the file to write
    see https://stackoverflow.com/a/41210204
    """

    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
    except:
        sha = "unknown"
        print("failed to get git hash")
    finally:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, "wt") as f:
            f.write(sha)


def omega_conf_to_hparams(omega_conf, prefix=None):
    """convert an omega configuration file to a flat dictionary of tensorboard hparams.
    nested dictionaries are recursively converted into a flat dictionary with the key as a a prefix.
    values are converted to hparams supported types (bool, string, float, int, or None)
    """
    if prefix is None:
        prefix = ""
    hparams = {}
    for key, value in omega_conf.items():
        if isinstance(value, dict):
            hparams.update(omega_conf_to_hparams(value, prefix=prefix + key + "."))
        else:
            if type(value) in [bool, str, float, int]:
                hparams[prefix + key] = value
            elif value is None:
                hparams[prefix + key] = None
            else:  # convert other types to string
                hparams[prefix + key] = str(value)
    return hparams


def write_dict_hdf5(file, dictionary):
    """writes a nested dictionary containing strings & arrays as data into
    a hdf5 file
    Args:
        file: a filename or opened writable file
        dictionary(dict): the dict to be saved
    """
    if isinstance(file, str):
        if os.path.exists(file):
            raise ValueError("File already exists!")
    file = h5py.File(file, "a")
    file.attrs["rsatoolbox_version"] = "0.0.1"
    _write_to_group(file, dictionary)


def _write_to_group(group, dictionary):
    """writes a dictionary to a hdf5 group, which can recurse"""
    if isinstance(dictionary, rsatoolbox.rdm.RDMs):
        dictionary = vars(dictionary)
    for key in dictionary.keys():
        value = dictionary[key]
        if isinstance(key, int):
            key = str(key)
        if isinstance(value, str):
            # needs another conversion to string to catch weird subtypes
            # like numpy.str_
            group.attrs[key] = str(value)
        elif isinstance(value, np.ndarray):
            if str(value.dtype)[:2] == "<U":
                group[key] = value.astype("S")
            else:
                group[key] = value
        elif isinstance(value, list):
            _write_list(group, key, value)
        elif isinstance(value, dict):
            subgroup = group.create_group(key)
            _write_to_group(subgroup, value)
        elif isinstance(value, int):
            group[key] = value
        elif value is None:
            group[key] = h5py.Empty("f")
        elif isinstance(value, Iterable):
            if isinstance(value[0], str):
                group.attrs[key] = value
        else:
            subgroup = group.create_group(key)
            _write_to_group(subgroup, value)


def _write_list(group, key, value):
    """
    writes a list to a hdf5 file. First tries conversion to np.array.
    If this fails the list is converted to a dict with integer keys.
    Parameters
    ----------
    group : hdf5 group
        where to write.
    key :  hdf5 key
    value : list
        list to be written
    """
    try:
        value = np.array(value)
        if str(value.dtype)[:2] == "<U":
            group[key] = value.astype("S")
        else:
            group[key] = value
    except TypeError:
        l_group = group.create_group(key)
        for i, v in enumerate(value):
            l_group[str(i)] = v


def write_dict_pkl(file, dictionary):
    """writes a nested dictionary containing strings & arrays as data into
    a pickle file
    Args:
        file: a filename or opened writable file
        dictionary(dict): the dict to be saved
    """
    dictionary_copy = dictionary.copy()
    if isinstance(file, str):
        file = open(file, "wb")
    dictionary_copy["rsatoolbox_version"] = "0.0.1"
    pickle.dump(dictionary_copy, file, protocol=-1)


def convert_to_legal_filename(fname):
    return re.sub("[^A-Za-z0-9]+", "_", fname)


def write_prior_probs(fname, model_prior_probs=None, layer_prior_probs=None):
    """write a model and layer prior probability file
    args:
    fname (str): path to the file to write
    model_prior_probs (dict): dictionary of model prior probabilities (model_name: prior_prob)
    layer_prior_probs (dict): dictionary of lists of layer prior probabilities (model_name: [layer_prior_probs])

    The file is saved as a JSON file.
    """

    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    combined_dict = {}

    if model_prior_probs is not None:
        combined_dict["model_prior_probs"] = model_prior_probs
    if layer_prior_probs is not None:
        combined_dict["layer_prior_probs"] = layer_prior_probs

    # dump both model and layer prior probabilities to the same file:
    with open(fname, "w") as f:
        json.dump(combined_dict, f, indent=2)
    print(f"Saved JSON to {fname}")


def _write_flat_prior_probs(fname=None, model_names=None, n_layers=None):
    """For testing purposes, write a flat prior probability file"""
    if fname is None:
        fname = "/mnt/locker/face-fmri/derivatives/model_posteriors/6_VGG_models_flat_model_prior.JSON"
    if model_names is None:
        model_names = [
            "VGG16_VGGFace2_128",
            "VGG16_BFM_identity_128",
            "VGG16_ImageNet_128",
            "VGG16_BFM_128",
            "VGG16_VGGFace2_VAE_encoder_128",
            "VGG16_BFM_VAE_encoder_128",
        ]
    if n_layers is None:
        n_layers = [16, 16, 16, 16, 16, 16]

    model_prior_probs = dict(
        zip(model_names, [1 / len(model_names)] * len(model_names))
    )
    layer_prior_probs = dict(
        zip(
            model_names,
            [[1 / n_layers[i]] * n_layers[i] for i in range(len(model_names))],
        )
    )
    write_prior_probs(
        fname, model_prior_probs=model_prior_probs, layer_prior_probs=layer_prior_probs
    )


def read_prior_probs(fname):
    """read a model prior probability file

        The format is a JSON file with a dictionary of model prior probabilities (model_name: prior_prob).
    args:
    fname (str): path to the file to read

    returns:
    model_prior_probs, layer_prior_probs

    details:
    model_prior_probs (dict):  dictionary of model prior probabilities (model_name: prior_prob)
    layer_prior_probs (dict): dictionary of lists of layer prior probabilities (model_name: [layer_prior_probs])
    """

    assert os.path.exists(fname), f"file {fname} does not exist"
    with open(fname, "r") as f:
        combined_dict = json.load(f)
    if "model_prior_probs" in combined_dict.keys():
        model_prior_probs = combined_dict["model_prior_probs"]
    else:
        model_prior_probs = None
    if "layer_prior_probs" in combined_dict.keys():
        layer_prior_probs = combined_dict["layer_prior_probs"]
    else:
        layer_prior_probs = None
    return model_prior_probs, layer_prior_probs


def print_ram_usage(idx, use_cuda=False):
    ram = psutil.virtual_memory()
    ram_percent = (ram.total - ram.available) / ram.total * 100
    swap_percent = psutil.swap_memory().percent
    print(f"RAM usage: {ram_percent:.2f}%, swap usage: {swap_percent:.2f}%, idx: {idx}")
    if torch.cuda.is_available() and use_cuda:
        vram_percent = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        print(f"GPU memory usage: {vram_percent * 100:.2f}%")


def jitter_images(images, max_jitter):
    """jitter 2D images"""
    N, C, H, W = images.shape
    device = images.device

    x_jitter = torch.randint(low=-max_jitter, high=max_jitter + 1, size=(N,)).to(device)
    y_jitter = torch.randint(low=-max_jitter, high=max_jitter + 1, size=(N,)).to(device)

    x1, y1 = max_jitter + x_jitter, max_jitter + y_jitter
    x2, y2 = W - max_jitter + x_jitter, H - max_jitter + y_jitter

    jittered = []
    for i in range(N):
        crop = images[i, :, y1[i] : y2[i], x1[i] : x2[i]].unsqueeze(0)
        crop = torch.nn.functional.interpolate(
            crop, size=(H, W), mode="bilinear", align_corners=True
        )
        jittered.append(crop)
    jittered = torch.vstack(jittered)

    return jittered
