import torch
import math
import pandas as pd

from utils import (
    numerically_stable_cosine_similarity,
    pearson_correlation,
    whitened_unbiased_cosine_similarity,
    whitened_pearson_correlation,
)


class SmoothMax(torch.nn.Module):
    """
    Smoothmax that interpolates between a max and a mean.
    alpha is the interpolation parameter (0 is mean, inf is max)
    see https://en.wikipedia.org/wiki/Smooth_maximum#Examples

    Note that this particular implementation is prone to overflow.
    if we'd like to use this for very large inputs,
    we can do something like https://math.stackexchange.com/a/2552979
    """

    def __init__(self, alpha=1.0):
        """
        args:
        alpha (float) - the interpolation parameter (0 is mean, inf is max)
        """
        super(SmoothMax, self).__init__()
        self.alpha = alpha

    def forward(self, x, dim=None, keepdim=False):
        # to avoid overflow, we scale all of the weights so the biggest one is 1
        if dim is None:
            max_x = torch.max(x)
        else:
            max_x = torch.max(x, dim=dim, keepdim=True)[0]
        weights = torch.exp(self.alpha * (x - max_x))
        return torch.sum(weights * x, dim=dim, keepdim=keepdim) / torch.sum(
            weights, dim=dim, keepdim=keepdim
        )


class LogSumExp(torch.nn.Module):
    """https://en.wikipedia.org/wiki/LogSumExp
    alpha -> infinity: maximum
    alpha -> -infinity: minimum
    """

    def __init__(self, alpha=1.0):
        super(LogSumExp, self).__init__()
        self.alpha = alpha

    def forward(self, x, dim=None):
        return torch.logsumexp(x * self.alpha, dim=dim) / self.alpha


class MellowMax(torch.nn.Module):
    """https://en.wikipedia.org/wiki/Smooth_maximum#Examples
    avoid overflow with large x.
    alpha -> infinity: maximum
    alpha -> 0: arithmetic mean
    alpha -> -infinity: minimum
    """

    def __init__(self, alpha=1.0):
        super(MellowMax, self).__init__()
        self.alpha = alpha

    def forward(self, x, dim=None):
        n = x.shape[dim]
        return (torch.logsumexp(x * self.alpha, dim=dim) - math.log(n)) / self.alpha


class CachedCorrelationCalc:
    """A module that does cached correlation matrix calculation.
    Depending on the utility function, certain correlation matrices may be evaluated multiple times
    and other may never be evaluated.  This module calculates each required correlation matrix only once.
    """

    def __init__(
        self,
        dissimilarities,
        calc_correlation_fun,
        pairs_per_correlation=None,
        device=None,
    ):
        """
        args:
        dissimilarities (dict) a dictionary of dictionaries of tensors.
        calc_correlation_fun (function) - a function that takes a list of dissimilarities and returns a correlation matrix
        pairs_per_correlation (int) if not None, the number of pairs to use for each correlation coefficient
        """

        self.calc_correlation_fun = calc_correlation_fun
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.dissimilarities = dissimilarities
        self.correlations = dict()
        self.pairs_per_correlation = pairs_per_correlation

    def calc_correlation(
        self, model1_name, model1_instance, model2_name, model2_instance
    ):
        key = (model1_name, model1_instance, model2_name, model2_instance)
        transposed_key = (model2_name, model2_instance, model1_name, model1_instance)
        """
        calculate the correlation between two models if it hasn't been calculated yet

        args:
        dissimilarities (dict) a dictionary of dictionaries of tensors.
        The outer key is the model name. The inner key is the realization index.
        The value is a tensor of shape (n_layers, n_pairs) or just (n_pairs)

        model1_name (str) - the name of the first model
        model1_instance (int) - the realization index of the first model
        model2_name (str) - the name of the second model
        model2_instance (int) - the realization index of the second model

        returns a tensor of shape (n_layers_model1, n_layers_model2), and caches it for later use.
        """
        if key in self.correlations:
            return self.correlations[key]
        elif transposed_key in self.correlations:
            return self.correlations[transposed_key].T
        else:
            model1_dissimilarities = self.dissimilarities[model1_name][
                model1_instance
            ].to(self.device)
            model2_dissimilarities = self.dissimilarities[model2_name][
                model2_instance
            ].to(self.device)

            if self.pairs_per_correlation is None:
                # the simple case - calculate a correlation coefficient across all pairs
                self.correlations[key] = self.calc_correlation_fun(
                    model1_dissimilarities,
                    model2_dissimilarities,
                )
            else:
                # calculate a correlation coefficient within each trial
                # model_RDMs_1 (torch.tensor) m1_representations (n_layers) x n_independent_pairs
                # model_RDMs_2 (torch.tensor) m2_representations (n_layers) x n_independent_pairs
                trial_level_correlations = []
                n_pairs = model1_dissimilarities.shape[-1]
                div_round_up = lambda x, y: x // y + (x % y != 0)
                n_trials = div_round_up(n_pairs, self.pairs_per_correlation)
                for i_trial in range(n_trials):
                    cur_trial_pair_indices = torch.arange(
                        i_trial * self.pairs_per_correlation,
                        min((i_trial + 1) * self.pairs_per_correlation, n_pairs),
                        device=self.device,
                    )
                    trial_level_correlations.append(
                        self.calc_correlation_fun(
                            model1_dissimilarities[..., cur_trial_pair_indices],
                            model2_dissimilarities[..., cur_trial_pair_indices],
                        )
                    )
                self.correlations[key] = torch.stack(
                    trial_level_correlations, axis=0
                ).mean(dim=0)
            return self.correlations[key]

    def __getitem__(self, args):
        return self.calc_correlation(*args)


class MultiInstance_Multilayer_Raw_Correlation_Utility(torch.nn.Module):
    def __init__(
        self,
        cfg,
        device=None,
        pairs_per_correlation=None,
        model_prior_probs=None,
        layer_prior_probs=None,
    ):
        """Superclass for maximize model discrimination between distances, taking into account multiple model layers and realizations.

        Assuming that the "correct" model is among the candidate models but we don't know which model is the correct model,
        this utility function approximates our ability to retreive the correct model.

        We follow a semi-Bayesian optimal experimental design approach. We calculate a global utility U(X) where X is the stimulus set
        by averaging a local utility u over all possible data-generating models, layers, and realizations (assuming flat priors).

        Approach 1:
        u(X,m,l,r) approximates our ability to correctly identify the data-generating model m, layer l, and realization r given stimulus set X using our frequentist model discrimination approach
        (selecting the model with the highest correlation with the human data, taking the maximum across layers and average across subjects).

        u(X,m,l,r) =  max_{l'} corr(d(X,m,l,r), d(X, m,l',r0))
                    - mean_{m' in M-m} max_{l'} corr(d(X,m,l,r), d(X,m',l',r0))

        where D(X,m,l,r) is the vector or RDM of representational distances according to the reference model m, layer l, and realization r,
        given stimulus set X. r0 is the reference realization (which is not used for data generation).

        Approach 2:

        args:
        dist_fun (str) 'cosine_similarity'/'pearson_correlation'/'whitened_cosine_similarity'/'whitened_pearson_correlation'
        utility_definition (str) particular definition of utility to use
        accuracy_definition (str) particular definition of accuracy to use
        model_comparison_matrix (currently not implemented)
        models_agg_fun (str) 'max', 'mean', or 'smoothmax'
        models_agg_fun_alpha (float) alpha parameter for models_agg_fun
        layers_agg_fun (str) 'max', 'mean', or 'smoothmax'
        layers_agg_fun_alpha (float) alpha parameter for layers_agg_fun
        fisher_transform (bool) whether to apply Fisher transform to the utility
        reference_realization_id (int) if single_reference_utility is used, which instance_id is used as reference
        device (torch.device) where to store the calculated correlations and utility
        exp_diminishing_returns (float) if not None, the local utility is scaled by exp(-exp_diminishing_returns * local utility)
        pairs_per_correlation (int) if not None, the number of pairs to use for each correlation coefficient calculation
        model_prior_probs (dict) a dictionary of log probabilities for each model (model_name -> probability)
        layer_prior_probs (dict) a dictionary of lists of log probabilities for each layer (model_name -> list of probabilities for each layer)

        """

        super().__init__()
        self.n_conditions = cfg.nFaces
        self.dist_fun = cfg.rsa_dist_fun
        self.utility_definition = cfg.utility_definition
        self.accuracy_definition = getattr(cfg, "accuracy_definition", None)

        models_agg_fun = cfg.models_agg_fun
        models_agg_fun_alpha = getattr(cfg, "models_agg_fun_alpha", None)
        layers_agg_fun = cfg.layers_agg_fun
        layers_agg_fun_alpha = getattr(cfg, "layers_agg_fun_alpha", None)

        if models_agg_fun == "mean":
            self.models_agg_fun = torch.mean
        elif models_agg_fun == "max":
            self.models_agg_fun = lambda *args, **kwargs: torch.max(*args, **kwargs)[
                0
            ]  # return the max value, get rid of the index
        elif models_agg_fun == "smoothmax":
            self.models_agg_fun = SmoothMax(models_agg_fun_alpha)

        if layers_agg_fun == "mean":
            self.layers_agg_fun = torch.mean
        elif layers_agg_fun == "max":
            self.layers_agg_fun = lambda *args, **kwargs: torch.max(*args, **kwargs)[0]
        elif layers_agg_fun == "smoothmax":
            self.layers_agg_fun = SmoothMax(layers_agg_fun_alpha)

        model_comparison_matrix = getattr(cfg, "model_comparison_matrix", None)
        if model_comparison_matrix is not None:
            raise NotImplementedError

        self.utility_fun = getattr(self, self.utility_definition)
        self.accuracy_fun = (
            getattr(self, self.accuracy_definition)
            if self.accuracy_definition is not None
            else None
        )
        self.fisher_transform = cfg.fisher_transform
        self.reference_realization_id = getattr(cfg, "reference_realization_id", None)
        self.exp_diminishing_returns = getattr(cfg, "exp_diminishing_returns", None)
        self.pairs_per_correlation = pairs_per_correlation

        self.model_prior_probs = model_prior_probs
        self.layer_prior_probs = layer_prior_probs

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

    def single_reference_utility(self, correlations, models, realizations):
        """Calculate the utility of a stimulus set for a single reference model.

        We use the first realization as the reference realization
        and the rest as the data generating representations.

        args:
        correlations (CachedCorrelationCalc)
        models (list) a list of model names
        realizations (dict) a dictionary of lists of realization indices/names

        returns a differentiable global utility (torch.Tensor)
        """

        # define data generating realizations
        n_models = len(models)
        n_realizations = {model: len(realizations[model]) for model in models}
        assert (
            self.reference_realization_id is not None
        ), "reference_realization_id must be set"
        data_generating_realizations = {}
        for model in models:
            assert self.reference_realization_id in realizations[model]
            if n_realizations[model] > 1:
                data_generating_realizations[model] = [
                    r for r in realizations[model] if r != self.reference_realization_id
                ]
            else:  # if there is only one realization, we use it as the data generating realization
                data_generating_realizations[model] = [realizations[model][0]]

        U = 0.0  # global utility

        # TODO: add support for non-flat model-priors and layer-priors?
        for data_generating_model in models:
            for data_generating_realization in data_generating_realizations[model]:
                incorrect_correlations = []
                for reference_model in models:
                    C = correlations[
                        (
                            data_generating_model,
                            data_generating_realization,
                            reference_model,
                            self.reference_realization_id,
                        )
                    ]
                    n_data_generating_layers, n_reference_layers = C.shape
                    # print(C.shape)
                    # take a maximum over the layers of the reference model
                    cur_max_corr = self.layers_agg_fun(
                        C, dim=1
                    )  # reduce over the reference model layers
                    assert cur_max_corr.shape == (n_data_generating_layers,)
                    # (each element is the best reference correlation for each data-generating layer)

                    if data_generating_model == reference_model:
                        # if the reference model is the data generating model,
                        # the maximual correlation contributes positively to the local utility
                        correct_correlations = cur_max_corr
                    else:
                        # if the reference model is not the data generating model,
                        # the maximal correlation contributes negatively to the local utility
                        incorrect_correlations.append(cur_max_corr)

                # aggerate correlations from different incorrect reference models (e.g., by averaging or taking the maximum)
                incorrect_correlations = self.models_agg_fun(
                    torch.stack(incorrect_correlations, axis=0), dim=0
                )  # dimensions: (n_data_generating_layers,)
                assert incorrect_correlations.shape == (n_data_generating_layers,)

                local_utility = correct_correlations - incorrect_correlations

                if self.exp_diminishing_returns is not None:
                    local_utility = -torch.exp(
                        -local_utility * self.exp_diminishing_returns
                    )

                # combine local utility across data generating layers
                if self.layer_prior_probs is not None:  # a layer prior was specified
                    cur_layer_prior_probs = torch.tensor(
                        self.layer_prior_probs[data_generating_model],
                        device=local_utility.device,
                        dtype=local_utility.dtype,
                    )
                    assert (
                        len(cur_layer_prior_probs) == n_data_generating_layers
                    ), f"layer_prior_probs for {data_generating_model} must have length {n_data_generating_layers}, but has length {len(cur_layer_prior_probs)}"
                    local_utility = torch.dot(cur_layer_prior_probs, local_utility)
                else:  # no layer prior was specified
                    local_utility = local_utility.mean()

                # currently we don't have non-flat priors over realizations
                p_realization = 1.0 / len(data_generating_realizations[model])

                if self.model_prior_probs is not None:
                    p_model = self.model_prior_probs[data_generating_model]
                else:
                    p_model = 1 / n_models
                U += (
                    p_model * p_realization * local_utility
                )  # add local utility time prior to global utility
        return U

    def single_reference_utility_xent(self, correlations, models, realizations):
        """Calculate the utility of a stimulus set for a single reference model.

        We use the first realization as the reference realization
        and the rest as the data generating representations.

        args:
        correlations (CachedCorrelationCalc)
        models (list) a list of model names
        realizations (dict) a dictionary of lists of realization indices/names

        returns a differentiable global utility (torch.Tensor)
        """

        # define data generating realizations
        n_models = len(models)
        n_realizations = {model: len(realizations[model]) for model in models}
        assert (
            self.reference_realization_id is not None
        ), "reference_realization_id must be set"
        data_generating_realizations = {}
        for model in models:
            assert self.reference_realization_id in realizations[model]
            if n_realizations[model] > 1:
                data_generating_realizations[model] = [
                    r for r in realizations[model] if r != self.reference_realization_id
                ]
            else:  # if there is only one realization, we use it as the data generating realization
                data_generating_realizations[model] = [realizations[model][0]]

        U = 0.0  # global utility

        # TODO: add support for non-flat model-priors and layer-priors?
        for i_data_generating_model, data_generating_model in enumerate(models):
            for data_generating_realization in data_generating_realizations[model]:
                cur_ref_correlations = []
                for reference_model in models:
                    C = correlations[
                        (
                            data_generating_model,
                            data_generating_realization,
                            reference_model,
                            self.reference_realization_id,
                        )
                    ]
                    n_data_generating_layers, n_reference_layers = C.shape

                    # take a maximum over the layers of the reference model
                    cur_max_corr = self.layers_agg_fun(
                        C, dim=1
                    )  # reduce over the reference model layers
                    assert cur_max_corr.shape == (n_data_generating_layers,)
                    # (each element is the best reference correlation for each data-generating layer)

                    cur_ref_correlations.append(cur_max_corr)

                cur_ref_correlations = torch.stack(cur_ref_correlations, axis=1)
                assert cur_ref_correlations.shape == (
                    n_data_generating_layers,
                    n_models,
                )

                labels = torch.tensor(
                    [i_data_generating_model] * n_data_generating_layers,
                    device=cur_ref_correlations.device,
                )
                local_utility = -torch.nn.CrossEntropyLoss(reduction="none")(
                    cur_ref_correlations, labels
                )

                # combine local utility across data generating layers
                # here we are just averaging over the data generating layers,
                # but we could also apply a prior over the data generating layers
                assert (
                    self.layer_prior_probs is None
                ), "layer_prior_probs not supported for xent utility"
                local_utility = local_utility.mean()
                # the same apply here  (we could have used a prior over models)
                p_realization = 1.0 / len(data_generating_realizations[model])
                assert (
                    self.model_prior_probs is None
                ), "model_prior_probs not supported for xent utility"
                p_model = 1 / n_models
                U += (
                    p_model * p_realization * local_utility
                )  # add local utility time prior to global utility
        return U

    def single_reference_accuracy(self, correlations, models, realizations):
        """calculate a non-differentiable model-retrieval accuracy estimate from between-model dissimilarty correlations
        This is the non-differentiable (but more interpretable) version of single_reference_utility()

        args:
        correlations (CachedCorrelationCalc)
        models (list) a list of model names
        realizations (dict) a dictionary of lists of realization indices/names

        returns a differentiable global utility (torch.Tensor)
        """

        # define data generating realizations
        n_models = len(models)
        n_realizations = {model: len(realizations[model]) for model in models}
        assert (
            self.reference_realization_id is not None
        ), "reference_realization_id must be set"

        if not all([n > 1 for n in n_realizations.values()]):
            return None  # we cannot calculate an accuracy estimate for a single realization

        numerator = 0
        denominator = 0

        with torch.no_grad():
            data_generating_realizations = {}
            for model in models:
                assert self.reference_realization_id in realizations[model]
                if n_realizations[model] > 1:
                    data_generating_realizations[model] = [
                        r
                        for r in realizations[model]
                        if r != self.reference_realization_id
                    ]
                else:  # if there is only one realization, we use it as the data generating realization
                    data_generating_realizations[model] = [realizations[model][0]]

            # TODO: add support for non-flat model-priors and layer-priors?
            for data_generating_model in models:
                for data_generating_realization in data_generating_realizations[model]:
                    incorrect_correlations = []
                    for reference_model in models:
                        C = correlations[
                            (
                                data_generating_model,
                                data_generating_realization,
                                reference_model,
                                self.reference_realization_id,
                            )
                        ]
                        n_data_generating_layers, n_reference_layers = C.shape

                        # take a maximum over the layers of the reference model
                        cur_max_corr = torch.max(C, dim=1)[
                            0
                        ]  # reduce over the reference model layers
                        assert cur_max_corr.shape == (n_data_generating_layers,)
                        # (each element is the best reference correlation for each data-generating layer)

                        if data_generating_model == reference_model:
                            correct_correlation = cur_max_corr
                        else:
                            incorrect_correlations.append(cur_max_corr)

                    incorrect_correlations = torch.stack(
                        incorrect_correlations, dim=0
                    )  # the resulting tensor has shape (n_models-1, n_data_generating_layers)
                    assert incorrect_correlations.shape == (
                        n_models - 1,
                        n_data_generating_layers,
                    )

                    incorrect_correlations = torch.max(incorrect_correlations, dim=0)[
                        0
                    ]  # take the highest correlation across "wrong" reference models.
                    # the resulting tensor has shape (n_data_generating_layers)
                    assert incorrect_correlations.shape == (n_data_generating_layers,)

                    correct_per_layer = (
                        correct_correlation > incorrect_correlations
                    ).float()

                    if self.layer_prior_probs is not None:
                        w = torch.tensor(
                            self.layer_prior_probs[data_generating_model],
                            device=correct_per_layer.device,
                            dtype=torch.float32,
                        )
                    else:
                        w = (
                            torch.ones(
                                n_data_generating_layers,
                                device=correct_per_layer.device,
                                dtype=torch.float32,
                            )
                            / n_data_generating_layers
                        )
                    if self.model_prior_probs is not None:
                        w = w * self.model_prior_probs[data_generating_model]
                    else:
                        w = w * 1 / n_models

                    numerator += torch.dot(correct_per_layer, w)
                    denominator += w.sum()

        return numerator / denominator

    def forward(self, dissimilarities):
        """Differentiable utility for multilayer representational models

        args:
        dissimilarities (dict) a dictionary of dictionaries of tensors.
        The outer key is the model name. The inner key is the realization index.
        The value is a tensor of shape (n_layers, n_pairs) or just (n_pairs)

        returns:
        loss - a differntiable loss (torch.Tensor) the reflects our abilities to retrieve the right model by its dissimilarities
        accuracy - a non-differentiable accuracy estimate
        """

        # step one: initialize CachedCorrelationCalc object
        correlations = CachedCorrelationCalc(
            dissimilarities=dissimilarities,
            calc_correlation_fun=self.calc_correlation_matrix,
            device="cpu",
            pairs_per_correlation=self.pairs_per_correlation,
        )  # n_data_generating_layers, n_reference_layers

        models = list(dissimilarities.keys())
        realizations = {model: list(dissimilarities[model].keys()) for model in models}
        del dissimilarities  # but it is still linked from within the CachedCorrelationCalc object

        # step two: calculate the utility
        utility = self.utility_fun(correlations, models, realizations)
        loss = -utility

        if self.accuracy_fun is not None:
            with torch.no_grad():
                accuracy = self.accuracy_fun(correlations, models, realizations)
        else:
            accuracy = None
        return loss, accuracy

    def calc_correlation_matrix(self, model_RDMs_1, model_RDMs_2):
        """Calculate correlation cofficients between data RDMs y and model RDMs.

        model_RDMs_1 (torch.tensor): n_layers x n_pairs (independent or all pairs)
        model_RDMs_2 (torch.tensor): n_layers x n_pairs (independent or all pairs)

        returns a matrix of correlations r [m1_representations x m2_representations]
        """

        m1_representations = model_RDMs_1.shape[0]
        m2_representations = model_RDMs_2.shape[0]
        assert model_RDMs_1.shape[1] == model_RDMs_2.shape[1]

        model_RDMs_2 = model_RDMs_2.to(model_RDMs_1.device)
        if self.dist_fun in [
            "quick_whitened_pearson_correlation",
            "quick_whitened_cosine_similarity",
        ]:
            # these functions operate on n_rdms x n_distances matrices
            if self.dist_fun == "quick_whitened_pearson_correlation":
                r = self.corr_cov_obj.whitened_pearson_correlation(
                    model_RDMs_1, model_RDMs_2, eps=0.0
                )
            elif self.dist_fun == "quick_whitened_cosine_similarity":
                r = self.corr_cov_obj.whitened_cosine_similarity(
                    model_RDMs_1, model_RDMs_2, eps=0.0
                )
        else:
            model_RDMs_1 = model_RDMs_1.unsqueeze(
                dim=1
            )  # make model_RDMs.shape = m_models x 1 x n_distances
            model_RDMs_2 = model_RDMs_2.unsqueeze(
                dim=0
            )  # make y.shape = 1 x n_samples x n_distances
            if self.dist_fun == "cosine_similarity":
                r = numerically_stable_cosine_similarity(
                    model_RDMs_1, model_RDMs_2, dim=2
                )  # r.shape = (m1_representations x m2_representations)
            elif self.dist_fun == "pearson_correlation":
                r = pearson_correlation(
                    model_RDMs_1, model_RDMs_2, dim=2
                )  # r.shape = (m1_representations x m2_representations)
            elif self.dist_fun == "whitened_cosine_similarity":
                r = whitened_unbiased_cosine_similarity(  # r.shape = (m_models x n_models)
                    model_RDMs_1, model_RDMs_2, dim=2, inv_V=self.inv_V, keepdim=False
                )
            elif self.dist_fun == "whitened_pearson_correlation":
                r = whitened_pearson_correlation(  # r.shape = (m_models x n_models)
                    model_RDMs_1,
                    model_RDMs_2,
                    dim=2,
                    inv_V=self.inv_V,
                    keepdim=False,
                    eps=0.0,
                )
            else:
                raise NotImplementedError
        assert r.shape == (
            m1_representations,
            m2_representations,
        )  # m1_layer x m2_layer

        # assert not r.isnan().any().item()
        if self.fisher_transform:
            r = torch.atanh(torch.clamp(r, min=-1 + 1e-6, max=1 - 1e-6))
        # assert not r.isnan().any().item()
        return r  # CachedCorrelationCalc

    def loss_terms_to_dataframe(self, dissimilarities):
        # for posthoc analysis purposes, return a Pandas DataFrame with each correlation coefficient between every pair of models and layers

        with torch.inference_mode():
            correlations = CachedCorrelationCalc(
                dissimilarities=dissimilarities,
                calc_correlation_fun=self.calc_correlation_matrix,
                device="cpu",
                pairs_per_correlation=self.pairs_per_correlation,
            )

            models = list(dissimilarities.keys())
            realizations = {
                model: list(dissimilarities[model].keys()) for model in models
            }
            del dissimilarities  # but it is still linked from within the CachedCorrelationCalc object

            results = []
            for model1_name in models:
                for model1_instance in realizations[model1_name]:
                    for model2_name in models:
                        for model2_instance in realizations[model2_name]:
                            cur_corr = correlations[
                                (
                                    model1_name,
                                    model1_instance,
                                    model2_name,
                                    model2_instance,
                                )
                            ]  # n_layers_model_1 x n_layers_model_2
                            for i_layer_model_1 in range(cur_corr.shape[0]):
                                for i_layer_model_2 in range(cur_corr.shape[1]):
                                    results.append(
                                        {
                                            "model1_name": model1_name,
                                            "model1_instance": model1_instance,
                                            "model2_name": model2_name,
                                            "model2_instance": model2_instance,
                                            "layer_model_1": i_layer_model_1,
                                            "layer_model_2": i_layer_model_2,
                                            "correlation": cur_corr[
                                                i_layer_model_1, i_layer_model_2
                                            ].item(),
                                        }
                                    )
            return pd.DataFrame(results)


class Pairwise_MultiInstance_Multilayer_Raw_Correlation_Utility(
    MultiInstance_Multilayer_Raw_Correlation_Utility
):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.domain = "pairwise_dissimilarities"
