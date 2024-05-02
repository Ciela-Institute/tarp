from typing import Tuple, Union
import numpy as np
from tqdm import tqdm
import deprecation


__all__ = ("get_tarp_coverage", "get_drp_coverage")


@deprecation.deprecated(
    deprecated_in="0.1.0",
    removed_in="0.2.0",
    current_version="0.1.0",
    details="Use get_tarp_coverage instead",
)
def get_drp_coverage(
    samples: np.ndarray,
    theta: np.ndarray,
    references: Union[str, np.ndarray] = "random",
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    return get_tarp_coverage(samples=samples,
                             theta=theta,
                             references=references,
                             metric=metric,
                             num_alpha_bins=None,
                             norm=True,
                             bootstrap=False,
                             seed=None)


def _get_tarp_coverage_single(
    samples: np.ndarray,
    theta: np.ndarray,
    references: Union[str, np.ndarray] = "random",
    metric: str = "euclidean",
    num_alpha_bins: Union[int, None] = None,
    norm: bool = True,
    seed: Union[int, None] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates coverage with the TARP method a single time.

    Reference: `Lemos, Coogan et al 2023 <https://arxiv.org/abs/2302.03026>`_

    Args:
        samples: the samples to compute the coverage of, with shape ``(n_samples, n_sims, n_dims)``.
        theta: the true parameter values for each samples, with shape ``(n_sims, n_dims)``.
        references: the reference points to use for the DRP regions, with shape
            ``(n_references, n_sims)``, or the string ``"random"``. If the later, then
            the reference points are chosen randomly from the unit hypercube over
            the parameter space.
        metric: the metric to use when computing the distance. Can be ``"euclidean"`` or
            ``"manhattan"``.
        norm : whether to apply or not the normalization (Default = True)
        num_alpha_bins: number of bins to use for the credibility values. If ``None``, then
            ``n_sims // 10`` bins are used.
        seed: the seed to use for the random number generator. If ``None``, then no seed

    Returns:
        Expected coverage probability (``ecp``) and credibility values (``alpha``) 
    """
    np.random.seed(seed)

    # Check that shapes are correct
    if samples.ndim != 3:
        raise ValueError("samples must be a 3D array")

    if theta.ndim != 2:
        raise ValueError("theta must be a 2D array")

    num_samples = samples.shape[0]
    num_sims = samples.shape[1]
    num_dims = samples.shape[2]

    if num_alpha_bins is None:
        num_alpha_bins = num_sims // 10

    if theta.shape[0] != num_sims:
        raise ValueError("theta must have the same number of rows as samples")

    if theta.shape[1] != num_dims:
        raise ValueError("theta must have the same number of columns as samples")

    # Reshape theta
    theta = theta[np.newaxis, :, :]

    # Normalize
    if norm:
        low = np.min(theta, axis=1, keepdims=True)
        high = np.max(theta, axis=1, keepdims=True)
        samples = (samples - low) / (high - low + 1e-10)
        theta = (theta - low) / (high - low + 1e-10)

    # Generate reference points
    if isinstance(references, str) and references == "random":
        references = np.random.uniform(low=0, high=1, size=(num_sims, num_dims))
    else:
        assert isinstance(references, np.ndarray)  # to quiet pyright
        if references.ndim != 2:
            raise ValueError("references must be a 2D array")

        if references.shape[0] != num_sims:
            raise ValueError("references must have the same number of rows as samples")

        if references.shape[1] != num_dims:
            raise ValueError(
                "references must have the same number of columns as samples"
            )

    # Compute distances
    if metric == "euclidean":
        samples_distances = np.sqrt(
            np.sum((references[np.newaxis] - samples) ** 2, axis=-1)
        )
        theta_distances = np.sqrt(np.sum((references - theta) ** 2, axis=-1))
    elif metric == "manhattan":
        samples_distances = np.sum(np.abs(references[np.newaxis] - samples), axis=-1)
        theta_distances = np.sum(np.abs(references - theta), axis=-1)
    else:
        raise ValueError("metric must be either 'euclidean' or 'manhattan'")

    # Compute coverage
    f = np.sum((samples_distances < theta_distances), axis=0) / num_samples

    # Compute expected coverage
    h, alpha = np.histogram(f, density=True, bins=num_alpha_bins, range=(0,1))
    dx = alpha[1] - alpha[0]
    ecp = np.cumsum(h) * dx
    return np.concatenate([[0], ecp]), alpha


def _get_tarp_coverage_bootstrap(samples: np.ndarray,
    theta: np.ndarray,
    references: Union[str, np.ndarray] = "random",
    metric: str = "euclidean",
    num_alpha_bins: Union[int, None] = None,
    num_bootstrap: int = 100,
    norm: bool = True,
    seed: Union[int, None] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates uncertainties on the expected probability and credibility values calculated with the
    _get_tarp_coverage_single function using the bootstrapping method

    Args:
        samples: the samples to compute the coverage of, with shape ``(n_samples, n_sims, n_dims)``.
        theta: the true parameter values for each samples, with shape ``(n_sims, n_dims)``.
        references: the reference points to use for the DRP regions, with shape
            ``(n_references, n_sims)``, or the string ``"random"``. If the later, then
            the reference points are chosen randomly from the unit hypercube over
            the parameter space.
        metric: the metric to use when computing the distance. Can be ``"euclidean"`` or
            ``"manhattan"``.
        num_alpha_bins: number of bins to use for the credibility values. If ``None``, then
            ``n_sims // 10`` bins are used.
        num_bootstrap: number of bootstrap iterations to perform (Default = 100)
        norm : whether to apply or not the normalization (Default = True)
        seed: the seed to use for the random number generator. If ``None``, then no seed

    Returns:
        Expected coverage probability, and credibility
    """
    num_sims = samples.shape[1]

    if num_alpha_bins is None:
        num_alpha_bins = num_sims // 10

    boot_ecp = np.empty(shape=(num_bootstrap, num_alpha_bins+1))
    for i in tqdm(range(num_bootstrap)):
        idx = np.random.randint(low=0, high=num_sims, size=num_sims)
        
        # Sample with replacement from the full set of simulations
        boot_samples = samples[:, idx, :]
        boot_theta = theta[idx, :]

        boot_ecp[i, :], alpha = _get_tarp_coverage_single(boot_samples,
                                                          boot_theta,
                                                          references=references,
                                                          metric=metric,
                                                          num_alpha_bins=num_alpha_bins,
                                                          norm=norm,
                                                          seed=seed)
    
    # ecp_mean = boot_ecp.mean(axis=0)
    # ecp_std = boot_ecp.std(axis=0)
    # alpha_mean = boot_alpha.mean(axis=0)
    # return ecp_mean, ecp_std, alpha_mean
    return boot_ecp, alpha


def get_tarp_coverage(
    samples: np.ndarray,
    theta: np.ndarray,
    references: Union[str, np.ndarray] = "random",
    metric: str = "euclidean",
    num_alpha_bins: Union[int, None] = None,
    num_bootstrap: int = 100,
    norm: bool = False,
    bootstrap: bool = False,
    seed: Union[int, None] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates coverage with the TARP method.

    Reference: `Lemos, Coogan et al 2023 <https://arxiv.org/abs/2302.03026>`_

    Args:
        samples: the samples to compute the coverage of, with shape ``(n_samples, n_sims, n_dims)``.
        theta: the true parameter values for each samples, with shape ``(n_sims, n_dims)``.
        references: the reference points to use for the DRP regions, with shape
            ``(n_references, n_sims)``, or the string ``"random"``. If the later, then
            the reference points are chosen randomly from the unit hypercube over
            the parameter space.
        metric: the metric to use when computing the distance. Can be ``"euclidean"`` or
            ``"manhattan"``.
        num_alpha_bins: number of bins to use for the credibility values. If ``None``, then
            ``n_sims // 10`` bins are used.
        num_bootstrap: number of bootstrap iterations to perform (Default = 100)
        norm : whether to apply or not the normalization (Default = False)
        bootstrap : whether to use bootstrap to estimate uncertainties (Default = False)
        seed: the seed to use for the random number generator. If ``None``, then no seed

    Returns:
        Expected coverage probability (``ecp``) and credibility values (``alpha``).
        If bootstrap is True, the ecp array has an extra dimension corresponding to the number of bootstrap iterations
    """
    if bootstrap:
        ecp, alpha = _get_tarp_coverage_bootstrap(samples, theta, references, metric, num_alpha_bins, num_bootstrap,
                                                  norm, seed)
    else:
        ecp, alpha = _get_tarp_coverage_single(samples, theta, references, metric, num_alpha_bins, norm, seed)
    return ecp, alpha

