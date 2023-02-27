from typing import Any, Callable
import numpy as np

__all__ = ("get_drp_coverage",)


def get_drp_coverage(samples, theta, references='random', metric='euclidean'):
    """
    Estimates coverage with the distance to random point method.

    Reference:
        `Lemos, Coogan et al 2023 <https://arxiv.org/abs/2302.03026>`_

    Parameters:
    -----------
    samples: array-like, shape (n_samples, n_sims, n_dims)
        The samples to compute the coverage of.
    theta: array-like, shape (n_sims, n_dims)
        The true parameter values for each samples.
    references: array-like, shape (n_references, n_sims) or 'random'
        The reference points to use for the DRP regions. If 'random', then
        the reference points are chosen randomly from the parameter space.
    metric: string
        The metric to use when computing the distance. Can be 'euclidean' or
        'manhattan'.
        
    Returns:
    --------
    alpha: array-like, 
        Credibility values
    ecp: array-like,
        Expected coverage probability
    """

    # Check that shapes are correct
    assert len(samples.shape) == 3, "samples must be a 3D array"
    assert len(theta.shape) == 2, "theta must be a 2D array"

    num_samples = samples.shape[0]
    num_sims = samples.shape[1]
    num_dims = samples.shape[2]

    assert theta.shape[0] == num_sims, "theta must have the same number of rows as samples"
    assert theta.shape[1] == num_dims, "theta must have the same number of columns as samples"

    # Reshape theta
    theta = theta[np.newaxis, :, :]

    # Normalize
    low = np.min(theta, axis=1, keepdims=True)
    high = np.max(theta, axis=1, keepdims=True)
    samples = (samples - low) / (high - low + 1e-10)
    theta = (theta - low) / (high - low + 1e-10)

    # Generate reference points
    if isinstance(references, str) and references == 'random':
        references = np.random.uniform(low=0, high=1, size=(num_sims, num_dims))
    else:
        assert len(references.shape) == 2, "references must be a 2D array"
        assert references.shape[0] == num_sims, "references must have the same number of rows as samples"
        assert references.shape[1] == num_dims, "references must have the same number of columns as samples"

    # Compute distances
    if metric == 'euclidean':
        samples_distances = np.sqrt(np.sum((references[np.newaxis] - samples) ** 2, axis=-1))
        theta_distances = np.sqrt(np.sum((references - theta) ** 2, axis=-1))
    elif metric == 'manhattan':
        samples_distances = np.sum(np.abs(references[np.newaxis] - samples), axis=-1)
        theta_distances = np.sum(np.abs(references - theta), axis=-1)
    else:
        raise ValueError("metric must be either 'euclidean' or 'manhattan'")

    # Compute coverage
    f = np.sum((samples_distances < theta_distances), axis=0) / num_samples

    # Compute expected coverage
    h, alpha = np.histogram(f, density=True, bins=num_sims//10)
    dx = alpha[1] - alpha[0]
    ecp = np.cumsum(h)*dx
    return ecp, alpha[1:]

