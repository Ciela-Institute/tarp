from typing import Tuple, Union

import numpy as np
from scipy.stats import chi2_contingency

__all__ = ("get_tarp_coverage", "get_drp_coverage")


def get_tarppp_chi2(x_samples, y_samples, num_refs=100):
    """
    Compute the chi2 statistic for the TARP++ test.

    Parameters
    ----------
    x_samples : np.ndarray
        Samples from the posterior.
    y_samples : np.ndarray
        True samples
    num_refs : int
        Number of reference samples to use.

    Returns
    -------
    float
        Chi2 statistic.
    """
    refs = np.random.choice(len(y_samples), num_refs, replace=False)
    refs = y_samples[refs]

    counts_x = np.zeros(num_refs, dtype="int")
    counts_y = np.zeros(num_refs, dtype="int")
    for x in x_samples:
        d = np.linalg.norm(x.reshape(1, -1) - refs, axis=-1)
        idx = np.argmin(d)
        counts_x[idx] += 1

    for y in y_samples:
        d = np.linalg.norm(y.reshape(1, -1) - refs, axis=-1)
        idx = np.argmin(d)
        counts_y[idx] += 1

    chi2_stat, _, _, _ = chi2_contingency(np.array([counts_x, counts_y]))
    return chi2_stat
