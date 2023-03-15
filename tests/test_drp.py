import numpy as np

from tarp import get_drp_coverage


def test():
    num_samples = 10
    num_sims = 10
    num_dims = 5
    theta = np.random.uniform(low=-5, high=5, size=(num_sims, num_dims))
    log_sigma = np.random.uniform(low=-5, high=-1, size=(num_sims, num_dims))
    sigma = np.exp(log_sigma)
    samples = np.random.normal(
        loc=theta, scale=sigma, size=(num_samples, num_sims, num_dims)
    )
    theta = np.random.normal(loc=theta, scale=sigma, size=(num_sims, num_dims))
    get_drp_coverage(samples, theta, references="random", metric="euclidean")


if __name__ == "__main__":
    test()
