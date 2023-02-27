"""
Code to replicate the results in section 4.1: "Gaussian Toy Model" of the paper
Sampling-Based Accuracy Testing of Posterior Estimators for General Inference
"""

import numpy as np
import matplotlib.pyplot as plt
from tarp import get_drp_coverage

# Set random seed
np.random.seed(0)

# latex rendering:
plt.rc('font', **{'size': 10, 'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)


def generate_samples(num_samples, num_sims, num_dims):
    """ Generate samples and true parameter values """
    theta = np.random.uniform(low=-5, high=5, size=(num_sims, num_dims))
    log_sigma = np.random.uniform(low=-5, high=-1, size=(num_sims, num_dims))
    sigma = np.exp(log_sigma)
    samples = np.random.normal(loc=theta, scale=sigma, size=(num_samples, num_sims, num_dims))
    theta = np.random.normal(loc=theta, scale=sigma, size=(num_sims, num_dims))
    return samples, theta


def main():
    """ Main function """
    samples, theta = generate_samples(num_samples=1000, num_sims=100, num_dims=10)
    alpha, ecp = get_drp_coverage(samples, theta, references='random', metric='euclidean')

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot([0, 1], [0, 1], ls='--', color='k')
    ax.plot(alpha, ecp, label='DRP')
    ax.legend()
    ax.set_ylabel("Expected Coverage")
    ax.set_xlabel("Credibility Level")
    plt.show()


if __name__ == '__main__':
    main()



