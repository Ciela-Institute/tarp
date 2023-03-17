"""
Code to replicate the results in section 4.2: "Uninformative Estimator" of the paper
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


def generate_samples(num_samples, num_sims, num_x, mu0, sigma0, sigma, references_x_dependent=False):
    """ Generate samples and true parameter values """
    theta = np.random.normal(loc=mu0, scale=sigma0, size=num_sims)
    x = np.random.normal(loc=theta, scale=sigma, size=(num_x, num_sims)).T

    # Sample from the prior
    samples = np.random.normal(loc=mu0, scale=sigma0, size=(num_samples, num_sims))

    return samples, theta, x


def main():
    """ Main function """
    num_x = 50
    mu0 = 0
    sigma0 = 1
    sigma = 0.1
    num_samples = 1000
    num_sims = 500

    samples, theta, x = generate_samples(num_samples=num_samples, num_sims=num_sims,
                                         num_x=num_x, mu0=mu0, sigma0=sigma0, sigma=sigma)

    # Generate x dependent references
    references_x = x[:, 0] + np.random.uniform(low=-1, high=1, size=num_sims)

    # Add empty axis for the dimensions
    samples = samples[:, :, np.newaxis]
    theta = theta[:, np.newaxis]
    references_x = references_x[:, np.newaxis]

    alpha, ecp = get_drp_coverage(samples, theta, references='random', metric='euclidean')
    alpha_x, ecp_x = get_drp_coverage(samples, theta, references=references_x, metric='euclidean')

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot([0, 1], [0, 1], ls='--', color='k')
    ax.plot(alpha, ecp, label=r'DRP Random $\theta_{\rm ref}$')
    ax.plot(alpha_x, ecp_x, label=r'DRP $\theta_{\rm ref} (x)$')
    ax.legend()
    ax.set_ylabel("Expected Coverage")
    ax.set_xlabel("Credibility Level")
    plt.show()


if __name__ == '__main__':
    main()



