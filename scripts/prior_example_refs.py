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
    references = []
    references.append('random')
    references.append(0.5*np.ones(num_sims))
    references.append(0.1*x[:, 0] + np.random.uniform(low=0, high=1, size=num_sims))
    references.append(x[:, 0] + np.random.uniform(low=0, high=1, size=num_sims))
    references.append(np.mean(x, axis=1) + np.random.uniform(low=0, high=1, size=num_sims))
    references.append(x[:, 0])
    
    # Labels 
    labels = [
        r'$\theta_{\rm ref} \sim \mathcal{U}(0, 1)$',
        r'$\theta_{\rm ref} = 0.5$',
        r'$\theta_{\rm ref} \sim 0.1 x_0 + \mathcal{U}(0, 1)$',
        r'$\theta_{\rm ref} \sim x_0 + \mathcal{U}(0, 1)$',
        r'$\theta_{\rm ref} \sim \bar{x} + \mathcal{U}(0, 1)$',
        r'$\theta_{\rm ref} \sim x_0$',
    ]
    
    #linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]
    linestyles = ['-', '-', '-.', '--', '--', '--']
    

    # Add empty axis for the dimensions
    samples = samples[:, :, np.newaxis]
    theta = theta[:, np.newaxis]

    fig, ax = plt.subplots(1, 1, figsize=(9 / 2.54, 8 / 2.54))
    for ref, label, ls in zip(references, labels, linestyles):
        print(label)
        if ref is not 'random':
            ref = ref[:, np.newaxis]
        alpha, ecp = get_drp_coverage(samples, theta, references=ref, metric='euclidean')
        ax.plot(alpha, ecp, label=label, ls=ls)

    ax.plot([0, 1], [0, 1], ls='--', color='k')
    ax.legend(fontsize = 8)
    ax.set_ylabel("Expected Coverage")
    ax.set_xlabel("Credibility Level")
    plt.subplots_adjust(left = 0.15, bottom = 0.15)
    plt.savefig("/Users/pablo/Desktop/prior_references.pdf")


if __name__ == '__main__':
    main()



