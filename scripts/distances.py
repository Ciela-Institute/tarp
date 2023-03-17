"""
Code to replicate the results in section 4.1: "Gaussian Toy Model" of the paper
Sampling-Based Accuracy Testing of Posterior Estimators for General Inference
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from tarp import get_drp_coverage

# Set random seed
np.random.seed(0)

# latex rendering:
plt.rc("font", **{"size": 8, "family": "serif", "serif": ["Computer Modern"]})
plt.rc("text", usetex=True)


def generate_samples(num_samples, num_sims, num_dims, mode = "callibrated"):
    """Generate samples and true parameter values"""
    assert mode in ["callibrated", "overconfident", "underconfident", "biased"]
    mu = np.random.uniform(low=-5, high=5, size=(num_sims, num_dims))
    log_sigma = np.random.uniform(low=-5, high=-1, size=(num_sims, num_dims))
    sigma = np.exp(log_sigma)
    theta = np.random.normal(loc=mu, scale=sigma, size=(num_sims, num_dims))

    if mode == "callibrated":
        samples = np.random.normal(
            loc=mu, scale=sigma, size=(num_samples, num_sims, num_dims))    
    elif mode == "overconfident":
        samples = np.random.normal(
            loc=mu, scale=0.5*sigma, size=(num_samples, num_sims, num_dims))    
    elif mode == "underconfident":
        samples = np.random.normal(
            loc=mu, scale=2*sigma, size=(num_samples, num_sims, num_dims))
    elif mode == "biased":
        mu = theta + theta/abs(theta)*norm.isf((1 - abs(theta/5))/2)*sigma
        samples = np.random.normal(
            loc=mu, scale=sigma, size=(num_samples, num_sims, num_dims))    

    return samples, theta


def main():
    """Main function"""
    num_samples = 1000
    num_sims = 500
    num_dims = 1
    
    samples_cal, theta_cal = generate_samples(num_samples=num_samples, num_sims=num_sims, 
                                              num_dims=num_dims, mode="callibrated")
    samples_oc, theta_oc = generate_samples(num_samples=num_samples, num_sims=num_sims, 
                                              num_dims=num_dims, mode="overconfident")
    samples_uc, theta_uc = generate_samples(num_samples=num_samples, num_sims=num_sims, 
                                              num_dims=num_dims, mode="underconfident")
    samples_biased, theta_biased = generate_samples(num_samples=num_samples, num_sims=num_sims, 
                                              num_dims=num_dims, mode="biased")

    # Different distance metrics
    distances = [
        "euclidean",
        "manhattan",
    ]
    
    labels = [
        "Euclidean (L2)",
        "Manhattan (L1)",
    ]

    # Plot
    fig, ax = plt.subplots(1, 4, figsize=(18 / 2.54, 4 / 2.54), sharey=True)
    ax[0].plot([0, 1], [0, 1], ls="--", color="k")
    ax[1].plot([0, 1], [0, 1], ls="--", color="k")
    ax[2].plot([0, 1], [0, 1], ls="--", color="k")
    ax[3].plot([0, 1], [0, 1], ls="--", color="k")
    
    for dist, label in zip(distances, labels):
        alpha_cal, ecp_cal = get_drp_coverage(
            samples_cal, theta_cal, metric=dist)
        alpha_uc, ecp_uc = get_drp_coverage(
            samples_oc, theta_oc, metric=dist)
        alpha_oc, ecp_oc = get_drp_coverage(
            samples_uc, theta_uc, metric=dist)
        alpha_bias, ecp_bias = get_drp_coverage(
            samples_biased, theta_biased, metric=dist)
        
        ax[0].plot(alpha_cal, ecp_cal, label=label)
        ax[1].plot(alpha_oc, ecp_oc, label=label)
        ax[2].plot(alpha_uc, ecp_uc, label=label)
        ax[3].plot(alpha_bias, ecp_bias, label=label)

    plt.legend(bbox_to_anchor=(1.05, 1), handlelength=0.5, fontsize=8)
    ax[0].set_ylabel("Expected Coverage")
    ax[0].set_xlabel("Credibility Level")
    ax[1].set_xlabel("Credibility Level")
    ax[2].set_xlabel("Credibility Level")
    ax[3].set_xlabel("Credibility Level")
    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[2].set_xticklabels([])
    ax[3].set_xticklabels([])

    ax[0].title.set_text("Correct case")
    ax[1].title.set_text("Overconfident case")
    ax[2].title.set_text("Underconfident case")    
    ax[3].title.set_text("Biased case")
     
    plt.subplots_adjust(right=0.8, bottom=0.2, top=0.8)
     
    
    #plt.tight_layout()
    #plt.show()
    plt.savefig("/Users/pablo/Desktop/distances.pdf")





if __name__ == "__main__":
    main()
