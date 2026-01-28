#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.optimize as opt 

np.random.seed(1998)

# --- the data --- #
num_datasets = 5
t_obs = np.linspace(0.0, 10.0, 15)
z_obs = [
    pd.read_csv('physics_informed_SR/Synthetic_Isomerisation/exp_data/exp_' + str(i + 1) + '.csv', header=None).to_numpy()
    for i in range(num_datasets)
]
initial_conditions = [
    np.array([2 , 0]),
    np.array([10, 0]),
    np.array([2 , 2]),
    np.array([10, 2]),
    np.array([10, 1]),
]

# --- the dynamic model --- #
def kinetic_model(t, state, *parameters):
    # Unpack state variables
    CA, CB = state
    # Unpack parameters
    k_1, k_2, k_3, k_4, k_5 = parameters
    # Define the rate equations
    common_term = ((k_1 * CA - k_2 * CB) / (k_3 * CA + k_4 * CB + k_5))
    dAdt = -common_term
    dBdt = common_term
    return np.array([dAdt, dBdt])

# --- the loss function (sum of squared errors) --- #
def sse(parameters):
    total_error = 0.0
    timesteps = 15
    time = np.linspace(0, 10, timesteps)
    t_span = [0, np.max(time)]
    for i in range(num_datasets):
        ic = initial_conditions[i]
        # Pass parameters as a tuple so that kinetic_model can unpack them
        sol = solve_ivp(kinetic_model, t_span, ic, t_eval=time, method="RK45", args=tuple(parameters))
        total_error += np.sum((sol.y - z_obs[i])**2)
    return total_error

# --- Define Log-Prior and Log-Likelihood ---
prior_means = np.array([5, 5, 5, 5, 5])
prior_covs = np.diag(np.array([5, 5, 5, 5, 5]))

def log_prior(parameters):
    return multivariate_normal.logpdf(parameters, mean=prior_means, cov=prior_covs)

def log_likelihood(parameters):
    return -sse(parameters) / 2.0

# --- Define the Objective Function (Negative Log-Posterior) ---
def neg_log_posterior(parameters):
    log_p = log_prior(parameters)
    log_l = log_likelihood(parameters)
    return -(log_p + log_l)

# --- Bayesian Inference using Laplace Approximation --- #
def laplace_approximation(initial_parameters, num_samples):
    """
    Performs Laplace Approximation to find the posterior distribution.
    
    1. Finds the MAP estimate (mode) by minimizing the negative log-posterior.
    2. Estimates the covariance from the inverse Hessian at the mode.
    3. Draws samples from the resulting multivariate Gaussian.
    """
    
    # --- Find the Posterior Mode (MAP Estimate) ---
    bounds = [(0, None) for _ in initial_parameters]  # All parameters must be >= 0
    map_result = opt.minimize(neg_log_posterior, 
                              initial_parameters, 
                              method='L-BFGS-B', 
                              bounds=bounds)
    
    mu_map = map_result.x
    print(f"MAP estimate: {np.round(mu_map, 3)}")

    # --- Estimate Covariance from Inverse Hessian ---
    # Use the BFGS method at mode to get the inverse Hessian. L-BFGS-B does not provide Hessian.
    cov_result = opt.minimize(neg_log_posterior, 
                              mu_map, 
                              method='BFGS')
    
    if not cov_result.success:
        print("Warning: Covariance estimation did not converge.")

    covariance = cov_result.hess_inv

    # --- Draw Samples from the Gaussian Approximation ---
    print(f"Drawing {num_samples} samples from the Gaussian approximation...")
    samples = np.random.multivariate_normal(mu_map, covariance, size=num_samples)
    # samples = np.maximum(samples, 0)
    
    return samples

# --- Example usage --- #
initial_parameters = np.array([7, 4, 3, 2, 6]) + np.random.normal(0, 1, 5)
num_samples = 10000 

samples = laplace_approximation(initial_parameters, num_samples)

# --- Plot the posterior distributions of the parameters --- #
num_params = samples.shape[1]
colors = ['royalblue', 'salmon', 'limegreen', 'darkviolet', 'goldenrod']

fig, axes = plt.subplots(nrows=2, ncols=3,
                         figsize=(18, 12),
                         constrained_layout=True)

axes = axes.ravel()

for i in range(num_params):
    ax = axes[i]
    ax.hist(samples[:, i],
            bins=50, density=True, alpha=0.7,
            color=colors[i % len(colors)])
    ax.set_title(f'Parameter {i + 1}', fontsize=28)
    ax.set_xlabel('Value', fontsize=28)
    if i % 3 == 0:
        ax.set_ylabel('Density', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.grid(True, linestyle='--', alpha=0.5)

if num_params < len(axes):
    axes[-1].set_visible(False)

file_path = ('Synthetic_Isomerisation_no_constraints/'
             'graphs/parameter_uncertainty_LAPLACE.png') # Changed name
fig.savefig(file_path, dpi=600, bbox_inches='tight')
plt.show()


# --- Prediction and Uncertainty Plot for All Experiments ---

color_1 = ['salmon', 'royalblue', 'darkviolet', 'limegreen']
species = ['A', 'B']
num_species = len(species)

time = np.linspace(0, 10, 15)
posterior_samples = samples
n_post = posterior_samples.shape[0]

for i in range(num_datasets):
    fig, ax = plt.subplots()
    ic = initial_conditions[i]
    pred_samples = np.zeros((n_post, 2, len(time)))
    
    print(f"Propagating uncertainty for experiment {i+1}...")
    for j in range(n_post):
        sol = solve_ivp(kinetic_model, [time[0], time[-1]], ic, t_eval=time, method="RK45", 
                        args=tuple(posterior_samples[j]))
        pred_samples[j, :, :] = sol.y

    for k in range(num_species):
        ax.set_ylabel("Concentration $(M)$", fontsize=18)
        ax.set_xlabel("Time $(h)$", fontsize=18)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(alpha=0.5)
        
        ax.plot(time, z_obs[i][k, :], '.', label=species[k], color=color_1[k], markersize=10)
        
        species_predictions = pred_samples[:, k, :]
        ave_mean = np.mean(species_predictions, axis=0)
        ave_std = np.std(species_predictions, axis=0)
        upper_bound = ave_mean + 3 * ave_std
        lower_bound = ave_mean - 3 * ave_std
        
        ax.plot(time, ave_mean, '-', color=color_1[k], linewidth=2)
        ax.fill_between(time, lower_bound, upper_bound, color=color_1[k], alpha=0.5)
        
    if i == 1:
        file_path = 'Synthetic_Isomerisation_no_constraints/graphs/uncertainty_experiment_2_LAPLACE.png' # Changed name
        plt.savefig(file_path, dpi = 600, bbox_inches = "tight")

    ax.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.show()