#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 15:51:59 2025

@author: md1621
"""

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

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

# --- the prior for the parameters --- #
def prior(parameters):
    prior_means = np.array([7.689, 1.896, 4.053, 1.608, 5.943])
    prior_covs = np.diag(np.array([2, 2, 2, 2, 2]))
    return multivariate_normal.pdf(parameters, mean=prior_means, cov=prior_covs)

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

# --- Define the likelihood based on the SSE --- #
def likelihood(parameters):
    # Assuming a Gaussian likelihood with variance=1 for simplicity.
    return np.exp(-sse(parameters) / 2.0)

# --- Bayesian Inference using Metropolis-Hastings --- #
def metropolis_hastings(initial_parameters, num_samples, proposal_std):
    current_parameters = np.array(initial_parameters)
    samples = []
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"Iteration: {i}")
        # Propose new parameters from a symmetric Gaussian proposal
        proposed_parameters = np.random.normal(current_parameters, proposal_std)
        # Ensure parameters remain non-negative (if that is required)
        proposed_parameters = np.maximum(proposed_parameters, 0)
        
        # Compute the target densities (likelihood * prior)
        current_target = likelihood(current_parameters) * prior(current_parameters)
        proposed_target = likelihood(proposed_parameters) * prior(proposed_parameters)
        
        # Compute acceptance ratio and ensure it is at most 1
        acceptance_ratio = min(1, proposed_target / current_target)
        
        # Accept or reject the proposed move
        if np.random.rand() < acceptance_ratio:
            current_parameters = proposed_parameters
        
        samples.append(current_parameters.copy())
    
    return np.array(samples)

# --- Example usage --- #
initial_parameters = np.array([7, 4, 3, 2, 6]) + np.random.normal(0, 1, 5)
num_samples = 1000 * 10  # e.g., 10,000 samples
proposal_std = np.array([0.15, 0.15, 0.15, 0.15, 0.15]) * 2  # standard deviation for proposals

samples = metropolis_hastings(initial_parameters, num_samples, proposal_std)

# --- Plot the posterior distributions of the parameters --- #
# plt.figure(figsize=(18, 6))
# num_params = samples.shape[1]
# colors = ['royalblue', 'salmon', 'limegreen']

# for i in range(num_params):
#     ax = plt.subplot(1, num_params, i + 1)
#     ax.hist(samples[:, i], bins=50, density=True, alpha=0.7, color=colors[i % len(colors)])
#     ax.set_title(f"Parameter {i + 1}", fontsize=28)
#     ax.set_xlabel('Value', fontsize=28)
#     if i == 0:
#         ax.set_ylabel('Density', fontsize=28)
#     ax.tick_params(axis = 'both', which = 'major', labelsize = 28)
#     ax.grid(True, linestyle='--', alpha=0.5)
    
# file_path = 'Hydrodealkylation_of_Toluene_no_constraints/graphs/parameter_uncertainty.png'
# plt.savefig(file_path, dpi = 600, bbox_inches = "tight")

# plt.tight_layout()
# plt.show()

num_params = samples.shape[1]          # here: 5
colors      = ['royalblue', 'salmon', 'limegreen', 'darkviolet', 'goldenrod']

# 2 rows × 3 columns ⇒ 6 axes; we’ll hide the last one
fig, axes = plt.subplots(nrows=2, ncols=3,
                         figsize=(18, 12),      # a bit taller than before
                         constrained_layout=True)

axes = axes.ravel()                     # flatten to 1-D array for easy indexing

for i in range(num_params):
    ax = axes[i]
    ax.hist(samples[:, i],
            bins=50, density=True, alpha=0.7,
            color=colors[i % len(colors)])
    ax.set_title(f'Parameter {i + 1}', fontsize=28)
    ax.set_xlabel('Value', fontsize=28)
    if i % 3 == 0:                      # leftmost column
        ax.set_ylabel('Density', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.grid(True, linestyle='--', alpha=0.5)

# turn off the 6th (unused) axis
if num_params < len(axes):
    axes[-1].set_visible(False)

file_path = ('Synthetic_Isomerisation_no_constraints/'
             'graphs/parameter_uncertainty.png')
fig.savefig(file_path, dpi=600, bbox_inches='tight')
plt.show()


# --- Prediction and Uncertainty Plot for All Experiments ---

# Define colors and species labels
color_1 = ['salmon', 'royalblue', 'darkviolet', 'limegreen']
species = ['A', 'B']
num_species = len(species)

# Define a time grid (consistent with your data)
time = np.linspace(0, 10, 15)

# Define burn-in: here we discard 10% of the samples.
burn_in = int(0.1 * len(samples))
posterior_samples = samples[burn_in:]
n_post = posterior_samples.shape[0]

# Loop over each experiment (dataset)
for i in range(num_datasets):
    # Create a new figure for the current experiment
    fig, ax = plt.subplots()
    
    # Get the initial condition for the current experiment
    ic = initial_conditions[i]
    
    # Preallocate an array to hold predictions.
    # Dimensions: (number of posterior samples, number of species, number of time points)
    pred_samples = np.zeros((n_post, 2, len(time)))
    
    # For each posterior sample, simulate the kinetic model over the time grid.
    for j in range(n_post):
        # Use tuple(posterior_samples[j]) so that kinetic_model can unpack the parameters
        sol = solve_ivp(kinetic_model, [time[0], time[-1]], ic, t_eval=time, method="RK45", 
                        args=tuple(posterior_samples[j]))
        pred_samples[j, :, :] = sol.y

    # Loop over each species (T, H, B, M)
    for k in range(num_species):
        # Format the axes
        ax.set_ylabel("Concentration $(M)$", fontsize=18)
        ax.set_xlabel("Time $(h)$", fontsize=18)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(alpha=0.5)
        
        # Plot the experimental data for experiment i and species k
        ax.plot(time, z_obs[i][k, :], '.', label=species[k], color=color_1[k], markersize=10)
        
        # Extract predictions for species k across all posterior samples
        species_predictions = pred_samples[:, k, :]
        ave_mean = np.mean(species_predictions, axis=0)
        ave_std = np.std(species_predictions, axis=0)
        upper_bound = ave_mean + 3 * ave_std
        lower_bound = ave_mean - 3 * ave_std
        
        # Plot the mean prediction as a solid line
        ax.plot(time, ave_mean, '-', color=color_1[k], linewidth=2)
        # Plot the uncertainty as a shaded region
        ax.fill_between(time, lower_bound, upper_bound, color=color_1[k], alpha=0.5)
        
    if i == 1:
        file_path = 'Synthetic_Isomerisation_no_constraints/graphs/uncertainty_experiment_2.png'
        plt.savefig(file_path, dpi = 600, bbox_inches = "tight")

    ax.legend(loc='upper right', fontsize=15)
    # ax.set_title(f"Experiment {i+1}", fontsize=18)
    plt.tight_layout()
    plt.show()
