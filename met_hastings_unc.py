#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:24:40 2023

@author: md1621
"""

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
np.random.seed(1998)

# --- the data --- #
num_datasets = 9
t_obs = np.linspace(0.0, 10.0, 15)
z_obs = [pd.DataFrame.to_numpy(pd.read_csv('physics_informed_SR/exp_data/exp_' + str(i + 1) + '.csv', \
                                           header = None)) for i in range(num_datasets)]
initial_conditions = [np.array([1.0, 8.0, 2.0, 3.0]),
                      np.array([5.0, 8.0, 0.0, 0.5]),
                      np.array([5.0, 3.0, 0.0, 0.5]),
                      np.array([1.0, 3.0, 0.0, 3.0]),
                      np.array([1.0, 8.0, 2.0, 0.5]),
                      
                      np.array([5.0, 7.00054882, 2.0, 2.46589092]),
                      np.array([1.94379294, 3.63272094, 0.0, 2.4295672]),
                      np.array([2.34042014, 3.0, 0.0, 2.42896106]),
                      np.array([1.94269857, 3.48852451, 0.0, 2.42959711])]

# --- the dynamic model --- #
def kinetic_model(t, state, *parameters):
    # internal definitions
    params = parameters

    # state vector
    CT = state[0]
    CH = state[1]
    CB = state[2]
    CM = state[3]

    # parameters
    k_1 = params[0]
    k_2 = params[1]
    k_3 = params[2]

    # variable rate equations
    dTdt = (-1) * ((k_1 * CH * CT) / (1 + k_2 * CB + k_3 * CT))
    dHdt = (-1) * ((k_1 * CH * CT) / (1 + k_2 * CB + k_3 * CT))
    dBdt = ((k_1 * CH * CT) / (1 + k_2 * CB + k_3 * CT))
    dMdt = ((k_1 * CH * CT) / (1 + k_2 * CB + k_3 * CT))
    dzdt = np.array([dTdt, dHdt, dBdt, dMdt])

    return dzdt


# --- the priors of the parameters --- #
def prior(parameters):
    prior_means = np.array([4.2, 12, 5])
    prior_covs = np.diag(np.array([2, 2, 2]))
    return multivariate_normal.pdf(parameters, mean=prior_means, cov=prior_covs)


# --- the likelihood function --- #
def sse(parameters):
    error = np.zeros(num_datasets)
    obs_data = z_obs
    timesteps = 15
    time = np.linspace(0, 10, timesteps)
    t = [0, np.max(time)]
    t_eval = list(time)

    
    for i in range(num_datasets):
        ic = initial_conditions[i]
        pred = solve_ivp(kinetic_model, t, ic, t_eval = t_eval, method = "RK45",\
                         args = (parameters))
        error[i] = np.sum((pred.y - obs_data[i])**2)
        
    return np.sum(error)


# --- Bayesian Inference using MH --- #
def metropolis_hastings(initial_parameters, num_samples, proposal_std):
    current_parameters = initial_parameters
    samples = []

    for _ in range(num_samples):
        if _ % 100 == 0:
            print(_)
        # Propose new parameters from a Gaussian distribution
        proposed_parameters = np.random.normal(current_parameters, proposal_std)
        epsilon = 0  # Define a small positive number
        proposed_parameters = [max(param, epsilon) for param in proposed_parameters]

        # Calculate the likelihood for the current and proposed parameters
        current_likelihood = sse(current_parameters)
        proposed_likelihood = sse(proposed_parameters)

        # Calculate the prior for the current and proposed parameters
        current_prior = prior(current_parameters)
        proposed_prior = prior(proposed_parameters)

        # Calculate the acceptance ratio
        acceptance_ratio = (proposed_likelihood * proposed_prior) / (current_likelihood * current_prior)

        # Accept or reject the proposed parameters based on the acceptance ratio
        if np.random.rand() < acceptance_ratio:
            current_parameters = proposed_parameters

        samples.append(current_parameters)

    return np.array(samples)

# Example usage
initial_parameters = np.array([7, 10, 4])
# initial_parameters = np.array([2.5, 9.5, 5])
num_samples = 1000 * 10
# proposal_std = np.array([0.015 for i in range(len(initial_parameters))])
proposal_std = np.array([0.15 for i in range(len(initial_parameters))])


samples = metropolis_hastings(initial_parameters, num_samples, proposal_std)

# Plot the posterior distributions of the parameters
# Set the overall figure size
plt.figure(figsize=(12, 6))

# Choose a color palette
colors = ['royalblue', 'salmon', 'limegreen', 'violet', 'gold', 'lightcoral']
gray_shades = ['#555555', '#888888', '#BBBBBB', '#DDDDDD', '#FFFFFF']

# Loop to create subplots for each parameter
for i in range(samples.shape[1]):
    ax = plt.subplot(1, samples.shape[1], i + 1)
    
    # Histogram for each parameter with customizations
    ax.hist(samples[:, i], bins=50, density=True, alpha=0.7, color=colors[i % len(colors)])
    
    # Titles and labels
    ax.set_title(f"Parameter {i + 1}", fontsize = 18)
    ax.set_xlabel('Value', fontsize = 18)
    ax.set_ylabel('Density' if i == 0 else '', fontsize = 18)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)

    # Adding grid for better readability
    ax.grid(True, linestyle='--', alpha=0.5)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# color_1 = ['salmon', 'royalblue', 'darkviolet', 'limegreen']

# species = ['T', 'H', 'B', 'M']
# num_species = len(species)
# for i in range(num_datasets):
#     fig, ax = plt.subplots()
#     timesteps = 15
#     time = np.linspace(0, 10, timesteps)
#     t = [0, np.max(time)]
#     t_eval = list(time)
#     ic = initial_conditions[i]
#     pred_samples = np.zeros((len(samples), 4, 15))
    
#     for j in range(len(samples)):
#         a = solve_ivp(kinetic_model, t, ic, t_eval = t_eval, method = "RK45",\
#                           args = (samples[j]))
#         pred_samples[j] = a.y
        

#     for k in range(num_species):
#         # ax.set_title("Experiment " + str(i + 1))
#         ax.set_ylabel("Concentration $(M)$", fontsize = 18)
#         ax.set_xlabel("Time $(h)$", fontsize = 18)
#         ax.spines["right"].set_visible(False)
#         ax.spines["top"].set_visible(False)
#         ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
#         ax.grid(alpha = 0.5)
        
#         ax.plot(time, z_obs[i][k], '.', label = species[k], color = color_1[k], markersize = 10)
#         a = [pred_samples[l][k] for l in range(int(len(samples)*0.1), len(samples))]
#         ave_mean = np.mean(a, axis = 0)
#         #TODO: add dashed line for real model
#         ax.plot(time, ave_mean, '-', color = color_1[k], linewidth = 2)
#         ave_std = np.std(a, axis = 0)
#         upper_bound = ave_mean + 2 * ave_std
#         lower_bound = ave_mean - 2 * ave_std
#         ax.fill_between(time, upper_bound, lower_bound, color = color_1[k], alpha = 0.5)
#         ax.legend(loc = 'upper right', fontsize = 15)
# plt.show()



