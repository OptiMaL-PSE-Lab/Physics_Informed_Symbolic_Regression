#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hamiltonian Monte Carlo (HMC) with numerical gradients
for kinetic parameter inference
"""

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal, halfnorm 
import matplotlib.pyplot as plt

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
    CA, CB = state
    k_1, k_2, k_3, k_4, k_5 = parameters
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
        sol = solve_ivp(kinetic_model, t_span, ic, t_eval=time, method="RK45", args=tuple(parameters))
        total_error += np.sum((sol.y - z_obs[i])**2)
    return total_error


def log_posterior(parameters):
    """Compute log(posterior) = log(likelihood) + log(prior)"""
    # Prior: multivariate normal
    prior_means = np.array([5, 5, 5, 5, 5])
    prior_cov = np.diag(np.array([2., 2., 2., 2., 2.]))
    log_prior = halfnorm.logpdf(parameters, prior_means, prior_cov)
    
    # Likelihood: use numerical value
    log_likelihood = -sse(np.array(parameters)) / 2.0
    
    return float(log_likelihood) + float(log_prior)

# --- Numerical gradient of log posterior --- #
def numerical_gradient(log_p, theta, epsilon=1e-5):
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon
        grad[i] = (log_p(theta_plus) - log_p(theta_minus)) / (2 * epsilon)
    return grad
 
# --- Leapfrog integrator --- #
def leapfrog(theta, momentum, log_posterior_fn, step_size, num_steps, epsilon=1e-5):
    theta = np.array(theta, dtype=float)
    momentum = np.array(momentum, dtype=float)
    
    # Half-step for momentum at the beginning
    grad_theta = numerical_gradient(log_posterior_fn, theta, epsilon)
    momentum = momentum + (step_size / 2.0) * grad_theta
    
    # Full steps for position and momentum
    for _ in range(num_steps):
        theta = theta + step_size * momentum
        theta = np.maximum(theta, 1e-6)  # Ensure non-negative
        
        grad_theta = numerical_gradient(log_posterior_fn, theta, epsilon)
        momentum = momentum + step_size * grad_theta
    
    # Half-step for momentum at the end
    grad_theta = numerical_gradient(log_posterior_fn, theta, epsilon)
    momentum = momentum - (step_size / 2.0) * grad_theta
    
    return theta, momentum

# --- Hamiltonian Monte Carlo sampler --- #
def hamiltonian_monte_carlo(initial_parameters, num_samples, step_size, num_steps):
    current_parameters = np.array(initial_parameters, dtype=float)
    current_parameters = np.maximum(current_parameters, 1e-6)
    
    samples = []
    num_accepts = 0
    
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"Iteration: {i}, Acceptance rate: {num_accepts / max(i, 1):.3f}")
        
        # Sample random momentum
        momentum_current = np.random.normal(0, 1, size=current_parameters.shape)
        
        # Compute current Hamiltonian
        log_p_current = log_posterior(current_parameters)
        hamiltonian_current = -log_p_current + 0.5 * np.sum(momentum_current**2)
        
        # Propose new state using leapfrog
        proposed_parameters, proposed_momentum = leapfrog(
            current_parameters, 
            momentum_current, 
            log_posterior, 
            step_size, 
            num_steps
        )
        
        proposed_parameters = np.maximum(proposed_parameters, 1e-6)
        
        # Compute proposed Hamiltonian
        log_p_proposed = log_posterior(proposed_parameters)
        hamiltonian_proposed = -log_p_proposed + 0.5 * np.sum(proposed_momentum**2)
        
        # Metropolis acceptance
        log_acceptance_ratio = hamiltonian_current - hamiltonian_proposed
        
        if np.log(np.random.rand()) < log_acceptance_ratio:
            current_parameters = proposed_parameters
            num_accepts += 1
        
        samples.append(current_parameters.copy())
    
    acceptance_rate = num_accepts / num_samples
    print(f"Final acceptance rate: {acceptance_rate:.3f}")
    return np.array(samples), acceptance_rate

# --- Example usage --- #
initial_parameters = np.array([7, 4, 3, 2, 6]) + np.random.normal(0, 1, 5)
initial_parameters = np.maximum(initial_parameters, 1e-6)

num_samples = 1000
step_size = 0.20
num_steps = 25

print("Running Hamiltonian Monte Carlo")
samples, acceptance_rate = hamiltonian_monte_carlo(initial_parameters, num_samples, step_size, num_steps)

print(f"\nAcceptance rate: {acceptance_rate:.3f}")

# --- Plot the posterior distributions --- #
num_params = samples.shape[1]
colors = ['royalblue', 'salmon', 'limegreen', 'darkviolet', 'goldenrod']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), constrained_layout=True)
axes = axes.ravel()

for i in range(num_params):
    ax = axes[i]
    ax.hist(samples[:, i], bins=50, density=True, alpha=0.7, color=colors[i % len(colors)])
    ax.set_title(f'Parameter {i + 1}', fontsize=28)
    ax.set_xlabel('Value', fontsize=28)
    if i % 3 == 0:
        ax.set_ylabel('Density', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.grid(True, linestyle='--', alpha=0.5)

if num_params < len(axes):
    axes[-1].set_visible(False)

file_path = 'Synthetic_Isomerisation_no_constraints/graphs/parameter_uncertainty_hmc.png'
fig.savefig(file_path, dpi=600, bbox_inches='tight')
plt.show()

# --- Prediction and Uncertainty Plot for All Experiments --- #
color_1 = ['salmon', 'royalblue', 'darkviolet', 'limegreen']
species = ['A', 'B']
num_species = len(species)

time = np.linspace(0, 10, 15)
burn_in = int(0.1 * len(samples))
posterior_samples = samples[burn_in:]
n_post = posterior_samples.shape[0]

for i in range(num_datasets):
    fig, ax = plt.subplots()
    ic = initial_conditions[i]
    pred_samples = np.zeros((n_post, 2, len(time)))
    
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
        file_path = 'Synthetic_Isomerisation_no_constraints/graphs/uncertainty_experiment_2_hmc.png'
        plt.savefig(file_path, dpi=600, bbox_inches='tight')

    ax.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.show()