#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:09:27 2023

@author: md1621
"""

import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
from tensorflow_probability.python.mcmc import RandomWalkMetropolis
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras

# Ensure that you have eager execution enabled (should be by default in TF 2.x)
tf.executing_eagerly()

# Example data (replace with your actual data)
num_datasets = 5
t_obs = tf.linspace(0.0, 10.0, num=15)
z_obs = [tf.transpose(tf.convert_to_tensor(pd.read_csv('exp_data/exp_' + str(i + 1) + '.csv', \
                                          header = None))) for i in range(num_datasets)]  # replace with actual data
initial_conditions = [tf.constant([1.0, 8.0, 2.0, 3.0]),
                       tf.constant([5.0, 8.0, 0.0, 0.5]),
                       tf.constant([5.0, 3.0, 0.0, 0.5]),
                       tf.constant([1.0, 3.0, 0.0, 3.0]),
                       tf.constant([1.0, 8.0, 2.0, 0.5])]

dtype = tf.float32

# Convert all your data to this dtype
t_obs = tf.cast(t_obs, dtype)
z_obs = [tf.cast(z, dtype) for z in z_obs]
initial_conditions = [tf.cast(ic, dtype) for ic in initial_conditions]

def kinetic_model(t, z, k1, k2, k3):
    dTdt = (-1) * ((k1 * z[:,1] * z[:,0]) / (1 + k2 * z[:,2] + k3 * z[:,0]))
    dHdt = (-1) * ((k1 * z[:,1] * z[:,0]) / (1 + k2 * z[:,2] + k3 * z[:,0]))
    dBdt = ((k1 * z[:,1] * z[:,0]) / (1 + k2 * z[:,2] + k3 * z[:,0]))
    dMdt = ((k1 * z[:,1] * z[:,0]) / (1 + k2 * z[:,2] + k3 * z[:,0]))
    return tf.stack([dTdt, dHdt, dBdt, dMdt], axis = -1)

def model():
    k1 = yield tfd.Normal(loc=tf.cast(2.0, dtype), scale=tf.cast(0.1, dtype), name='k1')
    k2 = yield tfd.Normal(loc=tf.cast(9.0, dtype), scale=tf.cast(0.1, dtype), name='k2')
    k3 = yield tfd.Normal(loc=tf.cast(5.0, dtype), scale=tf.cast(0.1, dtype), name='k3')
    for i in range(num_datasets):
        z0 = initial_conditions[i]
        solutions = tfp.math.ode.BDF().solve(
            lambda t, z: kinetic_model(t, tf.reshape(z, [-1, 4]), k1, k2, k3),
            initial_time=0.0, initial_state=z0, solution_times=t_obs)
        z_pred = solutions.states
        print("z_pred shape:", z_pred, "z_obs shape:", z_obs[i])
        yield tfd.Normal(loc=z_pred, scale=0.1, name=f'z_obs_{i}')

# Convert the generator to a joint distribution
joint_distribution = tfd.JointDistributionCoroutine(model)

# Define the target log probability function
def target_log_prob_fn(k1, k2, k3):
    return joint_distribution.log_prob((k1, k2, k3, *z_obs))

# # Run MCMC
num_results = 100
num_burnin_steps = 50
# adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
#     tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn, step_size=0.01, num_leapfrog_steps=3),
#     num_adaptation_steps=int(num_burnin_steps * 0.8))

@tf.function
def run_chain():
    # Initialize the sampler
    mh = RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn
    )
    
    # Run the chain
    samples, is_accepted = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=[tf.ones([], name='init_k1'),
                        tf.ones([], name='init_k2'),
                        tf.ones([], name='init_k3')],
        kernel=mh,
        num_burnin_steps=num_burnin_steps,
        trace_fn=lambda _, pkr: pkr.is_accepted
    )
    return samples, is_accepted


samples, is_accepted = run_chain()


