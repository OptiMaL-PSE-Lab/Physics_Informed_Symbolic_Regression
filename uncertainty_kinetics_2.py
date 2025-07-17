#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:32:14 2023

@author: md1621
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import functools
import operator
import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import odeint
import scipy.integrate as scp
import scipy.stats
import sobol_seq
from scipy.optimize import minimize 
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax.example_libraries import optimizers
import jax
import jaxopt
np.random.seed(1998)


# --- Ground truth model definition --- #
def ground_truth(state, t, *parameters):
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
    dzdt = jnp.array([dTdt, dHdt, dBdt, dMdt])

    return dzdt


# --- dynamic model definition --- #
def kinetic_model(state, t, *parameters):
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
    k_4 = params[3]
    k_5 = params[4]

    # variable rate equations
    dTdt = (-1) * (k_1*CH*CT/(k_2*CB + k_3*CH + k_4*CT - k_5))
    dHdt = (-1) * (k_1*CH*CT/(k_2*CB + k_3*CH + k_4*CT - k_5))
    dBdt = (k_1*CH*CT/(k_2*CB + k_3*CH + k_4*CT - k_5))
    dMdt = (k_1*CH*CT/(k_2*CB + k_3*CH + k_4*CT - k_5))
    dzdt = jnp.array([dTdt, dHdt, dBdt, dMdt])

    return dzdt

# --- simulation --- #
def JAX_rk4_integrator_true(parameters, steps, tf, x0):
    '''
    simulate for a number of steps to collect data, and a final time
    '''
    # internal definitions
    t = jnp.linspace(0, tf, steps + 1)
    model = ground_truth 
    return odeint(model, jnp.array(x0), t, *parameters) # rtol=1e-6, atol=1e-6

def JAX_rk4_integrator_model(parameters, steps, tf, x0):
    '''
    simulate for a number of steps to collect data, and a final time
    '''
    # internal definitions
    t = jnp.linspace(0, tf, steps + 1)
    model = kinetic_model 
    return odeint(model, jnp.array(x0), t, *parameters) # rtol=1e-6, atol=1e-6

# --- simulation --- #
def JAX_rk4_integrator_1s(parameters, steps, tf, x0):
    '''
    simulate for a number of steps to collect data, and a final time
    '''
    # internal definitions
    t = jnp.linspace(0, tf, steps + 1)
    model = kinetic_model # hybrid_dynamics
    return odeint(model, x0, t, *parameters)[-1, :] # rtol=1e-6, atol=1e-6


####################################################
# Creating data from existing model and parameters #
####################################################

p  = {'k_1' : 2, 'k_2' : 9, 'k_3' : 5}   # r system)
jnp_parameters = jnp.array([val for val in p.values()], dtype = float)

tf       = 10
steps_   = 30
tt = jnp.linspace(0, tf, steps_ + 1)

# jnp_x0 = {
#     "exp_1": jnp.array([3.631, 3.168, 1.267, 0.746], dtype = float),
#     "exp_2": jnp.array([5, 8, 0, 0.5], dtype = float),
#     "exp_3": jnp.array([5, 3, 0, 0.5], dtype = float),
#     "exp_4": jnp.array([2.864, 4.675, 1.145, 1.033], dtype = float),
#     "exp_5": jnp.array([2.340, 5.080, 1.732, 0.890], dtype = float),
#     }

jnp_x0 = {
    "exp_1": jnp.array([4.023263936332148, 5.440619732170679, 1.472497763119326, 2.456190975122891], dtype = float),
    "exp_2": jnp.array([2.6654401608035925, 6.293682364324466, 1.7245460949603153, 2.0991310216230294], dtype = float),
    "exp_3": jnp.array([2.16911060048957, 6.240108087399554, 0.10041158969859953, 2.9170774472030945], dtype = float),
    "exp_4": jnp.array([2.459171970962336, 7.5687702105417785, 1.2285441828944312, 2.416486845574237], dtype = float),
    "exp_5": jnp.array([1.8510109976248708, 3.6064970459101544, 0.9033685476282325, 2.039381757067267], dtype = float)
    }
num_exp = len(jnp_x0)
jnp_xt = {}
jnp_xobs = {}

for i in range(num_exp):
    name = "exp_" + str(i + 1)
    ic = jnp_x0[name]
    jnp_xt[name] = JAX_rk4_integrator_true(jnp_parameters, steps_, tf, ic)
    random_noise = np.random.multivariate_normal(np.array([0, 0, 0, 0]), \
                                                 np.diag(np.array([0.03, 0.03, 0.03, 0.03])), \
                                                     steps_ + 1)   # change definition if using linear system
    jnp_xobs[name] = jnp.clip(jnp_xt[name] + random_noise, 0, 1e99)
    jnp_xobs[name] = jnp.vstack((ic, jnp_xobs[name][1:]))


species = ['T', 'H', 'B', 'M']
num_species = len(species)
color_1 = ['salmon', 'royalblue', 'darkviolet', 'limegreen']

for i in range(num_exp):
    fig, ax = plt.subplots()
    name = "exp_" + str(i + 1)
    ax.set_ylabel("Concentrations $(M)$", fontsize = 18)
    ax.set_xlabel("Time $(h)$", fontsize = 18)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    
    for j in range(num_species):
        ax.plot(tt, jnp_xobs[name][:, j], '.', label = species[j], color = color_1[j])
        # ax.plot(tt,jnp_xt[name][:, j], '-', color = color_1[j])
    
    ax.grid(alpha = 0.5)
    ax.legend(loc='upper right', fontsize = 15)
plt.show()


# laplace approximation according to reasoning in https://dl.acm.org/doi/pdf/10.1145/3520304.3533964?casa_token=WRkXnJmGSvoAAAAA:tmfWZ5V5DnOgJy4IylB5EeKnAMcmHuW4RsGPsbQxffjaBKhxGpRVTH6wWlJ0X7Q66v3gWbrvZxy88A
# define ground truth parameter vector
p_num = jnp.array([p[key1] for key1 in p], dtype = float)
 
# define prior parameters
jnp_mean = jnp.array([0.32253465, 1.51341416155509, 0.14881949, 1, 0.6003178])#1.669, 7.347, 4.439])
jnp_cov  = jnp.diag(jnp.array([4, 4, 4, 4, 4])) # covariance massively influences the uncertainty


# define likelihood parameters
# obs_data = jnp.vstack(list(jnp_xobs.values()))
# model_data = jnp.vstack(list(jnp_xt.values()))
# jnp_Qmat = jnp.diag(jnp.power(jnp.std(obs_data, axis = 0), 2))
# jnp_Qmat = jnp.linalg.inv(jnp_Qmat)   # set to identity if using liner systems

# def JAX_NLL(theta, xt = model_data, xobs = obs_data, Qmat = jnp_Qmat):
#   x_error = xobs - xt
#   nll = jnp.array([0])
#   num_samples = x_error.shape[0]
  
#   for i in range(num_samples):
#     #jax.debug.print("🤯 state error at index {x[1]}, {x[0]} 🤯", x=[x_error[i],i])
#     nll += jnp.matmul(jnp.matmul(x_error[i], Qmat), x_error[i].T)
#     #jax.debug.print("🤯 NLL contribution at index {x[1]}, {x[0]} 🤯", x=[jnp.matmul(jnp.matmul(x_error[i],Qmat), x_error[i].T),i])

#   return nll.squeeze()


jnp_Qmat = {}

for i in range(num_exp):
    name = "exp_" + str(i + 1)
    jnp_Qmat[name] = jnp.diag(jnp.power(jnp.std(jnp_xobs[name], axis = 0), 2))
    jnp_Qmat[name] = jnp.linalg.inv(jnp_Qmat[name])   # set to identity if using liner systems


# init_cond = jnp_x0["exp_1"]
# observ = jnp_xobs["exp_1"]
# model = jnp_xt["exp_1"]
# Q_matrix = jnp_Qmat["exp_1"]

def JAX_NLL(theta, x0, xobs, Qmat, steps_ = steps_, tf = tf):
    xt = JAX_rk4_integrator_model(theta, steps_, tf, x0)
    x_error = xobs - xt
    nll = jnp.array([0])
    num_samples = x_error.shape[0]
  
    for i in range(num_samples):
        #jax.debug.print("🤯 state error at index {x[1]}, {x[0]} 🤯", x=[x_error[i],i])
        nll += jnp.matmul(jnp.matmul(x_error[i], Qmat), x_error[i].T)
        #jax.debug.print("🤯 NLL contribution at index {x[1]}, {x[0]} 🤯", x=[jnp.matmul(jnp.matmul(x_error[i],Qmat), x_error[i].T),i])

    return nll.squeeze()


def JAX_NLL_parameters(parameters, x0, xobs, Qmat):
    
  return JAX_NLL(parameters, x0, xobs, Qmat)


def one_step_prediction(theta, x0, tf = tf):
    
  return JAX_rk4_integrator_1s(theta, 1, tf, x0)


def grad_map_pred(theta, x0, tf):
    
  return one_step_prediction(theta, x0, tf)


def grad_x_pred(x0, theta, tf):
    
  return one_step_prediction(theta, x0, tf)


# define jacobian for MAP prediction without controller
grad_fun = jax.jacobian(grad_map_pred, has_aux = False)

# define jacobian for MAP prediction w.r.t state
grad_fun_x = jax.jacobian(grad_x_pred, has_aux = False)

# define NLL grad + Hessian wrt MAP parameters

# need to calc Hessian of MAP wrt. prior
def normal_prior(theta, mean, cov):
  # mean - ndim vector of mean parameters
  # cov - ndim x ndim covariance matrix on parameters
  # theta - ndim vector of map parameters
  theta, mean = theta.reshape(-1, 1), mean.reshape(-1, 1)
  k = theta.shape[0]
  cov_inv = jnp.linalg.inv(cov)
  log_ = jnp.log(jnp.linalg.det(cov)) \
      + jnp.matmul(jnp.matmul((theta - mean).T, cov_inv), (theta - mean)) + \
          jnp.multiply(k, jnp.log(jnp.multiply(2, jnp.pi)))
          
  return jnp.multiply(-0.5, log_)


def normal_prior_theta(theta):
  # evaluate probability of map parameters under prior
  
  return normal_prior(theta, mean = jnp_mean, cov = jnp_cov).squeeze()

grad_theta_prior = jax.grad(normal_prior_theta)
hess_theta_prior = jax.jacobian(grad_theta_prior)


def inv_curv_neg_log_posterior(theta, x0, xobs, Qmat):
  # implement means to calc inv. curvature of neg log post

  return AD_NLL_hess(theta, x0, xobs, Qmat) - hess_theta_prior(theta)


def pd_laplace_approx(theta, x0, xobs, Qmat, tf = tf, sigma = 0):
  mean_pred = one_step_prediction(theta, x0, tf)
  A = inv_curv_neg_log_posterior(theta, x0, xobs, Qmat)
  #print(A)
  A_inv = jnp.linalg.inv(A)
  #print(A_inv)
  grad_map = grad_fun(theta, x0, tf) # gradient of MAP parameters with respect to prediction
  # print(grad_map)
  covariance = jnp.matmul(jnp.matmul(grad_map, A_inv), grad_map.T) \
      + jnp.multiply(sigma, jnp.eye(x0.shape[0], x0.shape[0]))
  
  return mean_pred, covariance

# bring it all together propagate uncertainty over the horizon
for j in range(num_exp):
    name = "exp_" + str(j + 1)
    steps_ = 30
    state_history = [None for _ in range(steps_ + 1)]
    cov_history = [None for _ in range(steps_ + 1)]
    state_history[0] = jnp_x0[name].reshape(-1, 1)
    cov_history[0] = jnp.zeros((4, 4))
    
    x0 = jnp_x0[name]
    xobs = jnp_xobs[name]
    Qmat = jnp_Qmat[name]
    AD_NLL_first = jax.grad(JAX_NLL_parameters, argnums = 0)
    AD_NLL_hess = jax.jacobian(AD_NLL_first, argnums = 0)
    
    theta_map = jnp_mean # here we are assuming that the map parameters are the ground truth, but in practice we would estimate these.
    # this can potentially be jitted via https://github.com/google/jax/discussions/8706
    
    for i in range(steps_):
      print(i)
      tf_ = tf / steps_ * (i + 1)
      mean_, var_ = pd_laplace_approx(theta_map, x0, xobs, Qmat, \
                                      tf = tf_, sigma = 0.0)  # note if sigma changes to nonzero then we should modify the NLL definition also,
      state_history[i + 1], cov_history[i + 1] = mean_.copy(), var_.copy()
    
    state_history = jnp.vstack([s_h.reshape(1, -1) for s_h in state_history])
    cov_history = jnp.vstack([jnp.diag(c_h).reshape(1, -1) for c_h in cov_history])
    
    # Plot MAP prediction plus state uncertainty
    color_2 = ['lightsalmon', 'cornflowerblue', 'plum', 'palegreen']
    
    fig, ax = plt.subplots()
    for i in range(num_species):
        # plt.plot(tt,jnp_xobs.T[i,:]/np.max(jnp_xobs.T[i,:]), '.')
        # plt.plot(tt,jnp_xt.T[i,:]/np.max(jnp_xobs.T[i,:]), '-')
        ax.set_ylabel("Concentration $(M)$", fontsize = 18)
        ax.set_xlabel("Time $(h)$", fontsize = 18)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
        ax.plot(tt, jnp_xobs[name].T[i, :], '.', color = color_1[i])
        ax.plot(tt, state_history.T[i, :], '-', label = species[i], color = color_1[i])
        # plotting plus-minus two standard deviations
        ax.fill_between(tt, state_history.T[i,:] + 2 * jnp.power(cov_history.T[i, :], 0.5), \
                        state_history.T[i, :] - 2 * jnp.power(cov_history.T[i, :], 0.5), color = color_2[i])
        ax.grid(alpha = 0.5)
        ax.legend(loc='upper right', fontsize = 15)
    plt.show()






