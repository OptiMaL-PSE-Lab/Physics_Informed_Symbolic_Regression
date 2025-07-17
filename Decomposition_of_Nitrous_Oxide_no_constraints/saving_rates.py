#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:39:30 2023

@author: md1621
"""

"##############################################################################"
"######################## Importing important packages ########################"
"##############################################################################"

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from pysr import PySRRegressor
from sympy import *
from scipy.misc import derivative as der
import re
from scipy.integrate import solve_ivp
import itertools as it 
from time import perf_counter
import matplotlib.cm as cm
import pandas as pd
import os
np.random.seed(1998)


"##############################################################################"
"####################### Synthetic data from case study #######################"
"##############################################################################"

# Case study
def kinetic_model(t, z):
    k_1 = 2 
    k_2 = 5

    dNOdt = (-1) * ((k_1 * z[0]**2) / (1 + k_2 * z[0]))
    dNdt = ((k_1 * z[0]**2) / (1 + k_2 * z[0]))
    dOdt = (1/2) * ((k_1 * z[0]**2) / (1 + k_2 * z[0]))

    dzdt = [dNOdt, dNdt, dOdt]
    return dzdt

# Plotting the data given
species = ["NO", "N", "O"]
initial_conditions = {
    "ic_1": np.array([5 , 0, 0]),
    "ic_2": np.array([10, 0, 0]),
    "ic_3": np.array([5 , 2, 0]),
    "ic_4": np.array([5 , 0, 3]),
    "ic_5": np.array([0 , 2, 3]),
    
    "ic_6": np.array([1.24494203e-05, 1.52186994e+00, 7.31244243e-01]),
}

num_exp = len(initial_conditions)
num_species = len(species)

timesteps = 15
time = np.linspace(0, 10, timesteps)
t = [0, np.max(time)]
t_eval = list(time)
STD = 0.2
noise = [np.random.normal(0, STD, size = (num_species, timesteps - 1)) for i in range(num_exp)]
in_silico_data = {}
no_noise_data = {}

for i in range(num_exp):
    ic = initial_conditions["ic_" + str(i + 1)]
    solution = solve_ivp(kinetic_model, t, ic, t_eval = t_eval, method = "RK45")
    a = np.reshape(ic, (num_species, 1))
    in_silico_data["exp_" + str(i + 1)] = np.hstack((a, np.clip(solution.y[:, 1:] + noise[i], 0, 1e99)))
    no_noise_data["exp_" + str(i + 1)] = solution.y

color_1 = ['salmon', 'royalblue', 'darkviolet', 'limegreen']
marker = ['o', 'o', 'o', 'o']

# Plotting the in-silico data for visualisation
for i in range(num_exp):
    fig, ax = plt.subplots()
    # ax.set_title("Experiment " + str(i + 1))
    ax.set_ylabel("Concentration $(M)$", fontsize = 18)
    ax.set_xlabel("Time $(h)$", fontsize = 18)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)

    for j in range(num_species):
        y = in_silico_data["exp_" + str(i + 1)][j]
        ax.plot(time, y, marker[j], markersize = 4, label = species[j], color = color_1[j])

    ax.grid(alpha = 0.5)
    ax.legend(fontsize = 15)
    
    if i == 3:
        file_path = 'Physics-Informed_ADoK/Decomposition_of_Nitrous_Oxide_no_constraints/Graphs/PI_ADoK_in_silico_data_experiment_4.png'
        plt.savefig(file_path, dpi = 600, bbox_inches = "tight")

        
# plt.show()

def save_matrix_as_csv(matrix, filename):
    # Convert numpy matrix to pandas dataframe
    df = pd.DataFrame(matrix)
        
    # Save dataframe as CSV file in exp_data directory without index
    filepath = os.path.join("Physics-Informed_ADoK/physics_informed_SR/Decomposition_Nitrous_Oxide/exp_data", filename + ".csv")
    df.to_csv(filepath, index = False, header = False)

for i in range(num_exp):
    name = "exp_" + str(i + 1)
    matrix = in_silico_data[name]
    save_matrix_as_csv(matrix, name)


"##############################################################################"
"###################### Selecting Concentration Profiles ######################"
"##############################################################################"

def read_equations(path):
    # Read equations from CSV with different separator 
    data = pd.read_csv(path)
    # Convert dataframe into numpy array
    eqs = data["Equation"].values
    
    eq_list = []
    # For every string equation in list...
    
    def make_f(eq):
        # Function takes a string equation, 
        # Converts exp to numpy representation
        # And returns the expression of that string 
        # As a function 
        def f(t):
            equation = eq.replace("x0", "t")
            return eval(equation.replace("exp", "np.exp"))
        return f
    
    for eq in eqs:
        # Iterate over expression strings and make functions
        # Then add to expression list
        eq_list += [make_f(eq)]
    
    return eq_list

def number_param(path):
    # Read equations from CSV with different separator 
    data = pd.read_csv(path)
    # Convert dataframe into numpy array
    eqs = data["Equation"].values
    t = symbols("t")
    simple_traj = []
    param = []
    
    for eq in eqs:
        func = simplify(eq)
        simple_traj.append(func)
        things = list(func.atoms(Float))
        param.append(len(things))
    
    simple_traj = np.array(simple_traj).tolist()
    return param

def find_best_model(NLL, param):
    # Finding the model with the lowest AIC value
    AIC = 2 * np.array(NLL) + 2 * np.array(param)
    index = np.where(AIC == np.min(AIC))
    return index[0][0]

def NLL_models(eq_list, t, data, NLL_species, number_datapoints):
    # Make list of NLL values for each equation
    NLL = []
    
    for f in eq_list:
        y_T = []
        
        for a in t:
            y_T.append(f(a))
        
        NLL.append(NLL_species(data, y_T, number_datapoints))
    return NLL

def NLL(C, y_C, number_datapoints):
    # Calculate the NLL value of a given equation
    likelihood = np.empty(number_datapoints)
    mse = np.empty(number_datapoints)
    
    for i in range(number_datapoints):
        mse[i] = ((C[i] - y_C[i])**2)
    
    variance = np.sum(mse) / number_datapoints
    
    for i in range(number_datapoints):
        likelihood[i] = ((C[i] - y_C[i])**2) / (2 * (variance)) \
            - np.log(1 / (np.sqrt(2 * np.pi * (variance))))
    
    return np.sum(likelihood)

# Find out which concentration models are best for each experiment
equation_lists = {}
best_models = {}

for i in range(num_exp):
    data = in_silico_data["exp_" + str(i + 1)]
    
    for j in range(num_species):
        if j == 0:
            file_name = str("Physics-Informed_ADoK/physics_informed_SR/Decomposition_Nitrous_Oxide/hof_files/hall_of_fame_NO" + str(i + 1) + ".csv")
            name = "NO_"
        if j == 1:
            file_name = str("Physics-Informed_ADoK/physics_informed_SR/Decomposition_Nitrous_Oxide/hof_files/hall_of_fame_N" + str(i + 1) + ".csv")
            name = "N_"
        if j == 2:
            file_name = str("Physics-Informed_ADoK/physics_informed_SR/Decomposition_Nitrous_Oxide/hof_files/hall_of_fame_O" + str(i + 1) + ".csv")
            name = "O_"
        
        a = read_equations(file_name)
        nll_a = NLL_models(a, time, data[j], NLL, timesteps)
        param_a = number_param(file_name)
        best_models[name + str(i + 1)] = find_best_model(nll_a, param_a)
        equation_lists[name + str(i + 1)] = a

# Plotting the selected concentration profile and in-silico data
for i in range(num_exp):
    fig, ax = plt.subplots()
    # ax.set_title("Concentration Profiles - Experiment " + str(i + 1))
    ax.set_ylabel("Concentrations $(M)$", fontsize = 18)
    ax.set_xlabel("Time $(h)$", fontsize = 18)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)

    for j in range(num_species):
        y = in_silico_data["exp_" + str(i + 1)][j]
        name = species[j] + "_" + str(i + 1)
        model = best_models[name]
        yy = equation_lists[name][model](time)
        print(i, j)
        print(model)
        ax.plot(time, y, marker[j], markersize = 4, label = species[j], color = color_1[j])
        ax.plot(time, yy, color = color_1[j], linestyle = "-")

    ax.grid(alpha = 0.5)
    ax.legend(fontsize = 15)
    
    if i == 3:
        file_path = 'Physics-Informed_ADoK/Decomposition_of_Nitrous_Oxide_no_constraints/Graphs/PI_ADoK_concentration_profile_experiment_4.png'
        plt.savefig(file_path, dpi = 600, bbox_inches = "tight")


# plt.show()


"##############################################################################"
"########################## Derivative Calculations ###########################"
"##############################################################################"

derivatives = {}
SR_derivatives_NO = np.array([])
SR_derivatives_N  = np.array([])
SR_derivatives_O  = np.array([])

# Getting the rate measurements from the model (realistically, never available)
# But just to check the fit of our estimates of the rate which are obtained by
# Numerically differentiating the concentration models selected
for i in range(num_exp):
    
    for j in range(num_species):
        name = species[j] + "_" + str(i + 1)
        model = best_models[name]
        best_model = equation_lists[name][model]
        derivative = np.zeros(timesteps)
        
        for h in range(timesteps):
            derivative[h] =  der(best_model, time[h], dx = 1e-6)
        
        derivatives[name] = derivative

# Plotting the estimated rates and the actual rates
for i in range(num_exp):
    fig, ax = plt.subplots()
    # ax.set_title("Derivative Estimates - Experiment " + str(i + 1))
    ax.set_ylabel("Rate $(Mh^{-1})$", fontsize = 18)
    ax.set_xlabel("Time $(h)$", fontsize = 18)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    data = no_noise_data["exp_" + str(i + 1)]
    y = kinetic_model(time, data)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)

    for j in range(num_species):
        name = species[j] + "_" + str(i + 1)
        yy = derivatives[name]
        ax.plot(time, y[j], marker[j], markersize = 4, label = species[j], color = color_1[j])
        ax.plot(time, yy, color = color_1[j], linestyle = "-")

    ax.grid(alpha = 0.5)
    ax.legend(fontsize = 15)
    
    if i == 3:
        file_path = 'Physics-Informed_ADoK/Decomposition_of_Nitrous_Oxide_no_constraints/Graphs/PI_ADoK_rate_estimates_experiment_4.png'
        plt.savefig(file_path, dpi = 600, bbox_inches = "tight")


# plt.show()

# Preparing the data for the second step of the symbolic regression methodology
for i in range(num_exp):
    SR_derivatives_NO = np.concatenate([SR_derivatives_NO, derivatives["NO_" + str(i + 1)]])
    SR_derivatives_N  = np.concatenate([SR_derivatives_N , derivatives["N_"  + str(i + 1)]])
    SR_derivatives_O  = np.concatenate([SR_derivatives_O , derivatives["O_"  + str(i + 1)]])

a = in_silico_data["exp_1"].T
b = in_silico_data["exp_2"].T
sr_data = np.vstack((a, b))

for i in range(2, num_exp):
    c = in_silico_data["exp_" + str(i + 1)].T
    sr_data = np.vstack((sr_data, c))
    
def save_matrix_as_csv(matrix, filename):
    # Convert numpy matrix to pandas dataframe
    df = pd.DataFrame(matrix)
        
    # Save dataframe as CSV file in exp_data directory without index
    filepath = os.path.join("Physics-Informed_ADoK/physics_informed_SR/Decomposition_Nitrous_Oxide/const_data", filename + ".csv")
    df.to_csv(filepath, index = False, header = False)

size = len(SR_derivatives_NO)
save_matrix_as_csv(sr_data[:, 0:3].T, 'conc_data_for_rate_models')
save_matrix_as_csv(np.reshape(SR_derivatives_NO, (1, size)), 'rate_data_NO')
save_matrix_as_csv(np.reshape(SR_derivatives_N, (1, size)), 'rate_data_N')
save_matrix_as_csv(np.reshape(SR_derivatives_O, (1, size)), 'rate_data_O')


"##############################################################################"
"############################ Find Best Rate Model ############################"
"##############################################################################"

# In this first part, we read the rate equations generated by symbolic regression
# And we pick the best equation for each species by evaluating them in the rate space
# Aka, we find the predicted rates and we compare with the estimated rates
# This is not the best way to do it, so essentially, ignore it
def rate_n_param(path):
    # read equations from CSV with different separator 
    data = pd.read_csv(path)
    # convert dataframe into numpy array
    eqs = data["Equation"].values
    T, H, B, M = symbols("T H B M")
    simple_traj = []
    param = []
    
    for eq in eqs:
        func = simplify(eq)
        func = str(func)
        j = 0
        things = re.findall(r"(\*{2}|\*{0})(\d+\.?\d*)", func)
        
        for i in range(len(things)):
            if things[i][0] != "**":
                j += 1
        
        simple_traj.append(func)
        param.append(int(j))
    # simple_traj = np.array(simple_traj).tolist()
    
    return simple_traj, param

rate_models = {}
GP_models = {}

for i in range(num_species):
    if i == 0:
        path = "Physics-Informed_ADoK/physics_informed_SR/Decomposition_Nitrous_Oxide/hof_files/hall_of_fame_rate_NO" + str(num_exp) + ".csv"
        name_models = "NO_models"
        name_params = "NO_params"
    
    if i == 1:
        path = "Physics-Informed_ADoK/physics_informed_SR/Decomposition_Nitrous_Oxide/hof_files/hall_of_fame_rate_N" + str(num_exp) + ".csv"
        name_models = "N_models"
        name_params = "N_params"
    
    if i == 2:
        path = "Physics-Informed_ADoK/physics_informed_SR/Decomposition_Nitrous_Oxide/hof_files/hall_of_fame_rate_O" + str(num_exp) + ".csv"
        name_models = "O_models"
        name_params = "O_params"
    
    a, b = rate_n_param(path)
    GP_models[name_models, name_params] = a, b

def NLL_rates(rate_est, rate_pred, number_datapoints, num_exp):
    mse = (rate_est - rate_pred)**2
    variance = np.sum(mse) / (number_datapoints * num_exp)
    likelihood = ((rate_est - rate_pred)**2) / (2 * (variance)) \
        - np.log(1 / (np.sqrt(2 * np.pi * (variance))))
    
    return np.sum(likelihood)

def predicting_rate(equation, z):
    equation = str(equation)
    equation = equation.replace("CNO", "z[:, 0]")
    equation = equation.replace("N", "z[:, 1]")
    equation = equation.replace("O", "z[:, 2]")
    rate_pred = eval(equation)
    
    return rate_pred

def best_rate_model(NLL, param):
    AIC = 2 * np.array(NLL) + 2 * np.array(param)
    index = np.where(AIC == np.min(AIC))
    
    return index[0][0]

best_ODEs = {}

for i in range(num_species):
    if i == 0:
        equations, parameters = GP_models["NO_models", "NO_params"]
        rate_est = SR_derivatives_NO
        name = "NO"
    
    if i == 1:
        equations, parameters = GP_models["N_models", "N_params"]
        rate_est = SR_derivatives_N
        name = "N"
    
    if i == 2:
        equations, parameters = GP_models["O_models", "O_params"]
        rate_est = SR_derivatives_O
        name = "O"
        
    nll = []
    
    for equation in equations:
        rate_pred = predicting_rate(equation, sr_data)
        a = NLL_rates(rate_est, rate_pred, timesteps, num_exp)
        nll.append(a)
    
    best_ODEs[name] = best_rate_model(nll, parameters)

# In this second part, we read the symbolic expressions from the csv files, but now
# We make all possible combinations of ODEs from the proposed models and we evaluate
# Each of them and select the best one

# Here, we give make a function with a given ODE and we evaluated at a given initial condition
def rate_model(z0, equations, t, t_eval, event):
    i = 0

    for equation in equations:
        equation = str(equation)
        equation = equation.replace("CNO", "z[0]")
        equation = equation.replace("CN", "z[1]")
        equation = equation.replace("CO", "z[2]")
        equations[i] = equation
        i += 1

    def nest(t, z):
        dNOdt = eval(str(equations[0]))
        dNdt = (-1) * eval(str(equations[0]))
        dOdt = (-1/2) * eval(str(equations[0]))
        dzdt = [dNOdt, dNdt, dOdt]
        return dzdt

    sol = solve_ivp(nest, t, z0, t_eval = t_eval, method = "RK45", events = event)  

    return sol.y, sol.t, sol.status

equations = []
names = ["NO_models", "NO_params", "N_models", "N_params", "O_models", "O_params"]
all_models = []
params = []

# Here we make all the possible ODEs and save the number of parameters that exists in them
for i in np.arange(0, len(names), 2):
    all_models.append(GP_models[names[i], names[i + 1]][0])
    params.append(GP_models[names[i], names[i + 1]][1])

all_ODEs = list(it.product(*all_models))
param_ODEs = list(it.product(*params))

number_models = len(all_ODEs)
AIC_values = np.zeros(number_models)

# Here we evaluate the NLL for a given ODE and experiment
def NLL_kinetics(experiments, predictions, number_species, number_datapoints):
    output = np.zeros(number_species)
    mse = np.zeros(number_species)
    variance = np.zeros(number_species)

    for i in range(number_species):
        a = ((experiments[i] - predictions[i])**2)
        mse[i] = np.sum(a)
        variance[i] = mse[i] / (number_datapoints)

    for i in range(number_species):
        likelihood = ((experiments[i] - predictions[i])**2) / (2 * (variance[i])) \
            - np.log(1 / (np.sqrt(2 * np.pi * (variance[i]))))
        output[i] = np.sum(likelihood)

    return np.sum(output)


# Part of solve_ivp syntax - to make sure if the ODE takes longer than 5 seconds to solve
# It gets assigned a big ol' penalty
def my_event(t, y):
    time_out = perf_counter()

    if (time_out - time_in) > 2:
        return 0

    else:
        return 1

my_event.terminal = True

# Here we make all the possible ODEs and save the number of parameters that exists in them
# Evaluate over all possible models and experiments, save the NLL for each ODE system
all_ODEs = GP_models["NO_models", "NO_params"][0]
number_models = len(all_ODEs)
all_ODEs = [[x] for x in all_ODEs]
AIC_values = np.zeros(number_models)

for i in range(number_models):
    neg_log = 0
    print(i)

    for j in range(num_exp):
        t = time
        experiments = in_silico_data["exp_" + str(j + 1)]
        time_in = perf_counter()
        ics = initial_conditions["ic_" + str(j + 1)]
        y, tt, status = rate_model(ics, list(all_ODEs[i]), [0, np.max(t)], list(t), my_event)

        if status == -1:
            neg_log = 1e99
            break

        elif status == 1:
            neg_log = 1e99
            break

        else:
            neg_log += NLL_kinetics(experiments, y, num_species, timesteps)

    # num_parameters = np.sum(np.array(param_ODEs[i]))
    num_parameters = np.sum(np.array(params[0][i]))
    AIC_values[i] = 2 * neg_log + 2 * num_parameters

# Find the best model and plot it
best_model_index = np.argmin(AIC_values)
second_min_index = np.argpartition(AIC_values, 1)[1]
third_min_index = np.argpartition(AIC_values, 1)[2]

for i in range(num_exp):
    t = time
    time_in = perf_counter()
    ics = initial_conditions["ic_" + str(i + 1)]
    yy, tt, _ = rate_model(ics, list(all_ODEs[best_model_index]), [0, np.max(t)], list(t), my_event)

    fig, ax = plt.subplots()
    # ax.set_title("Experiment " + str(i + 1))
    ax.set_ylabel("Concentrations $(M)$", fontsize = 18)
    ax.set_xlabel("Time $(h)$", fontsize = 18)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)

    for j in range(num_species):
        y = in_silico_data["exp_" + str(i + 1)][j]
        ax.plot(t, y, "o", markersize = 4, label = species[j], color = color_1[j])
        ax.plot(tt, yy[j], color = color_1[j])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(alpha = 0.5)
    ax.legend(fontsize = 15)
    
    if i == 3:
        file_path = 'Physics-Informed_ADoK/Decomposition_of_Nitrous_Oxide_no_constraints/Graphs/PI_ADoK_rate_model_experiment_4.png'
        plt.savefig(file_path, dpi = 600, bbox_inches = "tight")


# plt.show()

print(all_ODEs[best_model_index])
print(all_ODEs[second_min_index])
print(all_ODEs[third_min_index])
print(np.argpartition(AIC_values, 1))
print(AIC_values)


