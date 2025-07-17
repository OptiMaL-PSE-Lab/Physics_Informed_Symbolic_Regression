#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:46:05 2023

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
noise = [np.random.normal(0, STD, size = (num_species, timesteps)) for i in range(num_exp)]
in_silico_data = {}
no_noise_data = {}

for i in range(num_exp):
    ic = initial_conditions["ic_" + str(i + 1)]
    solution = solve_ivp(kinetic_model, t, ic, t_eval = t_eval, method = "RK45")
    in_silico_data["exp_" + str(i + 1)] = np.clip(solution.y + noise[i], 0, 1e99)
    no_noise_data["exp_" + str(i + 1)] = solution.y

color_1 = ['salmon', 'royalblue', 'darkviolet']
marker = ['o', 'o', 'o', 'o']

# Plotting the in-silico data for visualisation
for i in range(num_exp):
    fig, ax = plt.subplots()
    ax.set_title("Experiment " + str(i + 1))
    ax.set_ylabel("Concentration $(M)$")
    ax.set_xlabel("Time $(h)$")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for j in range(num_species):
        y = in_silico_data["exp_" + str(i + 1)][j]
        ax.plot(time, y, marker[j], markersize = 3, label = species[j], color = color_1[j])

    ax.grid(alpha = 0.5)
    ax.legend()
        
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