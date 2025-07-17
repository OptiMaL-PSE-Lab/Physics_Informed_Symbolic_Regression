#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 15:38:36 2025

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
    k_2 = 9 
    k_3 = 5
    dTdt = (-1) * ((k_1 * z[1] * z[0]) / (1 + k_2 * z[2] + k_3 * z[0]))
    dHdt = (-1) * ((k_1 * z[1] * z[0]) / (1 + k_2 * z[2] + k_3 * z[0]))
    dBdt = ((k_1 * z[1] * z[0]) / (1 + k_2 * z[2] + k_3 * z[0]))
    dMdt = ((k_1 * z[1] * z[0]) / (1 + k_2 * z[2] + k_3 * z[0]))
    dzdt = [dTdt, dHdt, dBdt, dMdt]
    return dzdt

# Plotting the data given
species = ['T', 'H', 'B', 'M']
initial_conditions = {
    "ic_1": np.array([1, 8, 2, 3]),
    "ic_2": np.array([5, 8, 0, 0.5]),
    "ic_3": np.array([5, 3, 0, 0.5]),
    "ic_4": np.array([1, 3, 0, 3]),
    "ic_5": np.array([1, 8, 2, 0.5]),
    
    "ic_6": np.array([5.,         6.95383183, 2.,         2.65975052]),
    "ic_7": np.array([5.,        8.,        0.696312, 3.       ])
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
    ax.set_ylabel("Concentrations $(M)$", fontsize = 18)
    ax.set_xlabel("Time $(h)$", fontsize = 18)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    
    for j in range(num_species):
        y = in_silico_data["exp_" + str(i + 1)][j]
        ax.plot(time, y, marker[j], markersize = 4, label = species[j], color = color_1[j])
    
    ax.grid(alpha = 0.5)
    ax.legend(loc='upper right', fontsize = 15)

# plt.show()

def save_matrix_as_csv(matrix, filename):
    # Convert numpy matrix to pandas dataframe
    df = pd.DataFrame(matrix)
        
    # Save dataframe as CSV file in exp_data directory without index
    filepath = os.path.join("Physics-Informed_ADoK/physics_informed_SR/Hydrodealkylation_of_Toluene/exp_data", filename + ".csv")
    df.to_csv(filepath, index = False, header = False)

for i in range(num_exp):
    name = "exp_" + str(i + 1)
    matrix = in_silico_data[name]
    save_matrix_as_csv(matrix, name)
