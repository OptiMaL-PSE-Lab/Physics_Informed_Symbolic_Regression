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
from scipy.optimize import minimize
from beecolpy import abc
import os
np.random.seed(1998)


"##############################################################################"
"############################ Optimise Rate Model #############################"
"##############################################################################"

import re

def change_numbers(string):
    # Keep track of the current k_i variable to use
    k_count = 1
    
    # Use regular expressions to find decimal numbers in the string
    pattern = r'-?\d+\.\d+'
    matches = re.findall(pattern, string)

    # Replace each decimal number with a k_i variable in the order they appear
    for match in matches:
        k_var = "k_" + str(k_count)
        # Escape any regex special characters in the match
        escaped_match = re.escape(match)
        string = re.sub(escaped_match, k_var, string, count=1) # use count=1 to replace only the first occurrence
        k_count += 1

    # Now, replace T, H, B, and M with the appropriate z[i] terms
    string = string.replace('CNO', 'z[0]')
    string = string.replace('N', 'z[1]')
    string = string.replace('H', 'z[2]')

    return '(' + string + ')'
    
a = change_numbers('-0.3916939*CNO')
b = change_numbers('-CNO/(1.9231634*CNO - 0.01950799)')
print(b)

def competition(k, z0):
    k_1 = k[0]
    k_2 = k[1]
    k_3 = k[2]
    # k_4 = k[3]
    # k_5 = k[4]

    def nest(t, z):
        dNOdt = (-1) * ((k_1 * z[0]**2 - k_2 *z[0]) / (1 + k_3 * z[0]))
        dNdt = ((k_1 * z[0]**2 - k_2 *z[0]) / (1 + k_3 * z[0]))
        dOdt = (1/2) * ((k_1 * z[0]**2 - k_2 *z[0]) / (1 + k_3 * z[0]))

        dzdt = [dNOdt, dNdt, dOdt]
        return dzdt

        
    # time points
    time = np.linspace(0, 10, 15)
    t = [0, np.max(time)]
    t_eval = list(time)
    
    # solve ODE
    sol = solve_ivp(nest, t, z0, t_eval = t_eval, method = "RK45")
    
    return sol.y

# Function that reads files from a directory and returns a dictionary
def read_files(directory):
    files = os.listdir(directory)
    data = {}
    files.remove(".DS_Store")

    for file in files:
        data[file[:-4]] = pd.read_csv(directory + '/' + file, header = None).values

    return data

in_silico_data = read_files("Physics-Informed_ADoK/physics_informed_SR/Decomposition_Nitrous_Oxide/exp_data")

# This takes the first column from each entry of the dictionary and puts it into another dictionary
initial_conditions = {}
for key, value in in_silico_data.items():
    aa = "ic_" + key[4:]
    initial_conditions[aa] = value[:, 0]

def nll(params, model):
    num_exp = len(initial_conditions)

    for i in range(num_exp):
        ic = initial_conditions["ic_" + str(i+1)]
        observations = in_silico_data["exp_" + str(i + 1)]
        model_response = model(params, ic)

        SSE = (observations - model_response)**2
        shape_data = np.shape(in_silico_data["exp_" + str(i + 1)])
        num_species = shape_data[0]
        num_datapoints = shape_data[1]
        variance = np.sum(SSE) / (num_species * num_datapoints)

        placeholder = (SSE / (2 * variance)) - np.log(1 / (np.sqrt(2 * np.pi * (variance))))
        likelihood = np.sum(placeholder)

    return likelihood

def sse(params):
    num_exp = len(initial_conditions)
    total = np.zeros((num_exp, 1))

    for i in range(num_exp):
        ic = initial_conditions["ic_" + str(i+1)]
        observations = in_silico_data["exp_" + str(i + 1)]
        model_response = competition(params, ic)

        SSE = (observations - model_response)**2
        total[i] = np.sum(SSE)

    return np.sum(total)

def callback(xk):
    # Print out the current solution
    print(f"Current solution: {xk}")

def Opt_Rout(multistart, number_parameters, x0, lower_bound, upper_bound, to_opt):
    localsol = np.empty([multistart, number_parameters])
    localval = np.empty([multistart, 1])
    boundss = tuple([(lower_bound, upper_bound) for i in range(number_parameters)])
    
    for i in range(multistart):
        res = minimize(to_opt, x0, method = 'L-BFGS-B', \
                       bounds = boundss, callback = callback)
        localsol[i] = res.x
        localval[i] = res.fun

    minindex = np.argmin(localval)
    opt_val = localval[minindex]
    opt_param = localsol[minindex]
    
    return opt_val, opt_param

multistart = 10
number_parameters = 3
lower_bound = 0.0001
upper_bound = 10

# abc_obj = abc(sse, [(lower_bound, upper_bound) for i in range(number_parameters)])
# abc_obj.fit() 

# solution = abc_obj.get_solution()
# print('Initial guess = ', solution)

solution = np.array([2,0.001,5])

opt_val, opt_param = Opt_Rout(multistart, number_parameters, solution, lower_bound, \
    upper_bound, sse)

print('MSE = ', opt_val)
print('Optimal parameters = ', opt_param)
